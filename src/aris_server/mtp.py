"""MTP (multi-token prediction) speculative decoding for Qwen 3.5-architecture models (Qwen3.8-27B, Aris 3.8-27B).

Qwen3.8-27B ships one extra decoder layer, the MTP head. It reads the trunk's last hidden state at position t and the
embedding of token t+1, and predicts token t+2. Run on its own output, it drafts further ahead. The trunk then checks
the drafts in one forward pass: a decode step is bound by reading the weights, so checking k+1 tokens costs about as
much as generating one, and every accepted draft is a token for free.

The trunk is a hybrid. Its full-attention layers keep a KV cache, which can simply be trimmed when drafts are
rejected. Its Gated DeltaNet layers keep a recurrent state, which can't. During a verify pass those layers compute
their projections for every position at once (the costly part) but run the cheap recurrence one position at a time,
saving the state after each, so a rejection restores the state at the last accepted position exactly.

Drafts are greedy, so the draft distribution is a point mass and speculative sampling stays exact: a draft is
accepted with the target's probability of it, and a rejection resamples from the target with the draft removed. At
temperature 0 that means: accept while a draft equals the target's argmax. Batched verify math rounds differently
from one-token decode math, so greedy output can drift from plain decoding at near-ties; the distribution doesn't.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

# Qwen stores these RMSNorm weights zero-centered (applied as 1 + w); mlx-lm's RMSNorm applies w as is.
_ZERO_CENTERED = (
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


class MTPHead(nn.Module):
    """Qwen's MTP layer: norm(layer(fc([norm(embed(token t+1)), norm(hidden t)]))), shared embeddings and lm_head."""

    def __init__(self, args: qwen3_5.TextModelArgs):
        super().__init__()
        h, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = [qwen3_5.DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(h, eps=eps)

    def __call__(self, embeddings: mx.array, hidden: mx.array, cache: Optional[KVCache] = None) -> mx.array:
        x = mx.concatenate([self.pre_fc_norm_embedding(embeddings), self.pre_fc_norm_hidden(hidden)], axis=-1)
        x = self.fc(x)
        x = self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)
        return self.norm(x)

    @staticmethod
    def make_cache() -> KVCache:
        return KVCache()


def load_mtp_head(path: str | Path, args: qwen3_5.TextModelArgs, *, bits: int | None = None,
                  group_size: int = 64) -> MTPHead:
    """Load an MTP head from a safetensors file holding `mtp.*` tensors.

    Files cut straight from a Qwen checkpoint have zero-centered norms and get 1 added; a file whose metadata says
    `mtp_format: mlx` is taken as is. `bits` quantizes the head's linear layers like the trunk's."""
    arrays, meta = mx.load(str(path), return_metadata=True)
    raw = (meta or {}).get("mtp_format") != "mlx"
    weights = {}
    for name, value in arrays.items():
        name = name.removeprefix("mtp.")
        if raw and value.ndim == 1 and name.endswith(_ZERO_CENTERED):
            value = value + 1.0
        weights[name] = value
    head = MTPHead(args)
    head.load_weights(list(weights.items()), strict=True)
    if bits:
        nn.quantize(head, group_size=group_size, bits=bits)
    mx.eval(head.parameters())
    return head


# --- Verify passes on the hybrid trunk ---------------------------------------------------------------------------

@dataclass
class _Capture:
    """What a Gated DeltaNet layer records during a verify pass, to restore its state at any accepted position."""

    conv_input: mx.array | None = None  # the old conv window followed by this pass's inputs
    states: list = field(default_factory=list)  # recurrent state after each position


_original_gdn_call = qwen3_5.GatedDeltaNet.__call__


def _gdn_call(self, inputs, mask=None, cache=None):
    capture = getattr(cache, "mtp_capture", None) if cache is not None else None
    if capture is None:
        return _original_gdn_call(self, inputs, mask, cache)
    return _gdn_capture_call(self, inputs, mask, cache, capture)


qwen3_5.GatedDeltaNet.__call__ = _gdn_call


def _gdn_capture_call(self, inputs, mask, cache, capture: _Capture):
    """GatedDeltaNet.__call__ with the recurrence run one position at a time, saving each state (B = 1, no padding)."""
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    conv_state = cache[0] if cache[0] is not None else mx.zeros(
        (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(self.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)

    state = cache[1]
    outputs, states = [], []
    for i in range(S):
        step_mask = None if mask is None else mask[:, i:i + 1]
        out, state = gated_delta_update(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], a[:, i:i + 1], b[:, i:i + 1],
                                        self.A_log, self.dt_bias, state, step_mask, use_kernel=not self.training)
        outputs.append(out)
        states.append(state)

    capture.conv_input = conv_input
    capture.states = states
    cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
    cache[1] = state
    cache.advance(S)

    out = self.norm(mx.concatenate(outputs, axis=1), z)
    return self.out_proj(out.reshape(B, S, -1))


def text_model(model) -> nn.Module:
    """The language model inside a (possibly multimodal) Qwen 3.5 model."""
    return getattr(model, "language_model", model)


def trunk_hidden(model, tokens: mx.array, cache) -> mx.array:
    """The trunk's last hidden states (after its final norm) for `tokens` of shape (1, S)."""
    return text_model(model).model(tokens, cache)


def lm_logits(model, hidden: mx.array) -> mx.array:
    text = text_model(model)
    if text.args.tie_word_embeddings:
        return text.model.embed_tokens.as_linear(hidden)
    return text.lm_head(hidden)


def embed(model, tokens: mx.array) -> mx.array:
    return text_model(model).model.embed_tokens(tokens)


def draft_lm_head(model, bits: int | None = None, group_size: int = 64,
                  rows_per_chunk: int = 16384) -> Callable[[mx.array], mx.array]:
    """The output layer drafts use: the model's own, or a copy quantized to `bits`. A draft is only a guess the trunk
    checks, so a cheaper copy changes how often drafts are accepted, never the output. At batch size 1 a draft step
    reads the whole output layer (1.27 GB for Aris 3.8-27B at 8 bits): at 4 bits it reads half. The copy is built a
    block of rows at a time, so it never holds the layer at full precision."""
    if not bits:
        return lambda hidden: lm_logits(model, hidden)
    text = text_model(model)
    layer = text.model.embed_tokens if text.args.tie_word_embeddings else text.lm_head
    parts = []
    for start in range(0, layer.weight.shape[0], rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        if hasattr(layer, "scales"):
            dense = mx.dequantize(layer.weight[rows], layer.scales[rows], layer.biases[rows],
                                  group_size=layer.group_size, bits=layer.bits)
        else:
            dense = layer.weight[rows]
        part = mx.quantize(dense, group_size=group_size, bits=bits)
        mx.eval(part)
        parts.append(part)
    head = nn.QuantizedLinear(group_size, 1, bias=False, group_size=group_size, bits=bits)  # shapes replaced below
    head.weight, head.scales, head.biases = (mx.concatenate([p[i] for p in parts]) for i in range(3))
    mx.eval(head.parameters())
    return head


class _NormTap(nn.Module):
    """The trunk's final norm, also keeping what went into it: the hidden state before that norm."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.inner = norm
        self.last = None

    def __call__(self, x: mx.array) -> mx.array:
        self.last = x
        return self.inner(x)


def verify_forward(model, tokens: Sequence[int], cache) -> tuple[mx.array, list]:
    """Run the trunk over `tokens` (the newest committed token, then the drafts), recording what `commit` needs."""
    captures = []
    for c in cache:
        if isinstance(c, ArraysCache):
            c.mtp_capture = _Capture()
            captures.append(c)
    try:
        hidden = trunk_hidden(model, mx.array([list(tokens)]), cache)
        mx.eval(hidden)
    finally:
        for c in captures:
            c.mtp_capture_last = c.mtp_capture
            c.mtp_capture = None
    return hidden, captures


def commit(cache, length: int, kept: int) -> None:
    """After a verify pass over `length` tokens, keep only the first `kept` of them in every trunk cache."""
    drop = length - kept
    for c in cache:
        if isinstance(c, ArraysCache):
            capture = c.mtp_capture_last
            c.mtp_capture_last = None
            if drop == 0:
                continue
            n_keep = capture.conv_input.shape[1] - length
            c[0] = mx.contiguous(capture.conv_input[:, kept:kept + n_keep, :])
            c[1] = capture.states[kept - 1]
        elif drop:
            c.trim(drop)


# --- Sampling ------------------------------------------------------------------------------------------------------

Processor = Callable[[mx.array, mx.array], mx.array]


@dataclass
class Sampling:
    """The target distribution, built exactly as mlx-lm's `make_sampler` builds it: the filters (top-p, min-p,
    top-k, in that order) act on log-probabilities at temperature 1, then the temperature applies. 0 means greedy."""

    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0

    def probs(self, logits: mx.array) -> mx.array:
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if 0 < self.top_p < 1.0:
            logprobs = apply_top_p(logprobs, self.top_p)
        if self.min_p != 0.0:
            logprobs = apply_min_p(logprobs, self.min_p, 1)
        if self.top_k > 0:
            logprobs = apply_top_k(logprobs, self.top_k)
        return mx.softmax(logprobs / self.temperature, axis=-1)


def _choose(logits: mx.array, draft: int | None, sampling: Sampling, key) -> tuple[int, bool]:
    """Pick the token for one verify row (`logits` is (1, vocab)). Returns (token, whether it is the draft).

    A greedy draft is a point mass, so exact speculative sampling accepts it with the target's probability of it,
    and a rejection samples the target with the draft taken out."""
    if sampling.temperature == 0:
        token = int(mx.argmax(logits, axis=-1).item())
        return token, token == draft
    probs = sampling.probs(logits)[0]
    k_accept, k_sample = mx.random.split(key)
    if draft is not None:
        if mx.random.uniform(key=k_accept).item() < probs[draft].item():
            return draft, True
        probs = probs.at[draft].multiply(0.0)
        probs = probs / probs.sum()
    token = int(mx.random.categorical(mx.log(probs), key=k_sample).item())
    return token, False


# --- Generation ----------------------------------------------------------------------------------------------------

@dataclass
class MTPStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    accepted_at: list = field(default_factory=list)  # accepted[i] counts rounds where draft i+1 was accepted
    log_drafts: bool = False  # keep (tokens generated so far, drafts) for every round, for tests and diagnosis
    drafts_log: list = field(default_factory=list)
    prefill_seconds: float = 0.0  # reading the prompt, through the first token
    decode_seconds: float = 0.0  # everything after the first token

    @property
    def decode_tok_s(self) -> float:
        generated = self.accepted + self.rounds  # tokens produced after the first one
        return generated / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


Drafter = Callable[[int], list]


def generate_mtp(model, head: MTPHead | None, prompt: Sequence[int], *, max_tokens: int, depth: int = 1,
                 sampling: Sampling | None = None, processors: Sequence[Processor] = (), seed: int = 0,
                 stop_tokens: Sequence[int] = (), prefill_step: int = 2048, drafter: Drafter | None = None,
                 stats: MTPStats | None = None, draft_logits: Callable[[mx.array], mx.array] | None = None,
                 head_hidden: str = "post") -> list[int]:
    """Generate up to `max_tokens` tokens after `prompt` (batch size 1) with MTP drafts checked by the trunk.

    `drafter` replaces the MTP head in tests: it receives how many tokens have been generated and returns drafts.
    `draft_logits` is the output layer drafts use (draft_lm_head); the trunk always checks with its own.
    `head_hidden` is which trunk hidden state the head reads: "post" (after the trunk's final norm) or "pre"."""
    if head_hidden not in ("post", "pre"):
        raise ValueError(f"head_hidden must be 'post' or 'pre', not {head_hidden!r}")
    text = text_model(model)
    tap = None
    if head_hidden == "pre" and head is not None:
        tap = _NormTap(text.model.norm)
        text.model.norm = tap
    try:
        return _generate_mtp(model, head, prompt, max_tokens=max_tokens, depth=depth, sampling=sampling,
                             processors=processors, seed=seed, stop_tokens=stop_tokens, prefill_step=prefill_step,
                             drafter=drafter, stats=stats,
                             draft_logits=draft_logits or (lambda hidden: lm_logits(model, hidden)),
                             rows_of=(lambda hidden: tap.last) if tap is not None else (lambda hidden: hidden))
    finally:
        if tap is not None:
            text.model.norm = tap.inner


def _generate_mtp(model, head, prompt, *, max_tokens, depth, sampling, processors, seed, stop_tokens, prefill_step,
                  drafter, stats, draft_logits, rows_of) -> list[int]:
    sampling = sampling or Sampling()
    stats = stats if stats is not None else MTPStats()
    started = time.perf_counter()
    key = mx.random.key(seed)
    cache = make_prompt_cache(model)
    mtp_cache = head.make_cache() if head is not None else None
    prompt = list(prompt)
    history = mx.array(prompt)

    # Prefill: the trunk reads the prompt, and the head reads (hidden t, token t+1) for every prompt position.
    hidden = None
    for start in range(0, len(prompt), prefill_step):
        chunk = prompt[start:start + prefill_step]
        hidden = trunk_hidden(model, mx.array([chunk]), cache)
        rows = rows_of(hidden)
        if head is not None:
            nxt = prompt[start + 1:start + 1 + len(chunk)]
            if nxt:
                head(embed(model, mx.array([nxt])), rows[:, :len(nxt)], mtp_cache)
        mx.eval(hidden, *[c.state for c in cache], *([mtp_cache.state] if head is not None else []))
    last_hidden = hidden[:, -1:]

    logits = lm_logits(model, last_hidden)[:, -1]
    for p in processors:
        logits = p(history, logits)
    key, sub = mx.random.split(key)
    token, _ = _choose(logits, None, sampling, sub)
    out = [token]
    history = mx.concatenate([history, mx.array([token])])
    rows_hidden, rows_tokens = rows[:, -1:], [token]
    first_token_at = time.perf_counter()
    stats.prefill_seconds = first_token_at - started

    while len(out) < max_tokens and token not in stop_tokens:
        # Draft: add the newest true rows to the head's cache, then chain the head on its own output.
        if drafter is not None:
            drafts = list(drafter(len(out)))[:depth]
        else:
            h = head(embed(model, mx.array([rows_tokens])), rows_hidden, mtp_cache)[:, -1:]
            drafts = []
            for i in range(depth):
                d = int(mx.argmax(draft_logits(h)[0, -1]).item())
                drafts.append(d)
                if i + 1 < depth:
                    h = head(embed(model, mx.array([[d]])), h, mtp_cache)
        drafts = drafts[:max(0, max_tokens - len(out))]
        if stats.log_drafts:
            stats.drafts_log.append((len(out), list(drafts)))

        # Verify: the trunk reads the newest token and the drafts in one pass.
        verify_tokens = [token, *drafts]
        hidden, _ = verify_forward(model, verify_tokens, cache)
        rows = rows_of(hidden)
        rows_logits = lm_logits(model, hidden)[0]
        new, accepted_now = [], 0
        for i in range(len(verify_tokens)):
            logits = rows_logits[i:i + 1]
            for p in processors:
                logits = p(history, logits)
            key, sub = mx.random.split(key)
            draft = drafts[i] if i < len(drafts) else None
            chosen, accepted = _choose(logits, draft, sampling, sub)
            new.append(chosen)
            history = mx.concatenate([history, mx.array([chosen])])
            accepted_now += accepted
            if not accepted or chosen in stop_tokens:
                break
        # The trunk keeps the newest token and each accepted draft it read; the last chosen token is read next round.
        kept = len(new)
        commit(cache, len(verify_tokens), kept)
        if head is not None and depth > 1:
            mtp_cache.trim(depth - 1)

        stats.rounds += 1
        stats.drafted += len(drafts)
        stats.accepted += accepted_now
        while len(stats.accepted_at) < len(drafts):
            stats.accepted_at.append(0)
        for i in range(accepted_now):
            stats.accepted_at[i] += 1

        out.extend(new)
        token = new[-1]
        rows_hidden = rows[:, :kept]
        rows_tokens = new
    stats.decode_seconds = time.perf_counter() - first_token_at
    return out[:max_tokens]
