"""MTP (multi-token prediction) speculative decoding for Qwen 3.5-architecture models (Qwen3.8-27B, Aris 3.8-27B).

Qwen3.8-27B ships one extra decoder layer, the MTP head. It reads the trunk's last hidden state at position t and the
embedding of token t+1, and predicts token t+2. Run on its own output, it drafts further ahead. The trunk then checks
the drafts in one forward pass: a decode step is bound by reading the weights, so checking k+1 tokens costs about as
much as generating one, and every accepted draft is a token for free.

The trunk is a hybrid. Its full-attention layers keep a KV cache, which can simply be trimmed when drafts are
rejected. Its Gated DeltaNet layers keep a recurrent state, which can't. During a verify pass those layers compute
their projections for every position at once (the costly part) but run the cheap recurrence one position at a time,
saving the state after each, so a rejection restores the state at the last accepted position exactly.

Drafts are greedy, so speculative sampling stays exact: the loop and its acceptance rule are speculative.py's,
shared with Inkling's draft head (dspark.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from .speculative import DraftHook, Processor, Sampling, SpeculativeStats, speculate

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


def verify_forward(model, tokens: Sequence[int] | mx.array, cache) -> tuple[mx.array, list]:
    """Run the trunk over `tokens` (the newest committed token, then the drafts), recording what `commit` needs."""
    captures = []
    for c in cache:
        if isinstance(c, ArraysCache):
            c.mtp_capture = _Capture()
            captures.append(c)
    try:
        tokens = tokens if isinstance(tokens, mx.array) else mx.array(list(tokens))
        hidden = trunk_hidden(model, tokens[None], cache)
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


# --- Generation ----------------------------------------------------------------------------------------------------

class _MTPSpeculator:
    """The `Speculator` (speculative.py) for a Qwen 3.5-architecture trunk and its MTP head, or None when drafts come
    from the loop's test drafter. The head reads (hidden t, token t+1) for every committed position, then drafts by
    chaining on its own output; those chained rows leave its cache at commit."""

    def __init__(self, model, head: MTPHead | None, draft_logits: Callable[[mx.array], mx.array],
                 rows_of: Callable[[mx.array], mx.array]):
        self.model, self.head, self.draft_logits, self.rows_of = model, head, draft_logits, rows_of
        self.cache = make_prompt_cache(model)
        self.head_cache = head.make_cache() if head is not None else None
        # Committed rows the head hasn't read: their hidden states, and the tokens after them but the newest, which
        # draft() receives.
        self.rows_hidden, self.rows_tokens = None, []
        self._chained = 0
        self._verified = None

    def prefill(self, prompt: Sequence[int], step: int) -> mx.array:
        hidden = rows = None
        for start in range(0, len(prompt), step):
            chunk = list(prompt[start:start + step])
            hidden = trunk_hidden(self.model, mx.array([chunk]), self.cache)
            rows = self.rows_of(hidden)
            if self.head is not None:
                nxt = list(prompt[start + 1:start + 1 + len(chunk)])
                if nxt:
                    self.head(embed(self.model, mx.array([nxt])), rows[:, :len(nxt)], self.head_cache)
            mx.eval(hidden, *[c.state for c in self.cache],
                    *([self.head_cache.state] if self.head is not None else []))
        self.rows_hidden = rows[:, -1:]
        return lm_logits(self.model, hidden[:, -1:])[:, -1]

    def draft(self, token: int, depth: int) -> list[int]:
        if self.head is None:
            return []
        # The newest true rows first, even when nothing is drafted, so the head's cache never falls behind.
        tokens = self.rows_tokens + [token]
        h = self.head(embed(self.model, mx.array([tokens])), self.rows_hidden, self.head_cache)[:, -1:]
        self.rows_tokens, self._chained = [], 0
        drafts = []
        for i in range(depth):
            drafts.append(int(mx.argmax(self.draft_logits(h)[0, -1]).item()))
            if i + 1 < depth:
                h = self.head(embed(self.model, mx.array([[drafts[-1]]])), h, self.head_cache)
                self._chained += 1
        return drafts

    def verify(self, tokens: mx.array) -> mx.array:
        hidden, _ = verify_forward(self.model, tokens, self.cache)
        self._verified = (tokens, self.rows_of(hidden))
        return lm_logits(self.model, hidden)[0]

    def commit(self, total: int, kept: int) -> None:
        commit(self.cache, total, kept)
        if self._chained:
            self.head_cache.trim(self._chained)
            self._chained = 0
        tokens, rows = self._verified
        self._verified = None
        self.rows_hidden = rows[:, :kept]
        self.rows_tokens = tokens[1:kept].tolist()


def generate_mtp(model, head: MTPHead | None, prompt: Sequence[int], *, max_tokens: int, depth: int = 1,
                 sampling: Sampling | None = None, processors: Sequence[Processor] = (), seed: int = 0,
                 stop_tokens: Sequence[int] = (), prefill_step: int = 2048, drafter: DraftHook | None = None,
                 stats: SpeculativeStats | None = None, draft_logits: Callable[[mx.array], mx.array] | None = None,
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
        spec = _MTPSpeculator(model, head, draft_logits or (lambda hidden: lm_logits(model, hidden)),
                              (lambda hidden: tap.last) if tap is not None else (lambda hidden: hidden))
        return speculate(spec, prompt, max_tokens=max_tokens, depth=depth, sampling=sampling, processors=processors,
                         seed=seed, stop_tokens=stop_tokens, prefill_step=prefill_step, drafter=drafter, stats=stats)
    finally:
        if tap is not None:
            text.model.norm = tap.inner
