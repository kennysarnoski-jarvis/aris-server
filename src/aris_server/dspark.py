"""Inkling 2's own draft head (DSpark), for speculative decoding.

Inkling 2 checkpoints ship a small drafter, `mtp.safetensors`, configured by `mtp_config` in config.json (0.3B
parameters for Inkling 2 Mini). It reads the target's hidden states at a few layer visits (`target_layer_ids`:
the inputs of some visits, loop passes counted, and last the final normed output), each drafter layer mixing them
into the keys and values of its attention context. Given the newest chosen token (the anchor), it fills a block of
`block_size` positions in one pass, the anchor and then mask embeddings, each row predicting the token at its own
position. A small recurrent head (`rnn_hidden`) then turns the rows into a chain of guesses, each conditioned on the
guess before it. The target checks the guesses in one forward pass.

Every state the target keeps rewinds exactly after a rejection:
- full-attention visits keep a KVCache, which trims;
- sliding-window visits keep a RotatingKVCache, and a multi-token forward leaves it in temporal order with the rows
  it read at the end (`_update_concat`), so dropping the rejected rows from the end restores it;
- each visit's four short convolutions keep their last K-1 inputs; the verify forward hands each whole input window
  to the cache (`windows`), and the state after any accepted length is a slice of it.

The drafter's architecture and numerics (where values round to bf16, and how a guess is scored) follow vLLM's
Inkling2 DSpark draft, vllm/models/inkling2/nvidia/dspark.py and its kernels in .../ops/dspark.py (Apache License
2.0, Copyright contributors to the vLLM project), translated from PyTorch and Triton to MLX. Changes: one sequence
instead of a batch, no tensor parallelism or paged cache, and the head is folded in float32, not float64.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

from .speculative import DraftHook, Processor, Sampling, SpeculativeStats, speculate

_F32 = mx.float32


@dataclass(frozen=True)
class DraftConfig:
    """`mtp_config` of an Inkling 2 checkpoint, checked against its target as vLLM checks it."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: tuple[int, ...]
    vocab_size: int  # the target's real vocabulary: guesses are taken from it
    padded_vocab_size: int  # the target's embedding rows, which the recurrent head's token table matches
    final_logit_softcapping: float

    @classmethod
    def from_target(cls, config: dict, text) -> "DraftConfig":
        """`config` is the checkpoint's config.json; `text` its TextArgs. Raises ValueError when it has no drafter
        or one this code doesn't run."""
        draft = config.get("mtp_config")
        if not draft:
            raise ValueError("This Inkling 2 checkpoint has no mtp_config (no draft head)")
        visits = len(text.loop_core) * (text.loop_config["num_passes"] - 1) + text.num_hidden_layers \
            if text.loop_config else text.num_hidden_layers
        taps = list(draft.get("target_layer_ids", []))
        if len(taps) < 2 or taps != sorted(set(taps)) or taps[-1] != visits:
            raise ValueError("mtp_config target_layer_ids must be sorted layer visits followed by the number of "
                             f"visits ({visits}, the final normed output); got {taps}")
        if draft["hidden_size"] != text.hidden_size or draft.get("markov_head_type") != "rnn_hidden" \
                or draft.get("q_bias") or draft.get("o_bias"):
            raise ValueError("Unsupported Inkling 2 mtp_config")
        return cls(hidden_size=draft["hidden_size"], num_hidden_layers=draft["num_hidden_layers"],
                   num_attention_heads=draft["num_attention_heads"],
                   num_key_value_heads=draft["num_key_value_heads"], head_dim=draft["head_dim"],
                   intermediate_size=draft["intermediate_size"], rms_norm_eps=draft["rms_norm_eps"],
                   rope_theta=draft["rope_theta"], local_layer_ids=tuple(draft.get("local_layer_ids", ())),
                   sliding_window_size=draft["sliding_window_size"], block_size=draft["block_size"],
                   markov_rank=draft["markov_rank"], target_layer_ids=tuple(taps),
                   vocab_size=text.unpadded_vocab_size, padded_vocab_size=text.vocab_size,
                   final_logit_softcapping=text.final_logit_softcapping)


def _rms(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm in float32 math, returned in float32."""
    return mx.fast.rms_norm(x.astype(_F32), weight.astype(_F32), eps)


def _add_norm(x: mx.array, residual: mx.array, norm: nn.RMSNorm) -> tuple[mx.array, mx.array]:
    """FlashInfer's fused add + RMSNorm, as vLLM's draft uses it: the residual takes x (rounded to its dtype), and
    the norm reads the unrounded sum. Returns (residual, normed)."""
    total = x.astype(_F32) + residual.astype(_F32)
    return total.astype(residual.dtype), _rms(total, norm.weight, norm.eps).astype(x.dtype)


class _Context:
    """One drafter layer's attention context: keys and values of committed positions, each rotated at its own
    position. A local layer reads only the last `window` positions, so it keeps only those."""

    def __init__(self, window: int | None):
        self.window = window
        self.offset = 0
        self._full = KVCache() if window is None else None
        self.keys = self.values = None

    def append(self, keys: mx.array, values: mx.array) -> None:
        self.offset += keys.shape[2]
        if self._full is not None:
            self._full.update_and_fetch(keys, values)
            return
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys, self.values = keys[:, :, -self.window:], values[:, :, -self.window:]

    def read(self) -> tuple[mx.array | None, mx.array | None]:
        if self._full is not None:
            if self._full.keys is None:
                return None, None
            return self._full.keys_and_values()
        return self.keys, self.values

    def arrays(self) -> list[mx.array]:
        return [a for a in self.read() if a is not None]


class DraftCache:
    """What the drafter keeps between rounds: each layer's context, and the recurrent head's input from the newest
    committed position."""

    def __init__(self, config: DraftConfig):
        self.contexts = [_Context(config.sliding_window_size if i in config.local_layer_ids else None)
                         for i in range(config.num_hidden_layers)]
        self.f_rnn: mx.array | None = None

    @property
    def offset(self) -> int:
        return self.contexts[0].offset

    def arrays(self) -> list[mx.array]:
        out = [a for c in self.contexts for a in c.arrays()]
        return out + ([self.f_rnn] if self.f_rnn is not None else [])


def _heads(x: mx.array, head_dim: int) -> mx.array:
    B, L, _ = x.shape
    return x.reshape(B, L, -1, head_dim).transpose(0, 2, 1, 3)


class DraftAttention(nn.Module):
    """Bidirectional attention of the block over itself and the context (a local layer's context is the
    `sliding_window_size` positions before the block)."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        d, h, kv, hd = config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.head_dim, self.rope_theta = hd, config.rope_theta
        self.q_proj = nn.Linear(d, h * hd, bias=False)
        self.k_proj = nn.Linear(d, kv * hd, bias=False)
        self.v_proj = nn.Linear(d, kv * hd, bias=False)
        self.o_proj = nn.Linear(h * hd, d, bias=False)
        self.k_proj_ctx = nn.Linear(d, kv * hd, bias=False)
        self.v_proj_ctx = nn.Linear(d, kv * hd, bias=False)
        self.q_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(hd, eps=config.rms_norm_eps)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        # vLLM's default (neox) rotary embedding: rotate halves, MLX's traditional=False.
        return mx.fast.rope(x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)

    def context(self, features: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        """Keys and values of mixed target features at positions offset, offset + 1, ..."""
        keys = self._rope(self.k_norm_ctx(_heads(self.k_proj_ctx(features), self.head_dim)), offset)
        return keys, _heads(self.v_proj_ctx(features), self.head_dim)

    def __call__(self, x: mx.array, context: _Context) -> mx.array:
        B, L, _ = x.shape
        offset = context.offset  # the block's first position, the anchor's
        q = self._rope(self.q_norm(_heads(self.q_proj(x), self.head_dim)), offset)
        k = self._rope(self.k_norm(_heads(self.k_proj(x), self.head_dim)), offset)
        v = _heads(self.v_proj(x), self.head_dim)
        ctx_k, ctx_v = context.read()
        if ctx_k is not None:
            k = mx.concatenate([ctx_k, k], axis=2)
            v = mx.concatenate([ctx_v, v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DraftMLP(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        d, ff = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = self.gate_proj(x).astype(_F32), self.up_proj(x).astype(_F32)
        return self.down_proj((nn.silu(gate) * up).astype(x.dtype))  # silu(gate) * up in float32, rounded once


class DraftLayer(nn.Module):
    def __init__(self, config: DraftConfig, local: bool):
        super().__init__()
        d, eps = config.hidden_size, config.rms_norm_eps
        self.window = config.sliding_window_size if local else None
        self.input_layernorm = nn.RMSNorm(d, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=eps)
        self.dflare_norm = nn.RMSNorm(d, eps=eps)  # normalizes this layer's mix of the target features
        self.self_attn = DraftAttention(config)
        self.mlp = DraftMLP(config)

    def observe(self, features: mx.array, context: _Context) -> None:
        """Add mixed target features of newly committed positions to the context."""
        if self.window is not None and features.shape[1] > self.window:
            skipped = features.shape[1] - self.window  # a local layer never reads them
            context.offset += skipped
            features = features[:, -self.window:]
        context.append(*self.self_attn.context(features, context.offset))

    def __call__(self, x: mx.array, residual: mx.array | None, context: _Context) -> tuple[mx.array, mx.array]:
        """Returns the MLP output and the residual, which the next layer (or the final norm) adds."""
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            residual, h = _add_norm(x, residual, self.input_layernorm)
        h = self.self_attn(h, context)
        residual, h = _add_norm(h, residual, self.post_attention_layernorm)
        return self.mlp(h), residual


class MarkovHead(nn.Module):
    """The recurrent (`rnn_hidden`) correction of the block's base logits: a rank-r state updated by each guess.
    Its output goes through `hidden_out` and the target's output layer, folded once into (vocab, r) and (vocab,)."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        d, r = config.hidden_size, config.markov_rank
        self.rank = r
        self.dflare_norm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.markov_w1 = nn.Embedding(config.padded_vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * d, 3 * r, bias=True)  # columns: [state, previous token, row, f_rnn]
        self.hidden_out = nn.Linear(r, d, bias=True)


def _output_rows(lm_head: nn.Module, start: int, end: int) -> mx.array:
    """Rows start:end of the target's output layer weight, dequantized if it is quantized."""
    if isinstance(lm_head, nn.QuantizedLinear):
        biases = lm_head.get("biases")
        return mx.dequantize(lm_head.weight[start:end], lm_head.scales[start:end],
                             None if biases is None else biases[start:end],
                             group_size=lm_head.group_size, bits=lm_head.bits, mode=lm_head.mode)
    return lm_head.weight[start:end]


class Drafter(nn.Module):
    """Inkling 2's DSpark draft head. Parameter names are the checkpoint's (`mtp.safetensors`)."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
        self.width = config.block_size
        self.layers = [DraftLayer(config, i in config.local_layer_ids) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config)
        self.fuse_weight = mx.zeros((config.num_hidden_layers + 1, len(config.target_layer_ids)))
        self.mask_embed = mx.zeros((config.hidden_size,))
        # Derived by attach(): kept out of the parameters by their leading underscore.
        self._alpha = self._fold_weight = self._fold_bias = self._rnn_bias = None
        self._rnn_step = self._rnn_rows = None
        self._fold_bits = self._fold_q = None  # set by quantize(): the folded table, quantized

    def make_cache(self) -> DraftCache:
        return DraftCache(self.config)

    def attach(self, model) -> "Drafter":
        """Derive what depends on the loaded weights and the target's output layer. Call again after either changes."""
        r = self.config.markov_rank
        head = self.markov_head
        self._alpha = mx.softmax(self.fuse_weight.astype(_F32), axis=-1).astype(self.fuse_weight.dtype)
        rnn = head.rnn_proj.weight.astype(_F32)  # bf16 products are exact in float32: vLLM's bf16 GEMM, fp32 out
        self._rnn_step, self._rnn_rows = rnn[:, :2 * r], rnn[:, 2 * r:]
        self._rnn_bias = head.rnn_proj.bias.astype(_F32)
        weight, bias = head.hidden_out.weight.astype(_F32), head.hidden_out.bias.astype(_F32)
        fold_w, fold_b = [], []
        for start in range(0, self.config.vocab_size, 4096):
            rows = _output_rows(model.lm_head, start, min(start + 4096, self.config.vocab_size)).astype(_F32)
            fold_w.append((rows @ weight).astype(head.hidden_out.weight.dtype))
            fold_b.append((rows @ bias).astype(head.hidden_out.weight.dtype))
            mx.eval(fold_w[-1], fold_b[-1])
        self._fold_weight, self._fold_bias = mx.concatenate(fold_w), mx.concatenate(fold_b)
        mx.eval(self._alpha, self._rnn_step, self._rnn_rows, self._rnn_bias, self._fold_weight, self._fold_bias)
        if self._fold_bits is not None:
            self._quantize_fold()
        return self

    def quantize(self, bits: int = 8, group_size: int = 64) -> "Drafter":
        """Quantize the draft layers' projections and the folded table the recurrent head reads for every guess
        (Inkling 2 Mini: 0.5 GB and 102 MB in bf16). Guesses get cheaper and may change; the target checks every
        one, so the output doesn't. The recurrent head's own weights stay as trained: they feed float32 math."""
        nn.quantize(self, group_size=group_size, bits=bits,
                    class_predicate=lambda path, m: path.startswith("layers.") and isinstance(m, nn.Linear))
        self._fold_bits = (bits, group_size)
        if self._fold_weight is not None:
            self._quantize_fold()
        return self

    def _quantize_fold(self) -> None:
        bits, group_size = self._fold_bits
        self._fold_q = mx.quantize(self._fold_weight, group_size=group_size, bits=bits)
        mx.eval(self._fold_q)
        self._fold_weight = None

    def _mix(self, taps: Sequence[mx.array], row: int, norm: nn.RMSNorm) -> mx.array:
        """RMSNorm of the alpha-weighted sum of the taps, in float32 math, returned in the taps' dtype."""
        alpha = self._alpha[row].astype(_F32)
        mixed = taps[0].astype(_F32) * alpha[0]
        for i in range(1, len(taps)):
            mixed = mixed + taps[i].astype(_F32) * alpha[i]
        return _rms(mixed, norm.weight, norm.eps).astype(taps[0].dtype)

    def observe(self, taps: Sequence[mx.array], cache: DraftCache) -> None:
        """Read the target's taps for newly committed positions (each (1, n, hidden), in target_layer_ids order)."""
        for i, (layer, context) in enumerate(zip(self.layers, cache.contexts)):
            layer.observe(self._mix(taps, i, layer.dflare_norm), context)
        cache.f_rnn = self._mix([t[:, -1:] for t in taps], -1, self.markov_head.dflare_norm)

    def propose(self, model, anchor: int, count: int, cache: DraftCache) -> mx.array:
        """Up to `count` (< block_size) greedy guesses to follow `anchor`, unevaluated (shape (count,))."""
        if not 1 <= count < self.width:
            raise ValueError(f"a DSpark block guesses 1 to {self.width - 1} tokens, not {count}")
        if self._fold_bias is None:
            raise RuntimeError("Drafter.attach(model) must run before propose")
        text = model.model.language_model
        embed = text.embed(mx.array([[anchor]]))
        dtype = embed.dtype
        masks = mx.broadcast_to(self.mask_embed.astype(dtype), (1, self.width - 1, embed.shape[-1]))
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for layer, context in zip(self.layers, cache.contexts):
            x, residual = layer(x, residual, context)
        rows = _add_norm(x, residual, self.norm)[1][:, 1:count + 1]
        base = model.lm_head(rows)  # bf16 base logits, as vLLM's draft (the target's own head is float32)

        head, cap = self.markov_head, self.config.final_logit_softcapping
        inputs = mx.concatenate([rows, mx.broadcast_to(cache.f_rnn, rows.shape)], axis=-1)
        context = inputs.astype(_F32) @ self._rnn_rows.T + self._rnn_bias  # every step's [row, f_rnn] term
        state = mx.zeros((1, self.config.markov_rank), dtype)
        prev = mx.array([anchor])
        guesses = []
        for i in range(count):
            step = mx.concatenate([state, head.markov_w1(prev).astype(dtype)], axis=-1)
            raw = (step.astype(_F32) @ self._rnn_step.T + context[:, i]).astype(dtype)
            gate, candidate, out = mx.split(raw, 3, axis=-1)
            # vLLM's markov_gate: each op of the reference rounds to bf16.
            gate = mx.sigmoid(gate.astype(_F32)).astype(dtype)
            candidate = mx.tanh(candidate.astype(_F32)).astype(dtype)
            keep = (gate.astype(_F32) * state.astype(_F32)).astype(dtype)
            update = ((1.0 - gate.astype(_F32)).astype(dtype).astype(_F32) * candidate.astype(_F32)).astype(dtype)
            state = (keep.astype(_F32) + update.astype(_F32)).astype(dtype)
            latent = mx.tanh(out.astype(_F32)).astype(dtype)
            if self._fold_q is not None:
                bits, group_size = self._fold_bits
                bias = mx.quantized_matmul(latent, *self._fold_q, transpose=True, group_size=group_size,
                                           bits=bits) + self._fold_bias
            else:
                bias = mx.addmm(self._fold_bias, latent, self._fold_weight.T)
            # vLLM's add_softcap, then its argmax: bf16(cap * tanh(bf16(base + bias) / cap)), NaN as -inf.
            scores = (base[:, i, :self.config.vocab_size].astype(_F32) + bias.astype(_F32)).astype(dtype)
            scores = (mx.tanh(scores.astype(_F32) / cap) * cap).astype(dtype)
            scores = mx.where(mx.isnan(scores), -mx.inf, scores)
            prev = mx.argmax(scores, axis=-1)
            guesses.append(prev)
        return mx.concatenate(guesses)


def load_drafter(path: str | Path, model, *, weights: str | Path | None = None, bits: int | None = None) -> Drafter:
    """The draft head of the Inkling 2 checkpoint at `path`, configured by its config.json and loaded from `weights`
    (default: its mtp.safetensors), attached to `model`, the target loaded from the same checkpoint. `bits` quantizes
    it (Drafter.quantize)."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    drafter = Drafter(DraftConfig.from_target(config, model.args.text))
    drafter.load_weights(list(mx.load(str(Path(weights) if weights else path / "mtp.safetensors")).items()),
                         strict=True)
    mx.eval(drafter.parameters())
    drafter.attach(model)
    return drafter.quantize(bits) if bits else drafter


# --- The target's side --------------------------------------------------------------------------------------------

def _visits(cache) -> list:
    return [(entry[0], entry[1]) for entry in cache]


def read(model, tokens: mx.array, cache, taps: Sequence[int], *, record: bool = False):
    """The target reads `tokens` (1, n). Returns its final hidden states and the taps (each (1, n, hidden)).
    `record` keeps what `rewind` needs to undo this read."""
    if record:
        for _, conv in _visits(cache):
            conv.windows = [None] * len(conv.cache)
    captures = dict.fromkeys(taps)
    hidden = model.model.language_model(tokens, cache=cache, captures=captures)
    return hidden, [captures[t] for t in taps]


def _drop_tail(kv: RotatingKVCache, n: int) -> None:
    if kv._idx != kv.keys.shape[2]:
        raise RuntimeError("a rotating cache rewinds only after a multi-token read, which leaves it in temporal order")
    end = kv.keys.shape[2] - n
    kv.keys, kv.values = kv.keys[..., :end, :], kv.values[..., :end, :]
    kv.offset -= n
    kv._idx = end


def rewind(cache, total: int, kept: int) -> None:
    """Keep the first `kept` of the `total` tokens the last recorded `read` took, exactly as if the rest were never
    read."""
    drop = total - kept
    for kv, conv in _visits(cache):
        windows, conv.windows = getattr(conv, "windows", None), None
        if drop == 0:
            continue
        if windows is None:
            raise RuntimeError("rewind needs the read it undoes to be recorded")
        if isinstance(kv, RotatingKVCache):
            _drop_tail(kv, drop)
        else:
            kv.trim(drop)
        for i, window in enumerate(windows):
            n = conv[i].shape[1]  # a convolution's state is its last K-1 inputs; the window is state + all inputs
            conv[i] = mx.contiguous(window[:, kept:kept + n])


def cache_arrays(cache) -> list[mx.array]:
    out = []
    for kv, conv in _visits(cache):
        out += [a for a in (kv.keys, kv.values) if a is not None]
        out += [a for a in conv.cache if a is not None]
    return out


class DSparkSpeculator:
    """The `Speculator` for an Inkling 2 target and its draft head (None: drafts come from the loop's test drafter)."""

    def __init__(self, model, drafter: Drafter | None):
        self.model, self.drafter = model, drafter
        self.cache = make_prompt_cache(model)
        self.draft_cache = drafter.make_cache() if drafter is not None else None
        self.taps = drafter.config.target_layer_ids if drafter is not None else ()
        self._pending = None

    def prefill(self, prompt: Sequence[int], step: int) -> mx.array:
        hidden = None
        for start in range(0, len(prompt), step):
            hidden, taps = read(self.model, mx.array([list(prompt[start:start + step])]), self.cache, self.taps)
            if self.drafter is not None:
                self.drafter.observe(taps, self.draft_cache)
            hidden = hidden[:, -1:]
            mx.eval(hidden, cache_arrays(self.cache), self.draft_cache.arrays() if self.drafter is not None else [])
        return self.model.logits(hidden)[:, -1]

    def draft(self, token: int, depth: int) -> mx.array | list[int]:
        if self.drafter is None or depth < 1:
            return []
        return self.drafter.propose(self.model, token, min(depth, self.drafter.width - 1), self.draft_cache)

    def verify(self, tokens: mx.array) -> mx.array:
        hidden, self._pending = read(self.model, tokens[None], self.cache, self.taps, record=True)
        return self.model.logits(hidden)[0]

    def commit(self, total: int, kept: int) -> None:
        rewind(self.cache, total, kept)
        taps, self._pending = self._pending, None
        if self.drafter is not None:
            self.drafter.observe([t[:, :kept] for t in taps], self.draft_cache)


def generate_dspark(model, drafter: Drafter | None, prompt: Sequence[int], *, max_tokens: int, depth: int = 7,
                    sampling: Sampling | None = None, processors: Sequence[Processor] = (), seed: int = 0,
                    stop_tokens: Sequence[int] = (), prefill_step: int = 2048, test_drafter: DraftHook | None = None,
                    stats: SpeculativeStats | None = None) -> list[int]:
    """Generate up to `max_tokens` tokens after `prompt` with Inkling 2 and its draft head. `test_drafter` replaces
    the head's guesses in tests."""
    return speculate(DSparkSpeculator(model, drafter), prompt, max_tokens=max_tokens, depth=depth,
                     sampling=sampling, processors=processors, seed=seed, stop_tokens=stop_tokens,
                     prefill_step=prefill_step, drafter=test_drafter, stats=stats)
