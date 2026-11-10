"""A quantized KV cache for mlx-lm's batched generation.

mlx-lm quantizes the KV cache only outside batching (its server stops batching when kv_bits is set: there is no
batched quantized cache), and its quantized attention is built for neither job here. It holds the whole score matrix,
so reading a 512-token chunk of a prompt against 60k tokens of context needs about 1.5 GB of scratch per layer, and it
reads the cache once per query head that shares a KV head (six times for Arete) when decoding.

Here the cache batches like mlx-lm's BatchKVCache, over (packed, scales, biases) arrays with the sequence on axis 2.
A forward pass over a chunk of prompt reads a dequantized copy of one layer's cache through the fused kernel, which
never holds the score matrix (about 250 MB at 60k tokens for Arete); a decode step attends on the quantized cache
directly, all query heads of a KV head in one matmul: half the bytes of a full-precision cache to read.
"""

from __future__ import annotations

import sys

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models import base as mlx_base
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, QuantizedKVCache, _BaseCache, dynamic_roll

from .config import KV_BITS_CHOICES

KV_GROUP_SIZE = 64
# A forward pass over at most this many new tokens per sequence attends on the quantized cache (its score matrix is
# small); longer ones, chunks of a prompt, read a dequantized copy through the fused kernel.
QUANTIZED_ATTENTION_TOKENS = 16


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: int | None = None,
                       group_size: int = KV_GROUP_SIZE, dtype_bytes: int = 2) -> int:
    """The cache one token of context adds: keys and values for every full-attention layer."""
    if not bits:
        return layers * 2 * kv_heads * head_dim * dtype_bytes
    per_head = head_dim * bits // 8 + 2 * (head_dim // group_size) * dtype_bytes  # packed values, scales and biases
    return layers * 2 * kv_heads * per_head


def _empty(batch: int, heads: int, length: int, dim: int, dtype, group_size: int, bits: int):
    return (mx.zeros((batch, heads, length, dim * bits // 32), dtype=mx.uint32),
            mx.zeros((batch, heads, length, dim // group_size), dtype=dtype),
            mx.zeros((batch, heads, length, dim // group_size), dtype=dtype))


class BatchQuantizedKVCache(_BaseCache):
    """mlx-lm's BatchKVCache (left-padded sequences, per-sequence offsets) over quantized keys and values."""

    step = 256

    def __init__(self, left_padding: list[int], group_size: int = KV_GROUP_SIZE, bits: int = 8):
        self.keys = None  # (packed, scales, biases)
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-p for p in left_padding])
        self._idx = 0
        self._right_padding = None
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        prev, n = self._idx, keys.shape[2]
        if self.keys is None or prev + n > self.keys[0].shape[2]:
            batch, heads = keys.shape[:2]
            size = (self.step + n - 1) // self.step * self.step
            new_k = _empty(batch, heads, size, keys.shape[3], keys.dtype, self.group_size, self.bits)
            new_v = _empty(batch, heads, size, values.shape[3], values.dtype, self.group_size, self.bits)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(lambda x: x[..., :prev, :], (self.keys, self.values))
                self.keys = tuple(mx.concatenate([old, new], axis=2) for old, new in zip(self.keys, new_k))
                self.values = tuple(mx.concatenate([old, new], axis=2) for old, new in zip(self.values, new_v))
        self.offset = self.offset + n
        self._idx += n
        for buffer, x in ((self.keys, keys), (self.values, values)):
            for part, q in zip(buffer, mx.quantize(x, group_size=self.group_size, bits=self.bits)):
                part[..., prev:self._idx, :] = q
        k, v = self.keys_and_values()
        if n > QUANTIZED_ATTENTION_TOKENS:
            return self.dequantize(k), self.dequantize(v)
        return k, v

    def keys_and_values(self):
        return tree_map(lambda x: x[..., :self._idx, :], (self.keys, self.values))

    def dequantize(self, q):
        return mx.dequantize(*q, group_size=self.group_size, bits=self.bits)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError("Left padding can only be added to an empty BatchQuantizedKVCache")
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys, self.values = tree_map(lambda x: dynamic_roll(x, padding[:, None], axis=2),
                                              (self.keys, self.values))
            self.offset = self.offset - padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.left_padding, self._idx

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding, self._idx = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(N, offset=self._idx, left_padding=self.left_padding, **kwargs)

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys, self.values = tree_map(lambda x: x[batch_indices], (self.keys, self.values))
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        min_left_pad = min(self.left_padding.tolist())  # shift left to drop padding every sequence has
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys, self.values = tree_map(lambda x: x[..., min_left_pad:, :], (self.keys, self.values))
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        max_idx = max(self._idx, other._idx)
        max_size = max(c.keys[0].shape[2] for c in (self, other) if c.keys is not None)
        ref = self if self.keys is not None else other

        def pad(c):  # right-justify each cache at the shared index, at the same size
            if c.keys is None:
                rows = c.offset.shape[0]
                k, v = tree_map(lambda x: mx.zeros((rows, x.shape[1], 0, x.shape[3]), x.dtype), (ref.keys, ref.values))
            else:
                k, v = c.keys, c.values
            left = max_idx - c._idx
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k, v = tree_map(lambda x: x[..., :right, :], (k, v))
                right = 0
            if left or right:
                k, v = tree_map(lambda x: mx.pad(x, [(0, 0), (0, 0), (left, right), (0, 0)]), (k, v))
            return k, v, c.offset, c.left_padding + left

        (k1, v1, o1, p1), (k2, v2, o2, p2) = pad(self), pad(other)
        self.keys = tuple(mx.concatenate([a, b]) for a, b in zip(k1, k2))
        self.values = tuple(mx.concatenate([a, b]) for a, b in zip(v1, v2))
        self.offset = mx.concatenate([o1, o2])
        self.left_padding = mx.concatenate([p1, p2])
        self._idx = max_idx

    def extract(self, idx):
        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
        mx.eval(self.left_padding)
        padding = self.left_padding.tolist()[idx]
        cache.keys, cache.values = tree_map(lambda x: mx.contiguous(x[idx:idx + 1, :, padding:self._idx]),
                                            (self.keys, self.values))
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        ref = next((c for c in caches if isinstance(c, QuantizedKVCache)), None)
        group_size, bits = (ref.group_size, ref.bits) if ref else (KV_GROUP_SIZE, 8)
        caches = [c.to_quantized(group_size=group_size, bits=bits) if isinstance(c, KVCache) and c.keys is not None
                  else c for c in caches]
        lengths = [c.offset if c.keys is not None else 0 for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls([0] * len(caches), group_size, bits)
        full = next(c for c, n in zip(caches, lengths) if n)
        rows = len(caches)
        keys, values = tree_map(lambda x: mx.zeros((rows, x.shape[1], max_length, x.shape[3]), x.dtype),
                                (full.keys, full.values))
        padding = [max_length - n for n in lengths]
        for i, (p, n, c) in enumerate(zip(padding, lengths, caches)):
            if not n:
                continue
            for dst, src in zip((*keys, *values), (*c.keys, *c.values)):
                dst[i:i + 1, :, p:p + n] = src[..., :n, :]
        cache = cls(padding, group_size, bits)
        cache.keys, cache.values = keys, values
        cache.offset += max_length
        cache._idx = max_length
        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(x.nbytes for x in (*self.keys, *self.values))


def quantized_attention(queries, keys, values, scale: float, mask, group_size: int, bits: int):
    """Attention on quantized keys and values: every query head of a KV head in one matmul over that head's cache."""
    batch, q_heads, n, dim = queries.shape
    kv_heads = keys[0].shape[1]
    repeats = q_heads // kv_heads
    q = (queries * scale).reshape(batch, kv_heads, repeats * n, dim)
    scores = mx.quantized_matmul(q, *keys, transpose=True, group_size=group_size, bits=bits)
    if mask is not None:
        scores = scores.reshape(batch, kv_heads, repeats, n, -1)
        if isinstance(mask, str):  # "causal"
            length = scores.shape[-1]
            mask = mx.arange(length - n, length)[:, None] >= mx.arange(length)[None]
        if mask.ndim == 4:  # (batch, 1, n, length): one mask for every head
            mask = mx.expand_dims(mask, 2)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
        scores = scores.reshape(batch, kv_heads, repeats * n, -1)
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *values, transpose=False, group_size=group_size, bits=bits)
    return out.reshape(batch, q_heads, n, -1)


_ORIGINAL_ATTENTION = mlx_base.scaled_dot_product_attention


def scaled_dot_product_attention(queries, keys, values, cache, scale: float, mask, sinks=None):
    """mlx-lm's attention, choosing the quantized path by what the cache returned rather than by the cache's type: a
    BatchQuantizedKVCache returns quantized keys to a decode step and dequantized ones to a chunk of prompt."""
    if isinstance(keys, tuple):
        if sinks is not None:
            raise ValueError("quantized attention does not support attention sinks")
        return quantized_attention(queries, keys, values, scale, mask, cache.group_size, cache.bits)
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask, sinks=sinks)


def install_attention() -> None:
    """Route mlx-lm's attention through scaled_dot_product_attention above. Model modules import mlx-lm's function by
    name, so each loaded one is patched too, and base is patched for the ones imported later."""
    mlx_base.scaled_dot_product_attention = scaled_dot_product_attention
    for name, module in list(sys.modules.items()):
        original = getattr(module, "scaled_dot_product_attention", None) is _ORIGINAL_ATTENTION
        if name.startswith("mlx_lm.models.") and original:
            module.scaled_dot_product_attention = scaled_dot_product_attention


def quantize_kv(model, bits: int, group_size: int = KV_GROUP_SIZE) -> None:
    """Give a loaded model quantized caches for its full-attention layers; recurrent layers keep theirs."""
    if bits not in KV_BITS_CHOICES:
        raise ValueError(f"kv_bits must be one of {KV_BITS_CHOICES}, not {bits!r}")
    install_attention()
    make_cache = model.make_cache

    def make_quantized_cache():
        return [QuantizedKVCache(group_size=group_size, bits=bits) if type(c) is KVCache else c for c in make_cache()]

    model.make_cache = make_quantized_cache


# mlx-lm's single-sequence quantized cache is what a finished turn is kept as, and what a waiting sequence starts
# with: batching it needs a merge, the prompt cache measures entries by size(), and mlx-lm's server measures waiting
# sequences' caches by nbytes, which mlx-lm's own fails on while the cache is empty.
QuantizedKVCache.merge = classmethod(lambda cls, caches: BatchQuantizedKVCache.merge(caches))
QuantizedKVCache.size = lambda self: self.offset
QuantizedKVCache.nbytes = property(
    lambda self: 0 if self.keys is None else sum(x.nbytes for x in (*self.keys, *self.values)))
