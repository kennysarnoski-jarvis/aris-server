"""The quantized KV cache must hold what mlx-lm's full-precision batched cache holds through every batch operation the
generator uses, and attention on it must match attention on the full-precision cache."""

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator, _merge_caches
from mlx_lm.models import base as mlx_base
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache, QuantizedKVCache

from aris_server.disk_cache import DiskCache, model_id
from aris_server.kv_quant import (
    QUANTIZED_ATTENTION_TOKENS,
    BatchQuantizedKVCache,
    kv_bytes_per_token,
    quantize_kv,
    scaled_dot_product_attention,
)

H, D = 2, 64
TOLERANCE = 0.03  # 8-bit, groups of 64: a step is under 1/100 of the range of unit-normal values


def kv(batch, n, seed):
    mx.random.seed(seed)
    return mx.random.normal((batch, H, n, D)), mx.random.normal((batch, H, n, D))


def one(n, seed, quantized):
    cache = QuantizedKVCache(group_size=64, bits=8) if quantized else KVCache()
    if n:
        cache.update_and_fetch(*kv(1, n, seed))
    return cache


def both(*lengths):
    """The same sequences as mlx-lm's full-precision batch and as the 8-bit one."""
    return (BatchKVCache.merge([one(n, i, False) for i, n in enumerate(lengths)]),
            BatchQuantizedKVCache.merge([one(n, i, True) for i, n in enumerate(lengths)]))


def close(want, got, tolerance=TOLERANCE):
    assert want.shape == got.shape
    assert mx.abs(want - got).max().item() < tolerance


def same(ref, q):
    assert q._idx == ref._idx
    assert q.offset.tolist() == ref.offset.tolist()
    assert q.left_padding.tolist() == ref.left_padding.tolist()
    assert mx.array_equal(q.make_mask(1), ref.make_mask(1))
    if ref.keys is None:
        assert q.keys is None
        return
    (rk, rv), (qk, qv) = ref.keys_and_values(), q.keys_and_values()
    close(rk, q.dequantize(qk))
    close(rv, q.dequantize(qv))


def test_a_batched_8bit_cache_holds_what_the_full_precision_one_does_through_every_batch_operation():
    ref, q = both(5, 0, 9)
    same(ref, q)

    # A chunk of prompt, right-padded: it comes back dequantized, for the fused attention kernel.
    for c in (ref, q):
        c.prepare(lengths=[40, 30, 20], right_padding=[0, 10, 20])
    k, v = kv(3, 40, 10)
    (rk, rv), (qk, qv) = ref.update_and_fetch(k, v), q.update_and_fetch(k, v)
    assert not isinstance(qk, tuple)
    close(rk, qk)
    close(rv, qv)
    for c in (ref, q):
        c.finalize()
    same(ref, q)

    # Decode steps: they come back quantized.
    for step in range(3):
        k, v = kv(3, 1, 20 + step)
        (rk, _), (qk, _) = ref.update_and_fetch(k, v), q.update_and_fetch(k, v)
        assert isinstance(qk, tuple)
        close(rk, q.dequantize(qk))
    same(ref, q)

    # A sequence finishes, another joins with its own history, and the cache grows past a step.
    for c in (ref, q):
        c.filter([0, 2])
    same(ref, q)
    ref.extend(BatchKVCache.merge([one(300, 30, False)]))
    q.extend(BatchQuantizedKVCache.merge([one(300, 30, True)]))
    same(ref, q)
    k, v = kv(3, 1, 40)
    ref.update_and_fetch(k, v), q.update_and_fetch(k, v)
    same(ref, q)

    # A finished sequence comes out as a single-sequence cache, the shape it is kept in, and merges back.
    for i in range(3):
        r, out = ref.extract(i), q.extract(i)
        assert isinstance(out, QuantizedKVCache) and out.offset == r.offset == out.size()
        close(r.keys, mx.dequantize(*out.keys, group_size=64, bits=8))
    again = BatchQuantizedKVCache.merge([q.extract(i) for i in range(3)])
    assert again._idx == q._idx - min(q.left_padding.tolist())
    assert mx.dequantize(*again.keys_and_values()[0], group_size=64, bits=8).shape[2] == again._idx


def test_an_8bit_cache_trims_like_the_full_precision_one():
    ref, q = both(12, 7)
    for c in (ref, q):
        assert c.trim(4) == 4
    same(ref, q)


@pytest.mark.parametrize("n", [1, 3, QUANTIZED_ATTENTION_TOKENS + 8])
def test_attention_on_the_8bit_cache_matches_attention_on_the_full_precision_one(n):
    ref, q = both(30, 12)  # sequences of different lengths: left padding to mask
    mx.random.seed(50)
    queries = mx.random.normal((2, H * 6, n, D))  # six query heads to a KV head, as Arete has
    mask = ref.make_mask(n)
    assert mx.array_equal(mask, q.make_mask(n))
    k, v = kv(2, n, 51)
    (rk, rv), (qk, qv) = ref.update_and_fetch(k, v), q.update_and_fetch(k, v)
    assert isinstance(qk, tuple) == (n <= QUANTIZED_ATTENTION_TOKENS)
    want = mx.fast.scaled_dot_product_attention(queries, rk, rv, scale=D ** -0.5, mask=mask)
    got = scaled_dot_product_attention(queries, qk, qv, cache=q, scale=D ** -0.5, mask=mask)
    close(want, got)
    if isinstance(qk, tuple):  # the same answer as mlx-lm's quantized attention, from one matmul per KV head
        theirs = mlx_base.quantized_scaled_dot_product_attention(queries, qk, qv, scale=D ** -0.5, mask=mask,
                                                                 group_size=64, bits=8)
        close(theirs, got, 1e-4)


def tiny_hybrid_model():
    """Arete's architecture (Qwen3.5: three linear-attention layers to each full-attention one), tiny and random."""
    args = qwen3_5.TextModelArgs(model_type="qwen3_5", hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                                 num_attention_heads=12, num_key_value_heads=2, head_dim=64, vocab_size=64,
                                 linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
                                 linear_value_head_dim=32, full_attention_interval=2)
    mx.random.seed(0)
    model = qwen3_5.TextModel(args)
    mx.eval(model.parameters())
    return model


def test_a_hybrid_model_reads_and_generates_the_same_with_an_8bit_cache():
    model = tiny_hybrid_model()
    full_precision = model.make_cache
    quantize_kv(model, 8)
    assert [type(c) for c in model.make_cache()] == [ArraysCache, QuantizedKVCache, ArraysCache, QuantizedKVCache]

    def run(make_cache):
        prompts = [[3, 1, 4, 1, 5, 9, 2, 6] * 3, [2, 7, 1, 8, 2, 8]]
        caches = []
        for p in prompts:  # each conversation read on its own, the way a finished turn is kept
            cache = make_cache()
            model(mx.array([p]), cache=cache)
            caches.append(cache)
        batch = _merge_caches(caches)  # then batched together: the shorter one is left-padded
        logits = [model(mx.array([[5], [5]]), cache=batch)]
        logits.append(model(mx.array([[1, 2, 3, 4] * 5] * 2), cache=batch))  # a chunk of prompt: the fused path
        for _ in range(4):  # greedy decode steps
            logits.append(model(mx.argmax(logits[-1][:, -1:], axis=-1), cache=batch))
        return logits

    want, got = run(full_precision), run(model.make_cache)
    for w, g in zip(want, got):
        close(w, g, 0.05)
        assert mx.array_equal(mx.argmax(w, axis=-1), mx.argmax(g, axis=-1))


def generate(model, prompts, caches=None, max_tokens=6):
    """Greedy generation through mlx-lm's BatchGenerator, the server's path. Returns each sequence's tokens, logprobs
    and the cache it finished with."""
    gen = BatchGenerator(model, max_tokens=max_tokens, prefill_step_size=QUANTIZED_ATTENTION_TOKENS * 2)
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts), caches=caches)
    out = {uid: ([], [], None) for uid in uids}
    while any(cache is None for _, _, cache in out.values()):
        for r in gen.next_generated():
            tokens, logprobs, _ = out[r.uid]
            tokens.append(r.token)
            logprobs.append(r.logprobs)
            if r.finish_reason is not None:
                out[r.uid] = (tokens, logprobs, r.prompt_cache)
    gen.close()
    return [out[uid] for uid in uids]


def test_mlx_lm_batch_generator_runs_two_turns_of_two_conversations_the_same_with_an_8bit_cache():
    model = tiny_hybrid_model()
    first = [[3, 1, 4, 1, 5, 9, 2, 6] * 9, [2, 7, 1, 8, 2, 8]]  # 72 tokens read in chunks, and 6

    def two_turns():
        turn1 = generate(model, first)
        follow = [[11, 12, 13] * 7, [5]]  # each conversation continues from the cache its first turn left
        turn2 = generate(model, follow, caches=[cache for _, _, cache in turn1])
        return turn1 + turn2

    want = two_turns()
    quantize_kv(model, 8)
    got = two_turns()
    assert all(type(c) is QuantizedKVCache for _, _, cache in got for c in cache if not isinstance(c, ArraysCache))
    for (w_tokens, w_logprobs, _), (g_tokens, g_logprobs, _) in zip(want, got):
        assert w_tokens == g_tokens
        for w, g in zip(w_logprobs, g_logprobs):
            close(w, g, 0.05)


def test_an_empty_8bit_cache_has_no_bytes():
    # mlx-lm measures every waiting sequence's cache, which is empty until its prompt is read.
    assert QuantizedKVCache(group_size=64, bits=8).nbytes == 0


def test_the_cache_a_token_adds_is_counted_at_its_quantized_size():
    # Arete-27B: 16 full-attention layers, 4 KV heads of 256
    assert kv_bytes_per_token(16, 4, 256) == 65536
    assert kv_bytes_per_token(16, 4, 256, bits=8) == 34816
    assert kv_bytes_per_token(16, 4, 256, bits=4) == 18432
    for bits in (4, 8):
        cache = BatchQuantizedKVCache([0], bits=bits)
        cache.step = 512
        cache.update_and_fetch(mx.zeros((1, 4, 512, 256), mx.bfloat16), mx.zeros((1, 4, 512, 256), mx.bfloat16))
        assert cache.nbytes == 512 * kv_bytes_per_token(1, 4, 256, bits=bits)


def test_only_4_or_8_bits():
    with pytest.raises(ValueError):
        quantize_kv(tiny_hybrid_model(), 3)


def test_saved_caches_are_kept_apart_by_the_precision_of_their_kv_cache(tmp_path):
    assert model_id("model-a", None, kv_bits=None) == model_id("model-a", None)  # full precision keeps its caches
    assert model_id("model-a", None, kv_bits=8) != model_id("model-a", None)
    assert model_id("model-a", None, kv_bits=8) != model_id("model-a", None, kv_bits=4)
    assert DiskCache(tmp_path, "model-a", 1 << 30, kv_bits=8).dir != DiskCache(tmp_path, "model-a", 1 << 30).dir


def test_an_8bit_cache_is_saved_to_disk_and_comes_back(tmp_path):
    disk = DiskCache(tmp_path, "model-a", 1 << 30, kv_bits=8)
    tokens = list(range(2000))
    lin = ArraysCache(2)
    lin[0], lin[1] = mx.ones((1, 3)), mx.ones((1, 2, 2))
    disk.save(tokens, [one(2000, 1, True), lin], "assistant")
    restored = []
    disk.restore(lambda t, c, kind: restored.append((t, c)), 1 << 30)
    (t, (kv_cache, _)), = restored
    assert t == tokens and isinstance(kv_cache, QuantizedKVCache) and kv_cache.offset == 2000
    close(one(2000, 1, False).keys, mx.dequantize(*kv_cache.keys, group_size=64, bits=8))
