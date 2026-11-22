"""The prompt cache with Inkling's layer caches: each layer visit keeps a CacheList of an attention cache (a
sliding window's ring, or a full one) and its convolutions' state. A finished turn whose history comes back tokenized
differently is rebuilt to its prompt's end, as for a recurrent model, instead of failing the request."""

import gc

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache, RotatingKVCache, make_prompt_cache

from aris_server import engine
from aris_server.engine import MovingPromptCache, _end_state
from test_inkling2_model import LOOP_CONFIG, UNPADDED_VOCAB, create_model


def inkling_layers(n_tokens, seed, window=8):
    """Two visits as Inkling keeps them: a sliding-window one and a full-attention one, each with 4 conv states."""
    out = []
    for kv in (RotatingKVCache(max_size=window), KVCache()):
        kv.update_and_fetch(mx.full((1, 2, n_tokens, 4), seed, mx.float16),
                            mx.full((1, 2, n_tokens, 4), -seed, mx.float16))
        conv = ArraysCache(4)
        for i in range(4):
            conv[i] = mx.full((1, 3, 4), seed, mx.float32)
        out.append(CacheList(kv, conv))
    return out


PROMPT = list(range(1200)) + list(range(5000, 6000))


def test_a_diverged_turn_is_rebuilt_to_its_prompt_end():
    cache = MovingPromptCache(max_size=10)
    cache.reading.append(PROMPT)
    at_end = inkling_layers(len(PROMPT) - 1, 1)
    cache.insert_cache("m", PROMPT[:-1], at_end, cache_type="assistant")       # prompt end: state 1
    cache.reading.remove(PROMPT)
    generated = PROMPT + [5, 6, 7]
    cache.insert_cache("m", generated, inkling_layers(len(generated), 2), cache_type="assistant")
    rendered = PROMPT + [5, 67, 8]                                               # same text, re-tokenized
    got, rest = cache.fetch_nearest_cache("m", rendered)
    n = len(PROMPT) - 1
    assert rest == rendered[n:] and len(cache) == 0
    window, full = got
    assert window[0].offset == n and window[0].keys.shape[2] == 8                # the window's last rows, put back
    assert mx.array_equal(window[0].keys, at_end[0][0].keys[..., -8:, :])
    assert full[0].offset == n and mx.all(full[0].keys[..., :n, :] == 2).item()  # full attention: cut back
    assert all(mx.all(a == 1).item() for layer in got for a in layer[1].cache)   # conv state at the prompt end


def test_the_rebuilt_cache_reads_on_exactly_as_one_that_read_the_history():
    model = create_model(config=LOOP_CONFIG)
    prompt = mx.random.randint(0, UNPADDED_VOCAB, (40,), key=mx.random.key(1)).tolist()
    cache = MovingPromptCache(max_size=10)
    cache.light_min_tokens = 16
    live = make_prompt_cache(model)
    model(mx.array([prompt[:-1]]), cache=live)
    cache.reading.append(prompt)
    cache.insert_cache("m", prompt[:-1], live, cache_type="assistant")  # mlx-lm's snapshot at the prompt's end
    # The turn reads its last prompt token and decodes 12 more, one at a time: more than the 8-token window, so
    # the ring overwrites every row it held at the prompt's end.
    read, token = list(prompt[:-1]), prompt[-1]
    for _ in range(13):
        logits = model(mx.array([[token]]), cache=live)
        read.append(token)
        token = int(mx.argmax(logits[0, -1]).item())
    cache.reading.remove(prompt)
    cache.insert_cache("m", read, live, cache_type="assistant")
    rendered = prompt + [(t + 1) % UNPADDED_VOCAB for t in read[len(prompt):]] + [3, 4]
    got, rest = cache.fetch_nearest_cache("m", rendered)
    assert len(rendered) - len(rest) == len(prompt) - 1
    fresh = make_prompt_cache(model)
    expected = model(mx.array([rendered]), cache=fresh)[:, -len(rest):]
    assert mx.allclose(model(mx.array([rest]), cache=got), expected, atol=1e-4)


def test_an_end_state_is_a_copy_the_ring_cannot_overwrite():
    kv = RotatingKVCache(max_size=8)
    kv.update_and_fetch(mx.arange(1 * 2 * 20 * 4, dtype=mx.float32).reshape(1, 2, 20, 4), mx.ones((1, 2, 20, 4)))
    keys, values, offset = _end_state(kv)
    before = mx.array(keys)
    for _ in range(10):  # decode writes into the ring in place
        kv.update_and_fetch(mx.full((1, 2, 1, 4), -1.0), mx.full((1, 2, 1, 4), -1.0))
    assert offset == 20 and mx.array_equal(keys, before)
    assert _end_state(RotatingKVCache(max_size=8, keep=4)) == engine._NOT_KEPT


def _nested_batch(rows):
    """A prompt batch of `rows` sequences whose one layer visit keeps a CacheList: attention and 4 MB a row of
    convolution state."""
    from mlx_lm.generate import PromptProcessingBatch

    kv = BatchKVCache([0] * rows)
    kv.update_and_fetch(mx.random.normal((rows, 2, 300, 64)), mx.random.normal((rows, 2, 300, 64)))
    conv = ArraysCache(2)
    conv.cache = [mx.random.normal((rows, 1024, 1024)), mx.random.normal((rows, 4, 64))]
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.uids, batch.prompt_cache = list(range(rows)), [CacheList(kv, conv)]
    mx.eval(kv.state, conv.cache)
    return batch


def test_a_conversation_kept_from_a_shared_batch_copies_its_nested_state_out():
    base = mx.get_active_memory()
    batch = _nested_batch(4)
    whole = mx.get_active_memory() - base
    kept = batch.extract_cache(1)
    del batch
    gc.collect()
    held = mx.get_active_memory() - base
    assert held < whole * 0.4, f"kept {held >> 20} MB of a {whole >> 20} MB batch"
    assert kept[0][1].cache[0].shape == (1, 1024, 1024)

