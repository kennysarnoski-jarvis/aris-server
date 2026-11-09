import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from aris_server.disk_cache import DiskCache
from aris_server.engine import MovingPromptCache


def layers(n_tokens, seed):
    kv = KVCache()
    kv.update_and_fetch(mx.full((1, 2, n_tokens, 4), seed, mx.float16), mx.full((1, 2, n_tokens, 4), -seed, mx.float16))
    lin = ArraysCache(2)
    lin[0], lin[1] = mx.full((1, 3), seed, mx.float32), mx.full((1, 2, 2), seed, mx.float32)
    return [kv, lin]


SYSTEM = list(range(1200))
TURN1 = SYSTEM + list(range(5000, 6000))
TURN2 = TURN1 + list(range(7000, 7500))


def cache_with_disk(tmp_path):
    cache = MovingPromptCache(max_size=10)
    cache.disk = DiskCache(tmp_path, "model-a", max_bytes=1 << 30)
    return cache


def test_a_restarted_server_resumes_the_conversation_from_disk(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.insert_cache("m", TURN2, layers(2700, 3), cache_type="assistant")
    assert len(list(tmp_path.rglob("*.safetensors"))) == 3  # system, turn 1 (longest earlier snapshot), turn 2

    fresh = MovingPromptCache(max_size=10)
    disk = DiskCache(tmp_path, "model-a", max_bytes=1 << 30)
    disk.restore(lambda t, c, kind: fresh.insert_cache("m", t, c, cache_type=kind), budget=1 << 30)
    got, rest = fresh.fetch_nearest_cache("m", TURN2 + [1, 2])
    assert rest == [1, 2]
    kv, lin = got
    assert kv.offset == 2700 and mx.all(kv.state[0][..., :2700, :] == 3).item()
    assert mx.all(lin[1] == 3).item()
    got, rest = fresh.fetch_nearest_cache("m", SYSTEM + [9])
    assert rest == [9] and got[0].offset == 1200


def test_short_prompts_and_mid_turn_snapshots_stay_off_disk(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", list(range(100)), layers(100, 1), cache_type="assistant")
    cache.insert_cache("m", TURN1, layers(2200, 1), cache_type="user")
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_other_weights_never_see_these_caches_and_the_disk_cap_holds(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    assert DiskCache(tmp_path, "model-b", max_bytes=1 << 30).index == {}
    small = DiskCache(tmp_path, "model-a", max_bytes=1)
    small.save(list(range(9000, 11000)), layers(2000, 4), "assistant")
    assert len(small.index) == 0 or sum(p.stat().st_size for p in small.index) <= 1 << 20


def test_an_interrupted_write_is_ignored(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    (meta,) = list(tmp_path.rglob("*.json"))
    meta.unlink()
    assert DiskCache(tmp_path, "model-a", max_bytes=1 << 30).index == {}


def test_partial_writes_are_cleaned_up_at_startup(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    (meta,) = list(tmp_path.rglob("*.json"))
    meta.unlink()
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30)
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_a_retry_and_a_continuation_both_find_a_snapshot(tmp_path):
    prompt = TURN1
    answered = TURN1 + [42, 43]
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", prompt[:-1], layers(2199, 1), cache_type="assistant")   # prompt read
    cache.insert_cache("m", answered, layers(2202, 2), cache_type="assistant")      # answer finished
    fresh = MovingPromptCache(max_size=10)
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30).restore(
        lambda t, c, kind: fresh.insert_cache("m", t, c, cache_type=kind), budget=1 << 30)
    assert fresh.fetch_nearest_cache("m", prompt)[1] == prompt[-1:]               # the same request again
    assert fresh.fetch_nearest_cache("m", answered + [7])[1] == [7]               # the next turn


def test_a_third_turn_drops_the_first_turns_snapshots(tmp_path):
    cache = cache_with_disk(tmp_path)
    for n, length in enumerate([1500, 1600, 1700, 1800]):
        cache.insert_cache("m", list(range(length)), layers(length, n), cache_type="assistant")
    assert sorted(len(t) for t, _ in cache.disk.index.values()) == [1700, 1800]


def test_restored_caches_can_be_used_from_another_thread(tmp_path):
    import threading

    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    fresh = MovingPromptCache(max_size=10)
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30).restore(
        lambda t, c, kind: fresh.insert_cache("m", t, c, cache_type=kind), budget=1 << 30)
    result = []

    def use():  # generation runs on its own thread with its own streams
        with mx.stream(mx.new_stream(mx.cpu)):
            kv, _ = fresh.fetch_nearest_cache("m", TURN1 + [1])[0]
            result.append(mx.sum(kv.state[0][..., :2200, :]).item())

    t = threading.Thread(target=use)
    t.start()
    t.join()
    assert result == [2 * 2200 * 2 * 4]


def test_an_adapter_gets_its_own_caches(tmp_path):
    from aris_server.disk_cache import model_id

    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"x" * 10)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"y" * 10)
    assert model_id(str(base)) != model_id(str(base), str(adapter))
    before = model_id(str(base), str(adapter))
    (adapter / "adapters.safetensors").write_bytes(b"y" * 11)  # retrained: new weights, new caches
    assert model_id(str(base), str(adapter)) != before
