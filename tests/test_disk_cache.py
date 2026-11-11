import os
import shutil
import threading

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from aris_server.disk_cache import DiskCache, _claim, model_id
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
    assert len(list(tmp_path.rglob("*.safetensors"))) == 1  # the system prompt; turns wait in memory
    cache.flush()                                            # a graceful stop
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


def test_a_finished_turn_reaches_disk_only_when_memory_lets_it_go(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    assert cache.disk.index == {}                                       # its next turn usually takes it first
    cache.trim_to(n_bytes=0)                                            # evicted: written on the way out
    assert [len(t) for t, _ in cache.disk.index.values()] == [2200]
    taken = cache_with_disk(tmp_path / "b")
    taken.insert_cache("m", TURN2, layers(2700, 3), cache_type="assistant")
    assert taken.fetch_nearest_cache("m", TURN2 + [1])[1] == [1]        # taken by the next turn: never written
    assert taken.disk.index == {}


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
    cache.flush()
    (meta,) = list(tmp_path.rglob("*.json"))
    meta.unlink()
    assert DiskCache(tmp_path, "model-a", max_bytes=1 << 30).index == {}


def test_partial_writes_are_cleaned_up_at_startup(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.flush()
    (meta,) = list(tmp_path.rglob("*.json"))
    meta.unlink()
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30)
    assert list(tmp_path.rglob("*.safetensors")) == []


def test_a_conversation_evicted_from_memory_comes_back_from_disk(tmp_path):
    from mlx_lm.models.cache import LRUPromptCache

    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")
    cache.insert_cache("m", TURN2, layers(2700, 3), cache_type="assistant")
    cache.trim_to(n_bytes=cache._n_bytes_by_type["system"])          # memory keeps only the system prompt
    got, rest = cache.fetch_nearest_cache("m", TURN2 + [1, 2])
    assert rest == [1, 2] and got[0].offset == 2700 and mx.all(got[1][1] == 3).item()

    short = cache_with_disk(tmp_path)                                  # a gain under DISK_MIN_GAIN stays in memory
    LRUPromptCache.insert_cache(short, "m", TURN2[:-100], layers(2600, 4), cache_type="assistant")
    got, rest = short.fetch_nearest_cache("m", TURN2 + [1])
    assert len(rest) == 101 and mx.all(got[1][1] == 4).item()


def test_a_snapshot_of_a_prompt_still_being_answered_goes_to_disk_only(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.snapshot_min_tokens = 1024
    prompt = TURN2
    cache.reading.append(prompt)
    cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")           # shared: stays in memory
    cache.insert_cache("m", prompt[:-1], layers(2699, 2), cache_type="assistant")    # the request holds a copy
    assert len(cache) == 1 and sorted(len(t) for t, _ in cache.disk.index.values()) == [1200, 2699]
    cache.reading.remove(prompt)
    cache.insert_cache("m", prompt + [9, 9], layers(2702, 3), cache_type="assistant")  # the finished turn
    assert len(cache) == 2
    got, rest = cache_with_disk(tmp_path).fetch_nearest_cache("m", prompt)          # a retry finds it on disk
    assert rest == prompt[-1:] and got[0].offset == 2699


def test_a_retry_and_a_continuation_both_find_a_snapshot(tmp_path):
    prompt = TURN1
    answered = TURN1 + [42, 43]
    cache = cache_with_disk(tmp_path)
    cache.snapshot_min_tokens = 1024
    cache.reading.append(prompt)
    cache.insert_cache("m", prompt[:-1], layers(2199, 1), cache_type="assistant")   # prompt read
    cache.reading.remove(prompt)
    cache.insert_cache("m", answered, layers(2202, 2), cache_type="assistant")      # answer finished
    cache.flush()
    fresh = MovingPromptCache(max_size=10)
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30).restore(
        lambda t, c, kind: fresh.insert_cache("m", t, c, cache_type=kind), budget=1 << 30)
    assert fresh.fetch_nearest_cache("m", prompt)[1] == prompt[-1:]               # the same request again
    assert fresh.fetch_nearest_cache("m", answered + [7])[1] == [7]               # the next turn


def test_a_third_turn_drops_the_first_turns_snapshots(tmp_path):
    cache = cache_with_disk(tmp_path)
    for n, length in enumerate([1500, 1600, 1700, 1800]):
        cache.insert_cache("m", list(range(length)), layers(length, n), cache_type="assistant")
    cache.flush()
    assert sorted(len(t) for t, _ in cache.disk.index.values()) == [1700, 1800]


def test_restored_caches_can_be_used_from_another_thread(tmp_path):
    import threading

    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.flush()
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


def test_a_finished_turn_that_diverged_from_its_history_is_dropped_before_the_disk_copy_loads(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.snapshot_min_tokens = 1024
    cache.reading.append(TURN2)
    cache.insert_cache("m", TURN2[:-1], layers(2699, 1), cache_type="assistant")       # prompt snapshot, on disk
    cache.reading.remove(TURN2)
    generated = TURN2 + [5, 6, 7]                                                      # tokens as generated
    cache.insert_cache("m", generated, layers(2703, 2), cache_type="assistant")
    rendered = TURN2 + [5, 67] + [8]                                                   # the same text, re-tokenized
    got, rest = cache.fetch_nearest_cache("m", rendered)
    assert got[0].offset == 2699 and rest == rendered[2699:]                           # resumed from the snapshot
    assert len(cache) == 0 and not cache.disk.has(generated)                           # the dead turn is gone


def test_a_diverged_turn_is_rebuilt_to_its_prompt_end_in_memory_without_a_disk_snapshot(tmp_path):
    cache = cache_with_disk(tmp_path)
    prompt = TURN2
    cache.reading.append(prompt)
    cache.insert_cache("m", prompt[:-1], layers(2699, 1), cache_type="assistant")  # prompt end: recurrent state 1
    assert cache.disk.index == {}                                                    # no snapshot written
    cache.reading.remove(prompt)
    generated = prompt + [5, 6, 7]
    cache.insert_cache("m", generated, layers(2703, 2), cache_type="assistant")      # finished: state moved on
    rendered = prompt + [5, 67, 8]                                                   # same text, re-tokenized
    got, rest = cache.fetch_nearest_cache("m", rendered)
    assert got[0].offset == 2699 and rest == rendered[2699:]                         # attention trimmed
    assert mx.all(got[1][1] == 1).item()                                              # recurrent state put back
    assert len(cache) == 0 and cache.disk.index == {}


def test_only_the_latest_turns_prompt_end_state_is_kept():
    cache = MovingPromptCache(max_size=10)
    convo = list(range(2000))
    for turn in range(4):                                       # four turns of one conversation
        prompt = convo + [9000 + turn]
        cache.reading.append(prompt)
        cache.insert_cache("m", prompt[:-1], layers(len(prompt) - 1, turn), cache_type="assistant")
        cache.reading.remove(prompt)
        convo = prompt + [7, 7]
        cache.insert_cache("m", convo, layers(len(convo), turn), cache_type="assistant")
    assert len(cache._turn_state) == 1 and len(cache._prompt_state) == 0
    cache.fetch_nearest_cache("m", convo + [1])                 # continued as-is: no rebuild needed
    assert cache._turn_state == {}


def saved(disk, start, seed, age):
    """A 2000-token conversation saved by `disk`, its file dated `age` seconds after the epoch."""
    tokens = tuple(range(start, start + 2000))
    disk.save(list(tokens), layers(2000, seed), "assistant")
    (data,) = [p for p, (t, _) in disk.index.items() if t == tokens]
    os.utime(data, (age, age))
    return data


def test_one_cap_covers_every_setup_and_other_setups_caches_go_first(tmp_path):
    before = DiskCache(tmp_path, "model-b", max_bytes=1 << 30)            # the setup served before this one
    old, new = saved(before, 0, 1, age=1000), saved(before, 10000, 2, age=2000)
    size = old.stat().st_size
    del before                                                             # that server stopped
    here = DiskCache(tmp_path, "model-a", max_bytes=int(2.5 * size))
    assert old.exists() and new.exists()                                   # under the cap: kept for a swap back
    mine = saved(here, 20000, 3, age=3000)
    assert not old.exists() and new.exists()                               # over it: the other setup's oldest goes
    saved(here, 30000, 4, age=4000)
    assert not new.parent.exists()                                         # then its last, and its folder
    saved(here, 40000, 5, age=5000)
    assert not mine.exists() and len(here.index) == 2                      # then this setup's own oldest


def test_a_server_started_on_another_setup_clears_what_the_last_one_left_over_the_cap(tmp_path):
    before = DiskCache(tmp_path, "model-b", max_bytes=1 << 30)
    files = [saved(before, n * 10000, n, age=1000 * (n + 1)) for n in range(3)]
    del before
    (files[0].parent / "lock").unlink()                                    # a folder from before the lock
    other = tmp_path / "not-a-cache"
    other.mkdir()
    (other / "x.safetensors").write_bytes(b"x" * (1 << 20))
    DiskCache(tmp_path, "model-a", max_bytes=int(1.5 * files[0].stat().st_size))
    assert [p.exists() for p in files] == [False, False, True]             # the newest fits under the cap
    assert (other / "x.safetensors").exists()                              # only setup folders are touched


def test_a_running_servers_caches_stay_whatever_another_server_saves(tmp_path):
    running = DiskCache(tmp_path, "model-b", max_bytes=1 << 30)
    theirs = saved(running, 0, 1, age=1000)
    here = DiskCache(tmp_path, "model-a", max_bytes=1)
    here.save(list(range(10000, 12000)), layers(2000, 2), "assistant")
    assert theirs.exists() and here.index == {}                            # over the cap, only its own can go


def test_a_server_starting_while_another_clears_its_folder_holds_the_folder_made_again(tmp_path):
    folder = tmp_path / model_id("model-a")
    folder.mkdir()
    clearing = _claim(folder)                                              # another server clearing the folder
    started = []
    t = threading.Thread(target=lambda: started.append(DiskCache(tmp_path, "model-a", max_bytes=1 << 30)))
    t.start()
    t.join(0.5)
    assert t.is_alive()                                                    # it waits for the clearing to finish
    (folder / "lock").unlink()
    folder.rmdir()
    clearing.close()
    t.join(5)
    assert started and folder.exists() and _claim(folder) is None          # and holds the folder made again


def test_a_cache_gone_from_disk_is_forgotten_before_memory_lets_go_of_its_own(tmp_path):
    cache = cache_with_disk(tmp_path)
    longer = TURN2 + list(range(8000, 9500))                                    # 4200 tokens
    cache.insert_cache("m", longer, layers(4200, 3), cache_type="assistant")
    cache.trim_to(n_bytes=0)                                                    # evicted: on disk
    (data,) = cache.disk.index
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")     # an earlier turn, in memory
    data.unlink()                                                               # the cache deleted by hand
    got, rest = cache.fetch_nearest_cache("m", longer + [1])
    assert got is not None and got[0].offset == 2200 and rest == longer[2200:] + [1]  # memory's turn, not a re-read
    assert cache.disk.index == {} and not data.with_suffix(".json").exists()    # forgotten: no later turn looks


def test_a_damaged_cache_is_dropped_and_the_turn_reads_on_from_what_memory_has(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")
    cache.insert_cache("m", TURN2, layers(2700, 3), cache_type="assistant")
    cache.trim_to(n_bytes=cache._n_bytes_by_type["system"])                     # memory keeps only the system prompt
    (data,) = [p for p, (t, _) in cache.disk.index.items() if t == tuple(TURN2)]
    data.write_bytes(b"damaged")
    got, rest = cache.fetch_nearest_cache("m", TURN2 + [1, 2])
    assert got is not None and got[0].offset == 1200 and rest == TURN2[1200:] + [1, 2]  # not from the first token
    assert not data.exists() and not cache.disk.has(TURN2)                      # dropped: no later turn tries it


def test_a_restarted_server_drops_a_damaged_cache_and_restores_the_rest(tmp_path):
    cache = cache_with_disk(tmp_path)
    other = list(range(9000, 11000))
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.insert_cache("m", other, layers(2000, 4), cache_type="assistant")
    cache.flush()
    (damaged,) = [p for p, (t, _) in cache.disk.index.items() if t == tuple(TURN1)]
    damaged.write_bytes(b"damaged")
    restored = []
    DiskCache(tmp_path, "model-a", max_bytes=1 << 30).restore(lambda t, c, kind: restored.append(t), budget=1 << 30)
    assert restored == [other] and not damaged.exists()


def test_a_cache_that_cannot_be_written_never_stops_generation(tmp_path, caplog):
    caplog.set_level("WARNING")
    cache = cache_with_disk(tmp_path)
    cache.disk.dir.chmod(0o500)                                                 # MLX raises RuntimeError, not OSError
    try:
        cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")    # used to raise into mlx-lm's loop
    finally:
        cache.disk.dir.chmod(0o700)
    assert "could not save a prompt cache to disk" in caplog.text
    assert len(cache) == 1 and cache.disk.index == {}                           # kept in memory all the same


def test_a_cache_folder_deleted_while_serving_is_made_again_and_held(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", SYSTEM, layers(1200, 1), cache_type="system")       # written at once
    shutil.rmtree(cache.disk.dir)                                               # deleted by hand
    assert not cache.disk.has(SYSTEM)                                           # gone: to be saved again
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.trim_to(n_bytes=0)                                                    # both evicted: written on the way out
    assert sorted(len(t) for t, _ in cache.disk.index.values()) == [1200, 2200]
    assert all(p.exists() for p in cache.disk.index)
    assert _claim(cache.disk.dir) is None                                       # held again: no other server clears it


def test_a_cache_whose_metadata_was_cut_off_is_dropped_at_startup(tmp_path):
    cache = cache_with_disk(tmp_path)
    other = list(range(9000, 11000))
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.insert_cache("m", other, layers(2000, 4), cache_type="assistant")
    cache.flush()
    (data,) = [p for p, (t, _) in cache.disk.index.items() if t == tuple(TURN1)]
    meta = data.with_suffix(".json")
    meta.write_text(meta.read_text()[:100])                                     # killed mid-write
    fresh = DiskCache(tmp_path, "model-a", max_bytes=1 << 30)                   # raised at every start
    assert [t for t, _ in fresh.index.values()] == [tuple(other)]
    assert not data.exists() and not meta.exists()


def test_a_read_only_cache_folder_never_stops_startup(tmp_path):
    cache = cache_with_disk(tmp_path)
    cache.insert_cache("m", TURN1, layers(2200, 2), cache_type="assistant")
    cache.flush()
    (data,) = cache.disk.index
    data.with_suffix(".json").unlink()                                          # a write cut off by a kill
    cache.disk.dir.chmod(0o500)
    try:
        fresh = DiskCache(tmp_path, "model-a", max_bytes=1 << 30)               # raised: it couldn't clean up
    finally:
        cache.disk.dir.chmod(0o700)
    assert fresh.index == {}


def test_another_setups_folder_that_cannot_be_cleared_is_left_at_startup(tmp_path):
    before = DiskCache(tmp_path, "model-b", max_bytes=1 << 30)
    data = saved(before, 0, 1, age=1000)
    del before
    data.unlink()                                                               # its caches gone; metadata, lock left
    data.parent.chmod(0o500)
    try:
        DiskCache(tmp_path, "model-a", max_bytes=1 << 30)                       # raised: it couldn't clear the folder
    finally:
        data.parent.chmod(0o700)
    assert data.with_suffix(".json").exists()
