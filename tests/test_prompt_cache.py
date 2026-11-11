import pytest

from aris_server.engine import MovingPromptCache


class Layer:
    def __init__(self, nbytes, trimmable=False):
        self.nbytes, self._trimmable = nbytes, trimmable

    def is_trimmable(self):
        return self._trimmable


SYSTEM = list(range(100))
TURN1 = SYSTEM + list(range(100, 300))


def test_a_conversation_moves_to_its_next_turn_and_the_system_prompt_is_copied():
    cache = MovingPromptCache(max_size=10)
    sys_layers, conv_layers = [Layer(10)], [Layer(50)]
    cache.insert_cache("m", SYSTEM, sys_layers, cache_type="system")
    cache.insert_cache("m", TURN1, conv_layers, cache_type="assistant")
    assert cache.nbytes == 60

    got, rest = cache.fetch_nearest_cache("m", TURN1 + [7, 8])
    assert got is conv_layers and rest == [7, 8]        # handed over, not copied
    assert cache.nbytes == 10 and len(cache) == 1       # no longer counted twice

    got, rest = cache.fetch_nearest_cache("m", SYSTEM + [5])
    assert got is not sys_layers and rest == [5]        # a copy: other conversations still start from it
    assert cache.nbytes == 10 and len(cache) == 1


def test_exact_match_and_no_match():
    cache = MovingPromptCache(max_size=10)
    layers = [Layer(5)]
    cache.insert_cache("m", TURN1, layers, cache_type="user")
    assert cache.fetch_nearest_cache("m", TURN1) == (layers, [])
    assert cache.fetch_nearest_cache("m", TURN1) == (None, TURN1)


def test_only_the_system_prompt_segment_is_snapshotted_mid_turn():
    from aris_server.engine import merge_prompt_tail

    prompt = list(range(10))
    tokenized = (prompt, [prompt[:3], prompt[3:8], prompt[8:]], ["system", "user", "assistant"], "reasoning", None)
    assert merge_prompt_tail(tokenized) == (prompt, [prompt[:3], prompt[3:]], ["system", "assistant"], "reasoning", None)
    plain = (prompt, [prompt], ["assistant"], "normal", None)
    assert merge_prompt_tail(plain) == plain
    no_tail = (prompt, [prompt[:3], prompt[3:]], ["system", "user"], "normal", None)
    assert merge_prompt_tail(no_tail) == no_tail  # the last segment is never snapshotted mid-turn


def test_the_prompt_cache_shrinks_by_how_far_memory_got_into_the_headroom():
    from aris_server.engine import shrink_cap

    gb = 1 << 30
    assert shrink_cap(5 * gb, peak=30 * gb, limit=37 * gb, headroom=2 * gb) == 5 * gb       # plenty of room
    assert shrink_cap(5 * gb, peak=36 * gb, limit=37 * gb, headroom=2 * gb) == 4 * gb       # 1 GB too close
    assert shrink_cap(2 * gb, peak=40 * gb, limit=37 * gb, headroom=2 * gb) == 1 * gb       # never below the floor


def test_a_batch_that_moves_whole_to_generation_hands_its_cache_over():
    from mlx_lm.generate import PromptProcessingBatch

    import aris_server.engine  # noqa: F401  (installs the patch)

    def batch(uids):
        b = PromptProcessingBatch.__new__(PromptProcessingBatch)
        b.uids, b.prompt_cache, b.tokens = list(uids), [Layer(5)], [[1]] * len(uids)
        b.samplers = b.logits_processors = b.max_tokens = b.stop_sequences = [None] * len(uids)
        return b

    whole = batch([7])
    layers = whole.prompt_cache
    moved = whole.split([0])
    assert moved.prompt_cache is layers and moved.uids == [7]      # handed over, not deep-copied
    assert whole.uids == [] and whole.prompt_cache == []


def test_a_finished_turn_larger_than_the_budget_stays_and_older_entries_go():
    cache = MovingPromptCache(max_size=10, max_bytes=50)
    cache.insert_cache("m", SYSTEM, [Layer(10)], cache_type="system")
    cache.insert_cache("m", [7] * 50, [Layer(20)], cache_type="assistant")     # an older conversation
    turn = [Layer(60)]
    cache.insert_cache("m", TURN1, turn, cache_type="assistant")              # bigger than the whole budget
    assert cache.fetch_nearest_cache("m", TURN1 + [1])[0] is turn               # still there for the next turn
    assert cache.fetch_nearest_cache("m", [7] * 50 + [1])[0] is None            # the older one went instead


def test_the_context_that_fits_leaves_room_for_the_peak_reading_a_prompt_reaches():
    from aris_server.engine import CONTEXT_OVERHEAD, context_that_fits

    gb = 1 << 30
    # A 48 GB Mac (37.4 GB for the GPU) running Arete-27B at 8 bits: 64 KB of KV cache per token. Cold prompts of
    # 30k and 53k tokens peaked at 31.8 and 34.3 GB there, so a 64.5k one peaks near 35.6 GB.
    assert context_that_fits(int(37.4 * gb), int(26.6 * gb), 65536, CONTEXT_OVERHEAD) == 65536
    assert context_that_fits(int(10 * gb), int(9 * gb), 65536, CONTEXT_OVERHEAD) == 0


def _batch(rows):
    """A prompt batch of `rows` sequences: a KV layer and a recurrent layer, 4 MB of recurrent state per row."""
    import mlx.core as mx
    from mlx_lm.generate import PromptProcessingBatch
    from mlx_lm.models.cache import ArraysCache, BatchKVCache

    kv = BatchKVCache([0] * rows)
    kv.update_and_fetch(mx.random.normal((rows, 2, 3000, 64)), mx.random.normal((rows, 2, 3000, 64)))
    state = ArraysCache(2)
    state.cache = [mx.random.normal((rows, 1024, 1024)), mx.random.normal((rows, 4, 64))]
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.uids, batch.prompt_cache = list(range(rows)), [kv, state]
    mx.eval(kv.state, state.cache)
    return batch


def test_a_conversation_kept_from_a_shared_batch_does_not_keep_the_whole_batch_alive():
    import gc

    import mlx.core as mx

    import aris_server.engine  # noqa: F401  installs the extract patch

    mb = 1 << 20
    base = mx.get_active_memory()
    batch = _batch(4)
    whole = mx.get_active_memory() - base
    kept = batch.extract_cache(1)  # a conversation finished; mlx-lm keeps its cache for the next turn
    del batch
    gc.collect()
    held = mx.get_active_memory() - base
    # Its own quarter of the batch, not the batch: mlx-lm cuts recurrent state as a view of the batch's rows.
    assert held < whole * 0.4, f"kept {held / mb:.0f} MB of a {whole / mb:.0f} MB batch"
    assert kept[1].cache[0].shape == (1, 1024, 1024)


def test_a_conversation_alone_in_its_batch_is_not_copied_when_kept():
    import mlx.core as mx

    import aris_server.engine  # noqa: F401

    batch = _batch(1)
    before = mx.get_active_memory()
    kept = batch.extract_cache(0)
    # The batch is the conversation: copying it would hold its cache twice at the end of every turn.
    assert mx.get_active_memory() - before < 1 << 20
    assert kept[0].offset == 3000


def test_a_model_that_cannot_fit_stops_the_server_with_the_reason_instead_of_running_without_a_limit(caplog):
    from aris_server.engine import NotEnoughMemory, check_memory

    gb = 1 << 30
    # Arete-27B at 8 bits on a 32 GB Mac (about 21 GB for the GPU): the weights alone are more than it can hold.
    with pytest.raises(NotEnoughMemory, match=r"arete:27b does not fit .* 26\.6 GB, the GPU can use 21\.3 GB"):
        check_memory("arete:27b", int(21.3 * gb), int(26.6 * gb))
    # The weights fit, but not a single block of context beside them: still no use, and no limit would mean OOM.
    with pytest.raises(NotEnoughMemory):
        check_memory("arete:27b", int(30.2 * gb), int(26.6 * gb), fits=0)
    # Room for a little context: it serves, and says what to change.
    caplog.set_level("WARNING")
    check_memory("arete:27b-4bit", int(21.3 * gb), int(15.1 * gb), fits=8192)
    assert "only about 8192 tokens of context fit" in caplog.text
    caplog.clear()
    check_memory("arete:27b", int(37.4 * gb), int(26.6 * gb), fits=62464)
    assert not caplog.text


def test_the_weights_a_model_will_take_are_known_from_its_files_before_it_loads(tmp_path):
    from aris_server.engine import weights_on_disk

    model, adapter = tmp_path / "model", tmp_path / "adapter"
    for folder, sizes in ((model, [5 << 30, 4 << 30]), (adapter, [1 << 28])):
        folder.mkdir()
        for i, size in enumerate(sizes):
            with open(folder / f"part-{i}.safetensors", "wb") as f:
                f.truncate(size)  # sparse: no disk used
        (folder / "config.json").write_text("{}")
    assert weights_on_disk(str(model)) == 9 << 30
    assert weights_on_disk(str(model), str(adapter)) == (9 << 30) + (1 << 28)
    assert weights_on_disk("NousResearch/Arete-27B") is None  # a Hugging Face repo id: checked once loaded

    # An adapter folder as mlx-lm's trainer leaves it: only adapters.safetensors loads, not the checkpoints beside it.
    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "adapter_config.json").write_text("{}")
    for name in ("adapters.safetensors", "adapters-step75.safetensors"):
        with open(trained / name, "wb") as f:
            f.truncate(1 << 28)
    assert weights_on_disk(str(model), str(trained)) == (9 << 30) + (1 << 28)


def _arete_folder(folder, weight_bytes):
    """Weights the size of Arete-27B at some quantization, with its config: 16 of 64 layers keep a KV cache."""
    import json

    folder.mkdir()
    with open(folder / "model.safetensors", "wb") as f:
        f.truncate(weight_bytes)  # sparse: no disk used
    layers = (["linear_attention"] * 3 + ["full_attention"]) * 16
    (folder / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 64, "layer_types": layers, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "hidden_size": 5120}}))
    return str(folder)


@pytest.mark.parametrize("gpu_gb,kv_bits,chosen", [
    (37.4, 8, "8bit"),    # a 48 GB Mac: about 125k tokens beside the 8-bit weights
    (27.0, 8, "4bit"),    # a 36 GB Mac: the 8-bit weights leave no room to read a prompt
    (32.0, 8, "4bit"),    # 8-bit would leave about 32k tokens, 4-bit about 229k: context wins
    (21.3, 8, "4bit"),    # a 32 GB Mac: only the 4-bit fits, with about 45k tokens (short of 64k, but the most)
    (21.3, None, "4bit"),
    (10.0, 8, "4bit"),    # nothing fits: the smaller one, whose refusal names the smallest option
])
def test_a_model_given_several_weights_loads_the_best_this_mac_holds_with_room_for_a_conversation(
        tmp_path, gpu_gb, kv_bits, chosen):
    from aris_server.config import ModelSpec
    from aris_server.engine import choose_weights

    gb = 1 << 30
    q8 = _arete_folder(tmp_path / "8bit", int(26.6 * gb))
    q4 = _arete_folder(tmp_path / "4bit", int(15.1 * gb))
    spec = ModelSpec(name="arete:27b", path=q8, choices=(q8, "NousResearch/Arete-27B", q4), kv_bits=kv_bits)
    assert choose_weights(spec, int(gpu_gb * gb)).path == str(tmp_path / chosen)  # a repo id can't be sized: skipped


def test_a_model_whose_own_context_is_short_takes_the_best_weights_that_hold_it(tmp_path):
    from aris_server.config import ModelSpec
    from aris_server.engine import choose_weights

    gb = 1 << 30
    q8 = _arete_folder(tmp_path / "8bit", int(26.6 * gb))
    q4 = _arete_folder(tmp_path / "4bit", int(15.1 * gb))
    # The 8-bit weights leave about 41k tokens on this Mac: short of 64k, but more than a 32k model can use anyway.
    spec = ModelSpec(name="m", path=q8, choices=(q8, q4), kv_bits=8, context=32768)
    assert choose_weights(spec, int(32.5 * gb)).path == q8


@pytest.mark.parametrize("prompt_tokens,older_kept", [(4_000, False), (100, True)])
def test_room_for_the_cache_a_prompt_builds_is_made_before_the_prompt_is_read(prompt_tokens, older_kept):
    # mlx-lm trims saved entries to the budget minus what active sequences hold right after a request takes its
    # cache, before the prompt is read: a cold 53k-token prompt then built 3.2 GB of cache beside 3.5 GB of saved
    # ones and peaked at 36.9 of 37.4 GB on a 48 GB Mac. What the prompt is about to build has to count too.
    cache = MovingPromptCache(max_size=10, max_bytes=100_000)
    cache.kv_bytes_per_token = 10
    cache.insert_cache("m", SYSTEM, [Layer(1_000)], cache_type="system")
    cache.insert_cache("m", [7] * 50, [Layer(40_000)], cache_type="assistant")      # an older conversation
    prompt = [9] * prompt_tokens                                                    # nothing in memory matches
    assert cache.fetch_nearest_cache("m", prompt) == (None, prompt)
    cache.trim_to(n_bytes=cache.max_bytes)                  # mlx-lm's next call: nothing is active yet
    # 4,000 tokens plus a growth block build 80,960 bytes, so the older conversation has to make room; 100 tokens
    # build 41,960, which fits beside it. The system prompt stays either way.
    assert cache.nbytes == (41_000 if older_kept else 1_000)
