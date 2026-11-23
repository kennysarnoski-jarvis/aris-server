"""A request that leaves every cached conversation early (a client that rewrites part of its system prompt on each
request, say) reuses what it still shares with them. A hybrid model's linear-attention layers carry a running state
that can't be cut back to an earlier token, and Inkling's sliding windows keep only their last rows, so that reuse
comes from checkpoints of that state taken while prompts are read, beside the attention cache cut back to the same
token.

These tests drive a tiny model with Arete's architecture through mlx-lm's own batch generator, the way mlx-lm's server
drives it (ResponseGenerator._generate, batched path), so the checkpoints are taken where a real prompt read takes
them."""

import logging

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache

from mlx_lm.models.cache import CacheList, KVCache, QuantizedKVCache, RotatingKVCache

from aris_server import disk_cache, dspark, inkling, mtp, mtp_serve
from aris_server.engine import (MovingPromptCache, _Checkpoint, _checkpointable, _flat_caches, _split_segments,
                                merge_prompt_tail)
from aris_server.kv_quant import quantize_kv
from aris_server.speculative import Sampling
from test_dspark import random_drafter
from test_inkling2_model import LOOP_CONFIG, create_model
from tiny_models import tiny_hybrid_model

END = 9       # the tiny vocabulary's end of a turn (<|im_end|>)
NEWLINE = 10  # and its line break, for finding where a changed line starts
STEP = 64     # the checkpoint grid, scaled down from 512 with everything else


def decode(ids):
    return "".join("\n" if i == NEWLINE else "x" for i in ids)


def text(seed: int, n: int) -> list[int]:
    """n tokens of 'text' (no line breaks), different for each seed."""
    return [int(t) for t in mx.random.randint(11, 64, (n,), key=mx.random.key(seed)).tolist()]


def lines(seed: int, n_lines: int, width: int = 20) -> list[int]:
    return [tok for i in range(n_lines) for tok in text(seed * 1000 + i, width - 1) + [NEWLINE]]


TOOLS = lines(1, 12)                                   # 240 tokens: tool definitions, the same on every request
GUIDANCE = lines(2, 2)                                 # 40 more of system prompt that never changes
LEDGER = text(3, 6) + [21, 22, 23] + text(4, 8) + [NEWLINE]   # a line the client rewrites: "passed=0, pending=0"
SYSTEM_TAIL = lines(5, 1) + [END]
LEDGER_AT = len(TOOLS) + len(GUIDANCE)                 # 280: where the changing line starts
USER = text(6, 30)
TOOL_RESULT = text(7, 25)


def system(ledger=LEDGER):
    return TOOLS + GUIDANCE + ledger + SYSTEM_TAIL


def changed(position: int, value: int):
    """The ledger with one token of it rewritten."""
    ledger = list(LEDGER)
    ledger[position] = value
    return ledger


def cache_for(model, budget=1 << 30):
    cache = MovingPromptCache(max_size=10, max_bytes=budget)
    cache.decode = decode
    cache.light_min_tokens = cache.checkpoint_min_gain = 16
    if _checkpointable(model.make_cache()):  # as the engine decides
        cache.checkpoint_every, cache.system_ends = STEP, frozenset({END})
    return cache


def serve(cache, model, prompt, system_tokens=None, max_tokens=4, sampler=None):
    """One request, the way mlx-lm's server serves it: its prompt split into segments (the system prompt, then the
    rest; a request ending in tool results is one segment), the nearest cache fetched, the prompt read with a copy of
    the cache saved at each segment's end, then the finished sequence's cache saved. Returns (cached tokens, generated
    tokens, their logprobs). The finished sequence's cache is left in serve.final."""
    if system_tokens:
        segments = [prompt[:system_tokens], prompt[system_tokens:-2], prompt[-2:]]
        types = ["system", "user", "assistant"]
    else:
        segments, types = [prompt], ["assistant"]
    tokenized = merge_prompt_tail((prompt, segments, types, "normal", None))
    cache.reading = cache.reading[-15:] + [prompt]  # what the engine's tokenize hook does
    cache.tokenized = tokenized
    prompt, segments, types = tokenized[0], tokenized[1], tokenized[2]

    reused, rest = cache.fetch_nearest_cache("m", prompt)
    n_cached = n = len(prompt) - len(rest)
    while n > 0:  # mlx-lm's server drops what the cache covers from the segments
        if n >= len(segments[0]):
            n -= len(segments.pop(0))
            types.pop(0)
        else:
            segments[0] = segments[0][n:]
            break
    gen = BatchGenerator(model, max_tokens=max_tokens, prefill_step_size=32, sampler=sampler)
    gen.insert_segments(segments=[segments], max_tokens=[max_tokens], caches=[reused], all_tokens=[prompt[:n_cached]])
    del reused
    cache.trim_to(n_bytes=cache.max_bytes - gen.prompt_cache_nbytes)
    pending = types[::-1]
    tokens, logprobs = [], []
    while True:
        prompt_responses, gen_responses = gen.next()
        ends = [r.uid for r in prompt_responses if r.end_of_segment and not r.end_of_prompt and pending]
        for _, (c, key) in gen.extract_cache(ends).items():
            cache.insert_cache("m", key[:], c, cache_type=pending.pop())
        for r in gen_responses:
            tokens.append(r.token)
            logprobs.append(r.logprobs)
            if r.finish_reason is not None:
                cache.insert_cache("m", r.all_tokens[:], r.prompt_cache, cache_type="assistant")
                serve.final = r.prompt_cache
                gen.close()
                cache.reading.remove(prompt)
                return n_cached, tokens, logprobs


def kv8_hybrid_model():
    model = tiny_hybrid_model()
    quantize_kv(model, 8)  # as Aris 3.8-27B is served: 8-bit attention cache, recurrent layers as they are
    return model


def same_as_read_cold(model, prompt, tokens, logprobs):
    """What the request produced matches what it produces with nothing cached."""
    _, want, want_logprobs = serve(cache_for(model), model, prompt)
    assert tokens == want
    for got, w in zip(logprobs, want_logprobs):
        assert mx.allclose(got, w, atol=1e-4).item()


def checkpoints(cache):
    return sorted(len(k) for k in cache._checkpoints)


def test_a_request_that_rewrites_a_line_of_its_system_prompt_reuses_the_prefix_before_it(caplog):
    model = kv8_hybrid_model()
    cache = cache_for(model)
    first = system() + USER
    cached, out, _ = serve(cache, model, first, system_tokens=len(system()))
    assert cached == 0

    # The client rewrites its ledger line (one count changed) and sends the conversation so far plus a tool result.
    second = system(changed(7, 40)) + USER + out + TOOL_RESULT
    diverges = LEDGER_AT + 7
    with caplog.at_level(logging.INFO):
        cached, got, logprobs = serve(cache, model, second)
    # It reused nothing before (the recurrent layers can't be cut back); now it resumes from the last checkpoint
    # taken before the change, and says where it left the cache.
    assert cached == diverges // STEP * STEP
    assert f"leaves a cached {len(system())}-token entry at token {diverges} and reuses {cached}" in caplog.text
    same_as_read_cold(model, second, got, logprobs)


def test_a_line_that_changed_once_is_reused_up_to_its_start_the_next_time_it_changes():
    model = kv8_hybrid_model()
    cache = cache_for(model)
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    convo = USER + out + TOOL_RESULT
    serve(cache, model, system(changed(7, 40)) + convo)            # learns where that line starts
    assert LEDGER_AT in checkpoints(cache)
    third = system(changed(2, 41)) + convo + TOOL_RESULT          # the same line, changed earlier in it
    cached, got, logprobs = serve(cache, model, third)
    assert cached == LEDGER_AT
    same_as_read_cold(model, third, got, logprobs)
    fourth = system(changed(12, 42)) + convo + TOOL_RESULT + USER  # and again: the checkpoint is as it was taken
    cached, got, logprobs = serve(cache, model, fourth)
    assert cached == LEDGER_AT
    same_as_read_cold(model, fourth, got, logprobs)
    assert checkpoints(cache) == [64, 128, 192, 256, LEDGER_AT]    # a few tokens past one: none added


def test_the_next_turn_rebuilt_to_its_prompt_end_beats_a_checkpoint_which_is_not_counted_as_used(caplog):
    model = kv8_hybrid_model()
    cache = cache_for(model)
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    second = system(changed(7, 40)) + USER + out + TOOL_RESULT
    _, answer, _ = serve(cache, model, second)                   # resumed from a checkpoint
    order = list(cache._checkpoints)
    # Its answer comes back re-rendered (the client drops thinking from history): other tokens than generated.
    third = second + [(answer[0] + 1) % 64] + TOOL_RESULT
    with caplog.at_level(logging.INFO):
        cached, got, logprobs = serve(cache, model, third)
    assert cached == len(second) - 1                              # rebuilt at its prompt's end, as before
    assert "from a checkpoint" not in caplog.text and list(cache._checkpoints) == order
    same_as_read_cold(model, third, got, logprobs)


def test_checkpoints_cover_the_system_prompt_and_not_the_conversation_after_it():
    # Past the system prompt a checkpoint holds the state of one version of it: a client that rewrites a line of
    # its system prompt would make a new one at every rewrite, pushing out the ones that cover the system prompt.
    model = kv8_hybrid_model()
    cache = cache_for(model)
    convo = USER + text(8, 400)
    serve(cache, model, system() + convo, system_tokens=len(system()))
    assert checkpoints(cache) == [64, 128, 192, 256]
    serve(cache, model, system(changed(7, 40)) + convo + TOOL_RESULT)
    assert checkpoints(cache) == [64, 128, 192, 256, LEDGER_AT]


def test_the_entry_a_request_resumes_from_stays_as_it_was_for_whatever_continues_it():
    model = kv8_hybrid_model()
    cache = cache_for(model)
    first = system() + USER
    _, out, _ = serve(cache, model, first, system_tokens=len(system()))
    entry = cache._trie.get("m", first + out)
    before = [mx.array(x) for _, x in tree_flatten([c.state for c in entry.prompt_cache]) if isinstance(x, mx.array)]

    serve(cache, model, system(changed(7, 40)) + USER + out + TOOL_RESULT)   # resumed from that entry, cut back
    after = [x for _, x in tree_flatten([c.state for c in entry.prompt_cache]) if isinstance(x, mx.array)]
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, after))

    # The conversation it came from (an older version of the ledger) still continues from all of it.
    cont = first + out + TOOL_RESULT
    cached, got, logprobs = serve(cache, model, cont)
    assert cached == len(first + out)
    same_as_read_cold(model, cont, got, logprobs)


def test_checkpoints_hold_at_most_their_share_of_the_budget_and_count_against_it():
    model = kv8_hybrid_model()
    probe = cache_for(model)
    serve(probe, model, system() + USER, system_tokens=len(system()))
    each = probe._checkpoint_bytes() // len(probe._checkpoints)

    cache = cache_for(model, budget=8 * 2 * each + 7)              # room for two checkpoints
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    assert checkpoints(cache) == [128, 256]                        # spread out, the last one kept
    assert cache._checkpoint_bytes() <= cache.max_bytes // 8
    cached, _, _ = serve(cache, model, system(changed(7, 40)) + USER + out + TOOL_RESULT)
    assert cached == 256

    # Saved entries make room for them: entries that would fit the budget if checkpoints were free don't.
    cache = cache_for(model)
    serve(cache, model, system() + USER, system_tokens=len(system()))
    held, saved = cache._checkpoint_bytes(), cache.nbytes
    cache.trim_to(n_bytes=saved + held - 1)
    assert cache.nbytes < saved and len(cache._checkpoints) == 4
    # A budget lowered after memory ran close to the GPU's limit (Engine.adapt_memory) holds fewer.
    cache.max_bytes = 8 * each
    cache.trim_to(n_bytes=cache.max_bytes)
    assert checkpoints(cache) == [256]


def test_only_the_request_just_tokenized_is_cut_for_checkpoints():
    # mlx-lm serves a request with images on its own, fetching its cache by a copy of its prompt; its segments are
    # unused there, and the plan must not carry over to the next request.
    model = kv8_hybrid_model()
    cache = cache_for(model)
    prompt = system() + USER
    segments, types = [list(prompt)], ["assistant"]
    cache.tokenized = (prompt, segments, types, "normal", None)
    cache.fetch_nearest_cache("m", prompt[:])
    assert segments == [prompt] and types == ["assistant"] and cache.tokenized is None


def test_an_attention_only_model_takes_no_checkpoints_and_is_cut_back_as_before():
    args = qwen3_5.TextModelArgs(model_type="qwen3_5", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=64,
                                 linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
                                 linear_value_head_dim=32, full_attention_interval=1)
    mx.random.seed(1)
    model = qwen3_5.TextModel(args)
    mx.eval(model.parameters())
    quantize_kv(model, 8)
    cache = cache_for(model)
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    second = system(changed(7, 40)) + USER + out + TOOL_RESULT
    cached, got, logprobs = serve(cache, model, second)
    assert cached == LEDGER_AT + 7 and cache._checkpoints == {}    # every layer cuts back to the change itself
    same_as_read_cold(model, second, got, logprobs)


def test_a_checkpoint_taken_in_a_shared_batch_holds_only_its_own_rows():
    # Two prompts are read in one batch; a lazy slice of the batch's recurrent state, even evaluated, is a view that
    # would keep every row of it alive.
    import gc

    from mlx_lm.generate import PromptProcessingBatch

    gc.collect()
    start = mx.get_active_memory()
    state = ArraysCache(2)
    state.cache = [mx.random.normal((4, 1024, 1024)), mx.random.normal((4, 4, 64))]  # 16 MB: four rows of 4 MB
    mx.eval(state.cache)
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.uids, batch.prompt_cache = list(range(4)), [state]
    cache = MovingPromptCache(max_size=10)
    cache.insert_cache("m", list(range(300)), batch.extract_cache(2), cache_type="checkpoint")
    del batch, state
    gc.collect()
    held = mx.get_active_memory() - start
    assert cache._checkpoint_bytes() == (1024 * 1024 + 4 * 64) * 4
    assert held < 6 << 20, f"a checkpoint of one row held {held >> 20} MB of a 16 MB batch"


def test_segments_are_cut_where_checkpoints_go():
    prompt = list(range(20))
    segments, types = [prompt[:8], prompt[8:19], prompt[19:]], ["system", "assistant", "assistant"]
    _split_segments(segments, types, [4, 8, 12, 16, 4])           # 8 is already a boundary: no empty segment
    assert segments == [prompt[:4], prompt[4:8], prompt[8:12], prompt[12:16], prompt[16:19], prompt[19:]]
    assert types == ["checkpoint", "system", "checkpoint", "checkpoint", "assistant", "assistant"]


def greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


greedy.mtp_sampling = Sampling(0.0)  # labelled as mlx-lm's server labels it, so draft heads speculate


def inkling_model():
    """Inkling's layout, tiny: each layer visit keeps a sliding window of 8 rows or full attention, and conv state."""
    return create_model(config=LOOP_CONFIG)


def mtp_model():
    model = tiny_hybrid_model()
    mx.random.seed(1)
    head = mtp.MTPHead(mtp.text_model(model).args)
    mx.eval(head.parameters())
    mtp_serve.attach_mtp(model, head, depth=2)
    return model


def dspark_model():
    model = inkling_model()
    mtp_serve.attach_dspark(model, random_drafter(model), depth=3)
    return model


def each_checkpoints_bytes(model) -> int:
    probe = cache_for(model)
    serve(probe, model, system() + USER, system_tokens=len(system()))
    return probe._checkpoint_bytes() // len(probe._checkpoints)


def rows(c):
    """What a layer cache holds, in order: an attention cache's rows, a window's last rows, an ArraysCache's arrays."""
    if isinstance(c, RotatingKVCache):
        return dspark.context_rows(c)
    if isinstance(c, KVCache):
        return c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
    return [a for a in c.cache if a is not None]


def assert_same_state(got, want):
    for a, b in zip(_flat_caches(got), _flat_caches(want), strict=True):
        assert type(a) is type(b) and getattr(a, "offset", None) == getattr(b, "offset", None)
        for x, y in zip(rows(a), rows(b), strict=True):
            assert x.shape == y.shape and mx.allclose(x, y, atol=1e-4).item()


def test_which_models_take_checkpoints():
    assert _checkpointable(tiny_hybrid_model().make_cache())   # recurrent layers
    assert _checkpointable(kv8_hybrid_model().make_cache())    # and 8-bit attention, as Aris 3.8-27B is served
    assert _checkpointable(mtp_model().make_cache())           # and an MTP head's cache and hidden state
    assert _checkpointable(inkling_model().make_cache())       # sliding windows inside each visit's CacheList
    assert _checkpointable(dspark_model().make_cache())        # and its draft head's own windows
    assert not _checkpointable([KVCache(), QuantizedKVCache()])  # attention cuts back to any token
    assert not _checkpointable([CacheList(RotatingKVCache(8, keep=4), ArraysCache(1))])  # a ring that keeps a prefix


def test_inklings_system_region_ends_where_its_conversation_starts():
    ids = {"<|message_user|>": [5], "<|message_model|>": [6], "<|message_tool|>": [7]}

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return ids.get(text, [1, 2])

    assert inkling.conversation_starts(Tokenizer()) == {5, 6, 7}


def test_inkling_rewriting_a_line_of_its_system_prompt_resumes_from_a_checkpoint_of_its_windows():
    # Before, Inkling took no checkpoints: its windows can't be cut back, so this request read again from token 0.
    model = inkling_model()
    cache = cache_for(model)
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    second = system(changed(7, 40)) + USER + out + TOOL_RESULT
    cached, got, logprobs = serve(cache, model, second)
    assert cached == (LEDGER_AT + 7) // STEP * STEP
    resumed = serve.final
    same_as_read_cold(model, second, got, logprobs)
    assert_same_state(resumed, serve.final)
    third = system(changed(2, 41)) + USER + out + TOOL_RESULT + USER
    cached, got, logprobs = serve(cache, model, third)
    assert cached == LEDGER_AT  # from the line it learned
    same_as_read_cold(model, third, got, logprobs)


def test_a_draft_heads_cache_resumes_with_the_model_it_drafts_for():
    """The MTP head's cache and hidden state, and Inkling's draft head's windows and state, come back at the
    checkpoint too: after resuming, each holds what it holds when the prompt is read from its first token, and it
    drafts from there."""
    for model in (mtp_model(), dspark_model()):
        cache = cache_for(model)
        _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()), sampler=greedy)
        second = system(changed(7, 40)) + USER + out + TOOL_RESULT
        rounds = model._mtp_config.stats.rounds
        cached, got, _ = serve(cache, model, second, max_tokens=12, sampler=greedy)
        resumed = serve.final
        assert cached == (LEDGER_AT + 7) // STEP * STEP and model._mtp_config.stats.rounds > rounds
        _, cold, _ = serve(cache_for(model), model, second, max_tokens=12, sampler=greedy)
        assert got == cold
        assert_same_state(resumed, serve.final)


def test_a_tight_budget_keeps_checkpoints_evenly_spread_and_the_last_one():
    model = kv8_hybrid_model()
    each = each_checkpoints_bytes(model)
    tools = lines(9, 50) + system()                 # 1000 more tokens of tools: 20 grid points before the turn ends
    cache = cache_for(model, budget=8 * 4 * each + 7)  # room for four
    serve(cache, model, tools + USER, system_tokens=len(tools))
    at = checkpoints(cache)
    last = (len(tools) - 1) // STEP * STEP
    assert len(at) == 4 and at[-1] == last
    assert max(b - a for a, b in zip([0] + at, at)) <= last // 2  # no part of it twice as far as even spacing


def test_a_line_where_the_client_rewrites_its_prompt_outlasts_the_grid_beside_it():
    model = kv8_hybrid_model()
    each = each_checkpoints_bytes(model)
    cache = cache_for(model, budget=8 * 3 * each + 7)  # room for three
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    assert checkpoints(cache) == [128, 192, 256]
    convo = USER + out + TOOL_RESULT
    serve(cache, model, system(changed(7, 40)) + convo)
    assert checkpoints(cache) == [128, 192, LEDGER_AT]  # 256, just before the line, covered least
    cached, _, _ = serve(cache, model, system(changed(2, 41)) + convo + USER)
    assert cached == LEDGER_AT


def test_checkpoints_past_a_change_go_before_the_ones_the_new_prompt_runs_along():
    model = kv8_hybrid_model()
    each = each_checkpoints_bytes(model)
    cache = cache_for(model, budget=8 * 5 * each + 7)  # room for five
    _, out, _ = serve(cache, model, system() + USER, system_tokens=len(system()))
    assert checkpoints(cache) == [64, 128, 192, 256]
    tools = list(TOOLS)
    tools[110] = 11 + (tools[110] - 10) % 53       # a tool's description edited, inside the line at 100
    second = tools + system()[len(TOOLS):] + USER + out
    cached, _, _ = serve(cache, model, second)
    assert cached == 64
    # The old prompt's checkpoints past the change went; the new one's grid and the line it changed came in.
    assert checkpoints(cache) == [64, 100, 128, 192, 256]
    assert all(tuple(second[:len(k)]) == k for k in cache._checkpoints)


def with_checkpoints(*specs):
    """A prompt cache holding stand-in checkpoints: (tokens, the fetch that last ran along them, learned)."""
    cache = MovingPromptCache(max_size=10)
    for key, touched, learned in specs:
        cache._checkpoints[key] = _Checkpoint([mx.zeros(4)], 16, touched, learned)
    return cache


def drop_down_to(cache, n, newest):
    gone = []
    while len(cache._checkpoints) > n:
        key = cache._victim(newest)
        gone.append(key)
        cache._checkpoints.pop(key)
    return gone


def test_checkpoints_no_request_runs_along_go_first_least_recent_first_each_prompts_evenly():
    ended = tuple(range(2000, 2400))                    # a conversation last fetched before the other two
    old = tuple(range(1000, 1400))                      # a system prompt as it was
    new = old[:100] + (7,) + old[101:]                  # and as it is now, changed at token 100
    cache = with_checkpoints(*[(ended[:n], 0, False) for n in (64, 128)],
                             (old[:64], 2, False),      # the new prompt runs along it too
                             *[(old[:n], 1, False) for n in (128, 192, 256)],
                             *[(new[:n], 2, False) for n in (160, 224)])
    gone = drop_down_to(cache, 3, newest=new[:224])
    assert gone == [ended[:64], ended[:128], old[:192], old[:128], old[:256]]
    assert set(cache._checkpoints) == {old[:64], new[:160], new[:224]}


def test_a_line_where_a_request_left_the_cache_outlasts_grid_checkpoints_however_close():
    prompt = tuple(range(400))
    cache = with_checkpoints(*[(prompt[:n], 1, n == 100) for n in (64, 100, 128, 192, 256)])
    gone = drop_down_to(cache, 1, newest=prompt[:256])
    assert [len(k) for k in gone] == [128, 64, 192, 256]  # the grid by gap, its last one last; then the line
    assert list(cache._checkpoints) == [prompt[:100]]


def test_a_conversation_keeps_its_turns_in_memory_once_eight_sessions_fill_the_cache(tmp_path, monkeypatch):
    # Overnight on the M4, from the eighth agent task on, every finished turn was written to disk and read back by the
    # next turn: 371 GB in under three hours. Each session's system prompt is its own (it carries the working
    # directory), and once eight of them filled the cache's entries, mlx-lm's order let the turn just finished go.
    monkeypatch.setattr(disk_cache, "MIN_TOKENS", 16)  # these prompts are short; Claude Code's is over 10k tokens
    model = kv8_hybrid_model()
    cache = cache_for(model)
    cache.max_size = 8
    cache.disk = disk_cache.DiskCache(tmp_path, "tiny", max_bytes=1 << 30)
    sessions = [system(text(100 + s, len(LEDGER) - 1) + [NEWLINE]) for s in range(9)]
    for prompt in sessions[:8]:                                                 # eight earlier sessions, a turn each
        serve(cache, model, prompt + USER, system_tokens=len(prompt))
    live = sessions[8]
    _, out, _ = serve(cache, model, live + USER, system_tokens=len(live))

    def live_turns_on_disk():
        return [t for t, _ in cache.disk.index.values() if len(t) > len(live) and list(t[:len(live)]) == live]

    assert live_turns_on_disk() == []                                           # its turn waits in memory
    cached, _, _ = serve(cache, model, live + USER + out + TOOL_RESULT)
    assert cached > len(live + USER) and live_turns_on_disk() == []             # and the next turn takes it from there
