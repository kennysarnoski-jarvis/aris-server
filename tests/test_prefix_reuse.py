"""A request that leaves every cached conversation early (a client that rewrites part of its system prompt on each
request, say) reuses what it still shares with them. A hybrid model's linear-attention layers carry a running state
that can't be cut back to an earlier token, so that reuse comes from recurrent-state checkpoints taken while prompts
are read, beside the attention cache cut back to the same token.

These tests drive a tiny model with Arete's architecture through mlx-lm's own batch generator, the way mlx-lm's server
drives it (ResponseGenerator._generate, batched path), so the checkpoints are taken where a real prompt read takes
them."""

import logging

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import BatchGenerator
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import ArraysCache

from aris_server.engine import MovingPromptCache, _split_segments, merge_prompt_tail
from aris_server.kv_quant import quantize_kv
from tiny_models import tiny_hybrid_model

END = 9       # the tiny vocabulary's end of a turn (<|im_end|>)
NEWLINE = 10  # and its line break, for finding where a changed line starts
STEP = 64     # checkpoint spacing, scaled down from Arete's 4096 with everything else


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
    if any(isinstance(c, ArraysCache) for c in model.make_cache()):  # as the engine decides
        cache.checkpoint_every, cache.turn_ends = STEP, frozenset({END})
    return cache


def serve(cache, model, prompt, system_tokens=None, max_tokens=4):
    """One request, the way mlx-lm's server serves it: its prompt split into segments (the system prompt, then the
    rest; a request ending in tool results is one segment), the nearest cache fetched, the prompt read with a copy of
    the cache saved at each segment's end, then the finished sequence's cache saved. Returns (cached tokens, generated
    tokens, their logprobs)."""
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
    gen = BatchGenerator(model, max_tokens=max_tokens, prefill_step_size=32)
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
    assert checkpoints(cache) == [192, 256]                        # the latest ones
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
