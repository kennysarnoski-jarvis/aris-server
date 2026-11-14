import random
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.generate import TextStateMachine

from aris_server.config import ModelSpec
from aris_server.engine import _classify
from aris_server.guards import LoopGuard, loop_exempt_ids, stuck_call_repeats
from aris_server.translate import plan

ARETE = ModelSpec(name="arete:27b", path="x", context=131072, effort_kwarg="reasoning_effort",
                  effort_map={"high": "medium", "xhigh": "xhigh"})
NOTE = "You've made this exact tool call"


# --- stuck tool calls -------------------------------------------------------------------------------------------

def step(command, result, i, *, error=False, extra_calls=()):
    """One turn of the tool loop: the model runs `command`, Claude Code answers with `result`."""
    calls = [{"type": "tool_use", "id": f"toolu_{i}", "name": "Bash", "input": {"command": command}}]
    calls += [{"type": "tool_use", "id": f"toolu_{i}_{j}", "name": name, "input": inp}
              for j, (name, inp) in enumerate(extra_calls)]
    results = [{"type": "tool_result", "tool_use_id": c["id"], "content": result, **({"is_error": True} if error else {})}
               for c in calls]
    return [{"role": "assistant", "content": [{"type": "thinking", "thinking": f"look again {i}", "signature": "s"},
                                              *calls]},
            {"role": "user", "content": results}]


def conversation(*steps):
    msgs = [{"role": "user", "content": "implement the library"}]
    for s in steps:
        msgs += s
    return msgs


def body(messages):
    return {"model": "arete:27b", "max_tokens": 32000, "messages": messages,
            "thinking": {"type": "adaptive", "display": "omitted"}}


LS = "ls -la && cat Cargo.toml"


def test_the_third_identical_call_with_the_same_result_gets_a_reminder_before_the_model_answers():
    msgs = conversation(step("cat src/lib.rs", "fn main() {}", 1), *[step(LS, "lib.rs", i) for i in range(2, 5)])
    assert stuck_call_repeats(msgs) == 3
    p = plan(body(msgs), ARETE)
    assert p.stuck_repeats == 3
    assert p.messages[-1]["role"] == "user" and NOTE in p.messages[-1]["content"] and "3 times" in p.messages[-1]["content"]
    assert p.messages[-1]["content"].startswith("<system-reminder>")
    assert p.messages[-2]["role"] == "tool"      # it comes after the last result, so the cache covers what's before
    assert p.think_budget == plan(body(msgs[:-2] + step("ls", "x", 9)), ARETE).think_budget   # still a follow-up turn


def test_two_repeats_are_left_alone():
    msgs = conversation(*[step(LS, "lib.rs", i) for i in range(2)])
    assert stuck_call_repeats(msgs) == 2
    p = plan(body(msgs), ARETE)
    assert p.stuck_repeats == 0 and all(NOTE not in str(m["content"]) for m in p.messages)


@pytest.mark.parametrize("change", ["input", "result", "error", "typed", "second call"])
def test_anything_different_in_between_breaks_the_run(change):
    middle = {
        "input": step("ls -la", "lib.rs", 2),
        "result": step(LS, "lib.rs\nnew.rs", 2),
        "error": step(LS, "lib.rs", 2, error=True),
        "typed": step(LS, "lib.rs", 2)[:1] + [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "lib.rs"},
            {"type": "text", "text": "please stop listing"}]}],
        "second call": step(LS, "lib.rs", 2, extra_calls=[("Read", {"file_path": "a"})]),
    }[change]
    msgs = conversation(step(LS, "lib.rs", 1), middle, step(LS, "lib.rs", 3), step(LS, "lib.rs", 4))
    assert stuck_call_repeats(msgs) == 2
    assert plan(body(msgs), ARETE).stuck_repeats == 0


def test_reminders_claude_code_adds_and_list_form_results_still_count_as_the_same_step():
    def listed(i):
        a, u = step(LS, [{"type": "text", "text": "lib.rs"}], i)
        u["content"].append({"type": "text", "text": f"<system-reminder>todo list, turn {i}</system-reminder>"})
        return [a, u]

    assert stuck_call_repeats(conversation(*[listed(i) for i in range(4)])) == 4


def test_parallel_calls_count_when_every_call_and_result_repeats():
    same = [("Read", {"file_path": "a"})]
    assert stuck_call_repeats(conversation(*[step(LS, "x", i, extra_calls=same) for i in range(3)])) == 3


def test_only_a_request_that_answers_tool_calls_is_checked():
    msgs = conversation(*[step(LS, "lib.rs", i) for i in range(3)]) + [
        {"role": "assistant", "content": "Done."}, {"role": "user", "content": "now add tests"}]
    assert stuck_call_repeats(msgs) == 0
    assert stuck_call_repeats([{"role": "user", "content": "hi"}]) == 0
    assert stuck_call_repeats([]) == 0


def test_guards_off_means_no_reminder():
    msgs = conversation(*[step(LS, "lib.rs", i) for i in range(5)])
    p = plan(body(msgs), ARETE, guards=False)
    assert p.stuck_repeats == 0 and all(NOTE not in str(m["content"]) for m in p.messages)


# --- looping output ---------------------------------------------------------------------------------------------

VOCAB = 64


def feed(guard, tokens):
    """Run the guard over `tokens` as a response would generate them; each call sees the history so far."""
    outs = []
    for i in range(1, len(tokens) + 1):
        outs.append(guard(mx.array(tokens[:i]), mx.zeros((1, VOCAB))))
    return outs


def banned(out):
    return [i for i, inf in enumerate(mx.isinf(out[0]).tolist()) if inf]


def test_a_run_of_one_token_is_broken_once_it_spans_the_window():
    g = LoopGuard(span=40, cycles=8, read_every=1)
    outs = feed(g, [7] * 40)
    assert all(not banned(o) for o in outs[:-1])
    assert banned(outs[-1]) == [7] and g.breaks == 1 and not g.gave_up


def test_a_repeated_chunk_is_broken_at_the_token_that_would_continue_it():
    rng = random.Random(1)
    chunk = [rng.randrange(1, VOCAB) for _ in range(5)]
    tokens = (chunk * 20)[:43]                  # the window (40) fills mid-chunk
    g = LoopGuard(span=40, cycles=8, read_every=1)
    outs = feed(g, tokens)
    hit = next(i for i, o in enumerate(outs) if banned(o))
    assert banned(outs[hit]) == [tokens[hit + 1 - 5]]   # the token one chunk back from the next position
    assert hit == 39                                    # once the last 40 tokens are one chunk repeated


def test_ordinary_text_is_never_touched():
    rng = random.Random(2)
    g = LoopGuard(span=40, cycles=8, read_every=1)
    assert not any(banned(o) for o in feed(g, [rng.randrange(VOCAB) for _ in range(3000)]))
    assert g.breaks == 0 and not g.gave_up


def test_a_chunk_longer_than_span_over_cycles_is_not_a_loop():
    rng = random.Random(3)
    chunk = [rng.randrange(1, VOCAB) for _ in range(6)]     # max chunk is 40 // 8 = 5
    g = LoopGuard(span=40, cycles=8, read_every=1)
    assert not any(banned(o) for o in feed(g, chunk * 30))


def test_after_the_allowed_breaks_the_next_loop_gives_up_on_the_response():
    g = LoopGuard(span=40, cycles=8, breaks=3, read_every=1)
    outs, hist = [], []
    for _ in range(4):
        for t in [9] * 40 + [3]:           # each time: a loop, broken; one other token; then back into the loop
            hist.append(t)
            outs.append(g(mx.array(hist), mx.zeros((1, VOCAB))))
            if g.gave_up:
                break
    assert [banned(o) for o in outs if banned(o)] == [[9], [9], [9]]
    assert g.breaks == 3 and g.gave_up
    assert len(hist) == 4 * 41 - 1         # gave up the moment the fourth loop filled the window


def test_tokens_are_read_in_batches_so_a_loop_is_caught_at_most_a_batch_late():
    rng = random.Random(4)
    chunk = [rng.randrange(1, VOCAB) for _ in range(5)]
    tokens = [rng.randrange(VOCAB) for _ in range(13)] + chunk * 20     # the loop's window fills at index 52
    g = LoopGuard(span=40, cycles=8, read_every=8)
    outs = feed(g, tokens)
    hits = [i for i, o in enumerate(outs) if banned(o)]
    # The first read after the window filled (reads at calls 8, 16, ... 56), then again once the scripted tokens,
    # which ignore the block, have filled a fresh window.
    assert hits == [55, 95] and g.breaks == 2
    assert banned(outs[55]) == [tokens[56 - 5]]   # still the token that would continue the chunk next
    assert len(g.pending) == len(tokens) % 8


def test_a_token_that_ends_something_is_never_blocked():
    END = 5
    g = LoopGuard(exempt=frozenset({END}), span=40, cycles=8, read_every=1)
    out = feed(g, [END, 1] * 20)[-1]       # the next token would be END: blocking it could trap the model
    assert not banned(out) and g.breaks == 1


def test_loop_exempt_ids_are_the_ending_tokens():
    class Tok:
        eos_token_ids = {50}
        think_end = "</think>"
        tool_call_end = "</tool_call>"
        ids = {"</think>": [41], "</tool_call>": [42], "<|im_end|>": [50], "<|endoftext|>": [43, 44]}

        def encode(self, text, add_special_tokens=False):
            return self.ids.get(text, [1, 2])

    assert loop_exempt_ids(Tok()) == frozenset({41, 42, 50})


# --- the engine ends a response the guard gave up on ------------------------------------------------------------

def test_a_response_the_guard_gives_up_on_ends_as_if_it_hit_the_token_cap():
    sm = TextStateMachine({"normal": [("<tool_call>", "tool")], "tool": [("</tool_call>", "normal")]})
    stopped = []
    ctx = SimpleNamespace(text_sm=sm, initial_state="normal", stop=lambda: stopped.append(True))
    state = {"gave_up": False}

    def responses():
        for i in range(100):
            if i == 3:
                state["gave_up"] = True      # set by the guard on the generation thread
            yield SimpleNamespace(text="0", finish_reason=None)

    out = list(_classify(ctx, responses(), lambda: state["gave_up"]))
    assert [finish for _, finish, _ in out] == [None, None, None, "length"]
    assert stopped == [True]
    assert "".join(p.text for pieces, _, _ in out for p in pieces) == "0000"


def test_without_the_guard_classify_is_unchanged():
    sm = TextStateMachine({"normal": [("<tool_call>", "tool")], "tool": [("</tool_call>", "normal")]})
    ctx = SimpleNamespace(text_sm=sm, initial_state="normal", stop=lambda: None)
    resp = [SimpleNamespace(text=t, finish_reason=f) for t, f in [("a", None), ("b", None), ("", "stop")]]
    out = list(_classify(ctx, iter(resp)))
    assert [f for _, f, _ in out] == [None, None, "stop"]
