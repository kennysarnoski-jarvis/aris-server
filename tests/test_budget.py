import mlx.core as mx

from aris_server.budget import ThinkBudget

END, PARA, FENCE, WORD, NL = 9, 1, 2, 3, 4
VOCAB = 10


def step(proc, history):
    return proc(mx.array(history), mx.zeros((VOCAB,)))


def forced(out):
    return int(mx.argmax(out).item()) == END and bool(mx.isinf(out[0]).item())


def make(budget, overrun=1024):
    return ThinkBudget(budget, END, NL, frozenset({PARA}), {FENCE: 1}, overrun=overrun)


def test_past_budget_paragraphs_are_blocked_and_thinking_closes_after_the_next_newline():
    p, hist = make(4), [0]
    for tok in [WORD, NL, WORD]:
        hist.append(tok)
        out = step(p, hist)
        assert not forced(out) and not mx.isinf(out[PARA]).item()     # under budget: untouched
    hist.append(WORD)
    logits = mx.zeros((VOCAB,))
    out = p(mx.array(hist), logits)
    assert not forced(out) and mx.isinf(out[PARA]).item()
    assert abs(out[NL].item() - float(mx.log(mx.array(2.0)).item())) < 1e-5   # the break's odds move to the newline
    hist.append(NL)
    assert forced(step(p, hist))       # text\n</think>: what the chat template re-renders
    assert p.forced and p.done


def test_waits_out_an_open_code_fence():
    p, hist = make(2), [0]
    for tok in [FENCE, WORD, WORD, NL, PARA, WORD]:
        hist.append(tok)
        out = step(p, hist)
        assert not forced(out) and not mx.isinf(out[PARA]).item(), tok
    hist += [FENCE, NL]
    step(p, hist[:-1])
    assert forced(step(p, hist))


def test_inert_after_the_model_ends_thinking_itself():
    p, hist = make(2), [0, WORD, END]
    step(p, hist[:2])
    step(p, hist)
    for tok in [WORD, WORD, NL, PARA]:
        hist.append(tok)
        out = step(p, hist)
        assert not forced(out) and not mx.isinf(out[PARA]).item()
    assert p.done and not p.forced


def test_a_runaway_line_or_open_fence_is_closed_after_the_overrun():
    for opener in (WORD, FENCE):                       # a line that never ends; a code fence never closed
        p, hist = make(2, overrun=3), [0]
        for _ in range(4):
            hist.append(opener if len(hist) == 1 else WORD)
            assert not forced(step(p, hist))
        hist.append(WORD)
        out = step(p, hist)
        assert int(mx.argmax(out).item()) == NL and mx.isinf(out[WORD]).item()   # newline forced first
        hist.append(NL)
        assert forced(step(p, hist))
