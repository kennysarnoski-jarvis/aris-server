import mlx.core as mx

from aris_server.budget import ThinkBudget

END, PARA, FENCE, WORD, NL = 9, 1, 2, 3, 4
VOCAB = 10


def step(proc, history):
    return proc(mx.array(history), mx.zeros((VOCAB,)))


def forced(out):
    return int(mx.argmax(out).item()) == END and bool(mx.isinf(out[0]).item())


def make(budget):
    return ThinkBudget(budget, END, NL, frozenset({PARA}), {FENCE: 1})


def test_past_budget_paragraphs_are_blocked_and_thinking_closes_after_the_next_newline():
    p, hist = make(4), [0]
    for tok in [WORD, NL, WORD]:
        hist.append(tok)
        out = step(p, hist)
        assert not forced(out) and not mx.isinf(out[PARA]).item()     # under budget: untouched
    hist.append(WORD)
    out = step(p, hist)
    assert not forced(out) and mx.isinf(out[PARA]).item() and not mx.isinf(out[NL]).item()
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
