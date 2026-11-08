import mlx.core as mx

from aris_server.budget import ThinkBudget

END, PARA, FENCE, WORD = 9, 1, 2, 3
VOCAB = 10


def step(proc, history):
    out = proc(mx.array(history), mx.zeros((VOCAB,)))
    return out


def forced(out):
    return int(mx.argmax(out).item()) == END and bool(mx.isinf(out[0]).item())


def make(budget):
    return ThinkBudget(budget, END, frozenset({PARA}), {FENCE: 1})


def test_closes_thinking_at_first_paragraph_break_past_budget():
    p, hist = make(3), [0]
    for tok in [WORD, WORD, WORD, WORD]:
        hist.append(tok)
        assert not forced(step(p, hist))
    hist.append(PARA)
    assert forced(step(p, hist))
    assert p.forced and p.done


def test_waits_out_an_open_code_fence():
    p, hist = make(2), [0]
    for tok in [FENCE, WORD, WORD, PARA, WORD]:
        hist.append(tok)
        assert not forced(step(p, hist)), tok
    hist += [FENCE, PARA]
    step(p, hist[:-1])
    assert forced(step(p, hist))


def test_inert_after_the_model_ends_thinking_itself():
    p, hist = make(2), [0, WORD, END]
    step(p, hist[:2])
    step(p, hist)
    for tok in [WORD, WORD, PARA, PARA]:
        hist.append(tok)
        assert not forced(step(p, hist))
    assert p.done and not p.forced
