"""The depth controller: each round drafts as many tokens as pay, learned from what rounds accept and cost."""

import random

import pytest

from aris_server.speculative import DepthControl


def throughput(accept, cost, d):
    """True expected tokens a second at depth d."""
    expected = run = 1.0
    for p in accept[:d]:
        run *= p
        expected += run
    return expected / cost(d)


def best_depth(accept, cost, max_depth):
    return max(range(max_depth + 1), key=lambda d: throughput(accept, cost, d))


def near_best(d, accept, cost, max_depth, share=0.97):
    """Depth d gives at least `share` of the best throughput (near the top the curve is flat: noise picks within it)."""
    return throughput(accept, cost, d) >= share * throughput(accept, cost, best_depth(accept, cost, max_depth))


def run(control, accept, cost, rounds, seed=0, room=100):
    """Simulated rounds: each draft is accepted with its position's probability while all before it were."""
    rng = random.Random(seed)
    picks = []
    for _ in range(rounds):
        d = control.choose(room)
        kept = 0
        while kept < d and rng.random() < accept[kept]:
            kept += 1
        control.observe(d, kept, cost(d) * (1 + 0.05 * rng.uniform(-1, 1)))
        picks.append(d)
    return picks


def settled(picks, tail=200):
    """The depth picked most often over the last rounds."""
    last = picks[-tail:]
    return max(set(last), key=last.count)


MOE = lambda d: 1.0 + 0.35 * d    # noqa: E731  Inkling 2 Mini: each verified token reads more experts
DENSE = lambda d: 1.0 + 0.03 * d  # noqa: E731  a dense 27B: a wider verify costs little more
FALLING = [0.9, 0.78, 0.7, 0.6, 0.5, 0.4, 0.3]


@pytest.mark.parametrize("cost", [MOE, DENSE])
def test_it_settles_on_the_depth_that_pays_most(cost):
    picks = run(DepthControl(7, start=3), FALLING, cost, 600)
    assert near_best(settled(picks), FALLING, cost, 7)


def test_hard_text_decodes_plainly():
    picks = run(DepthControl(7, start=3), [0.2] * 7, MOE, 400)
    assert settled(picks) == 0
    assert picks[-200:].count(0) > 150  # only the exploring rounds draft


def test_easy_text_on_a_dense_model_drafts_deep():
    picks = run(DepthControl(7, start=2), [0.97] * 7, DENSE, 400)
    assert settled(picks) >= 5 and near_best(settled(picks), [0.97] * 7, DENSE, 7)


def test_it_follows_the_text_as_it_changes():
    control = DepthControl(7, start=3)
    picks = run(control, [0.95] * 7, MOE, 300, seed=1)
    assert settled(picks) >= 4 and near_best(settled(picks), [0.95] * 7, MOE, 7)
    picks = run(control, [0.15] * 7, MOE, 300, seed=2)  # the text turns hard
    assert settled(picks) == 0


def test_room_and_the_maximum_are_respected():
    control = DepthControl(4, start=9)
    assert control.choose(100) == 4  # the start is capped at the maximum
    control.observe(4, 4, 1.0)
    for room in range(0, 6):
        assert 0 <= control.choose(room) <= min(room, 4)
    assert DepthControl(3).choose(0) == 0


def test_the_first_rounds_measure_the_start_and_a_plain_step():
    control = DepthControl(5, start=3)
    assert control.choose(100) == 3
    control.observe(3, 1, 1.5)
    assert control.choose(100) == 0
    control.observe(0, 0, 1.0)
    assert set(control.seconds) == {0, 3}
    assert control.cost(1) == pytest.approx(1.0 + 0.5 / 3)  # between the two measured depths
    assert control.cost(5) == pytest.approx(1.0 + 5 * 0.5 / 3)  # along the same line beyond them
