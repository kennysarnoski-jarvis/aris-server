"""MTP through mlx-lm's BatchGenerator, the path Aris Server serves through: the same tokens as plain decoding, several
per step when drafts are accepted, the head's cache aligned with the trunk's, and exact finishes mid-round."""

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache

from aris_server import mtp, mtp_serve
from aris_server.speculative import Sampling
from aris_server.kv_quant import QUANTIZED_ATTENTION_TOKENS, quantize_kv
from tiny_models import tiny_hybrid_model

PROMPTS = [[5, 17, 3, 42, 8, 9, 23, 61, 2, 14, 30, 7] * 3, [11, 4, 50, 2, 9]]


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


greedy.mtp_sampling = Sampling(0.0)


def with_mtp(depth=2, drafter=None):
    model = tiny_hybrid_model()
    mx.random.seed(1)
    head = mtp.MTPHead(mtp.text_model(model).args)
    mx.eval(head.parameters())
    mtp_serve.attach_mtp(model, head, depth=depth)
    model._mtp_config.drafter = drafter
    return model


def generate(model, prompts, max_tokens, caches=None, stop_tokens=None, processors=None, sampler=greedy):
    """Through BatchGenerator. Returns ([tokens], [final cache], number of next() calls that returned tokens)."""
    gen = BatchGenerator(model, max_tokens=max(max_tokens), sampler=sampler, stop_tokens=stop_tokens,
                         logits_processors=processors, prefill_step_size=QUANTIZED_ATTENTION_TOKENS * 2)
    uids = gen.insert(prompts, max_tokens=max_tokens, caches=caches)
    tokens = {u: [] for u in uids}
    final = {}
    steps = 0
    while len(final) < len(uids):
        responses = gen.next_generated()
        steps += bool(responses)
        for r in responses:
            tokens[r.uid].append(r.token)
            if r.finish_reason is not None:
                final[r.uid] = r.prompt_cache
    gen.close()
    return [tokens[u] for u in uids], [final[u] for u in uids], steps


def offsets(cache):
    """(trunk full-attention offset, head offset) of a finished single-sequence MTP cache."""
    trunk_kv = [c for c in cache[:-2] if not isinstance(c, ArraysCache)]
    return trunk_kv[0].offset, cache[-2].offset


def oracle(reference, right=3):
    """Drafts the true continuation (wrong after the first `right` drafts)."""
    def draft(so_far):
        n = len(so_far) - len(PROMPTS[0])
        good = reference[n:n + 3]
        return [t if i < right else (t + 1) % 64 for i, t in enumerate(good)]
    return draft


def test_a_lone_sequence_with_a_random_head_decodes_exactly_as_plain():
    want, _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [30])
    model = with_mtp(depth=2)
    got, caches, _ = generate(model, PROMPTS[:1], [30])
    assert got == want
    assert model._mtp_config.stats.rounds > 0
    assert len(caches[0]) == model._mtp_trunk_layers + 2
    trunk, head = offsets(caches[0])
    assert trunk == head == len(PROMPTS[0]) + len(got[0])


@pytest.mark.parametrize("right", [0, 1, 3])
@pytest.mark.parametrize("depth", [1, 3])
def test_accepted_drafts_come_out_several_per_step_and_unchanged(right, depth):
    (want,), _, plain_steps = generate(tiny_hybrid_model(), PROMPTS[:1], [40])
    model = with_mtp(depth=depth, drafter=oracle(want, right))
    (got,), (cache,), steps = generate(model, PROMPTS[:1], [40])
    assert got == want
    stats = model._mtp_config.stats
    if right >= depth:
        assert stats.acceptance == 1.0
        assert steps < plain_steps  # several tokens per call
    if right == 0:
        assert stats.accepted == 0
    trunk, head = offsets(cache)
    assert trunk == head == len(PROMPTS[0]) + len(got)


def test_two_sequences_decode_plainly_and_the_survivor_then_speculates():
    want, _, _ = generate(tiny_hybrid_model(), PROMPTS, [30, 6])
    model = with_mtp(depth=2)
    got, caches, _ = generate(model, PROMPTS, [30, 6])
    assert got == want
    assert model._mtp_config.stats.rounds > 0  # the long one went on alone
    for prompt, tokens, cache in zip(PROMPTS, got, caches):
        trunk, head = offsets(cache)
        assert trunk == head == len(prompt) + len(tokens)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_heads_cache_holds_the_true_rows_when_its_drafts_are_accepted(depth):
    """Run the real head (its rows and trims) with correct drafts swapped in, so most rounds keep several tokens.
    Its cache must then hold exactly the rows a fresh pass computes: row p = (trunk hidden p-1, token p)."""
    (want,), _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [40])
    model = with_mtp(depth=depth)
    model._mtp_config.draft_override = oracle(want, 3)
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [40])
    assert got == want
    stats = model._mtp_config.stats
    assert stats.acceptance > 0.9
    trunk, head = offsets(cache)
    assert trunk == head == len(PROMPTS[0]) + len(got)

    seq = PROMPTS[0] + got
    hidden = mtp.trunk_hidden(model, mx.array([seq]), model.make_cache()[:-2])
    rows = mx.concatenate([mx.zeros_like(hidden[:, :1]), hidden[:, :-1]], axis=1)
    fresh = KVCache()
    model._mtp_head(mtp.embed(model, mx.array([seq])), rows, fresh)
    served = cache[-2]
    assert mx.allclose(served.keys[..., :served.offset, :], fresh.keys[..., :fresh.offset, :], atol=1e-4)
    assert mx.allclose(served.values[..., :served.offset, :], fresh.values[..., :fresh.offset, :], atol=1e-4)
    assert mx.allclose(cache[-1][0], hidden[:, -1:], atol=1e-4)  # the pending hidden is the last position's


def mid_round_stop(want, depth):
    """A token whose first occurrence falls inside a fully accepted round, not at its end."""
    per_round = depth + 1
    for i in range(per_round, len(want)):
        if want[i] not in want[:i] and i % per_round != depth:
            return i
    raise AssertionError("no mid-round stop candidate")


@pytest.mark.parametrize("depth", [2, 3])
def test_a_stop_token_inside_an_accepted_round_ends_it_there(depth):
    (want,), _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [40])
    stop = want[mid_round_stop(want, depth)]
    end = want.index(stop) + 1
    (plain,), _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [40], stop_tokens=[[stop]])
    model = with_mtp(depth=depth)
    model._mtp_config.draft_override = oracle(want, 3)
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [40], stop_tokens=[[stop]])
    assert got == plain == want[:end]
    trunk, head = offsets(cache)
    assert trunk == head == len(PROMPTS[0]) + len(got)  # the cache holds exactly what was returned


@pytest.mark.parametrize("max_tokens", [1, 2, 5, 11])
def test_max_tokens_inside_a_round_is_exact(max_tokens):
    (want,), _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [max_tokens])
    model = with_mtp(depth=3, drafter=oracle(generate(tiny_hybrid_model(), PROMPTS[:1], [40])[0][0], 3))
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [max_tokens])
    assert got == want and len(got) == max_tokens
    trunk, head = offsets(cache)
    assert trunk == head == len(PROMPTS[0]) + len(got)


def test_a_second_turn_continues_from_the_first_turns_cache():
    plain_model = tiny_hybrid_model()
    (t1,), (c1,), _ = generate(plain_model, PROMPTS[:1], [12])
    (t2,), _, _ = generate(plain_model, [[9, 8, 7]], [20], caches=[c1])

    model = with_mtp(depth=2)
    (g1,), (m1,), _ = generate(model, PROMPTS[:1], [12])
    (g2,), (m2,), _ = generate(model, [[9, 8, 7]], [20], caches=[m1])
    assert (g1, g2) == (t1, t2)
    trunk, head = offsets(m2)
    assert trunk == head == len(PROMPTS[0]) + len(g1) + 3 + len(g2)


def test_an_8bit_cache_holds_the_head_too():
    plain_model = tiny_hybrid_model()
    quantize_kv(plain_model, 8)
    want, _, _ = generate(plain_model, PROMPTS[:1], [25])
    model = with_mtp(depth=2)
    quantize_kv(model, 8)
    got, (cache,), _ = generate(model, PROMPTS[:1], [25])
    assert got == want
    assert type(cache[-2]) is QuantizedKVCache and isinstance(cache[-1], ArraysCache)


def test_processors_see_the_same_calls_as_plain_decoding():
    def recorder(log):
        def p(tokens, logits):
            log.append(tokens.tolist())
            return logits
        return p

    plain_log, mtp_log = [], []
    (want,), _, _ = generate(tiny_hybrid_model(), PROMPTS[:1], [20], processors=[recorder(plain_log)])
    model = with_mtp(depth=3, drafter=oracle(want, 2))
    (got,), _, _ = generate(model, PROMPTS[:1], [20], processors=[recorder(mtp_log)])
    assert got == want
    assert mtp_log[:len(plain_log)] == plain_log[:len(mtp_log)]
    assert len(mtp_log) >= len(want) - 1


def test_sampled_requests_decode_and_count_acceptance():
    s = Sampling(temperature=0.6, top_p=0.95, top_k=20)

    def sampler(logprobs):
        return mx.random.categorical(logprobs / 0.6)

    sampler.mtp_sampling = s
    model = with_mtp(depth=2)
    mx.random.seed(4)
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [30], sampler=sampler)
    assert len(got) == 30
    assert model._mtp_config.stats.rounds > 0
    trunk, head = offsets(cache)
    assert trunk == head == len(PROMPTS[0]) + len(got)


def test_a_cache_without_head_slots_decodes_plainly():
    """A cache made before MTP was on (no head slots) still decodes: without speculation."""
    plain_model = tiny_hybrid_model()
    (t1,), (c1,), _ = generate(plain_model, PROMPTS[:1], [8])
    (want,), _, _ = generate(plain_model, [[3, 3]], [10], caches=[c1])
    model = with_mtp(depth=2)
    (got,), _, _ = generate(model, [[3, 3]], [10], caches=[c1])
    assert got == want
    assert model._mtp_config.stats.rounds == 0
