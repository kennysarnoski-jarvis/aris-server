"""Inkling 2's draft head through mlx-lm's BatchGenerator, the path Aris Server serves through: the same tokens as plain
decoding, several per step when drafts are accepted, the drafter's contexts in step with the target, batched reads
included, and exact finishes mid-round."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache, make_prompt_cache

from aris_server import dspark, mtp_serve
from aris_server.kv_quant import quantize_kv
from test_dspark import MTP_CONFIG, random_drafter
from test_inkling2_model import LOOP_CONFIG, create_model
from test_mtp_serve import generate

PROMPTS = [[5, 17, 3, 42, 8, 9, 23, 61, 2, 14, 30, 7] * 3, [11, 4, 50, 2, 9]]


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def plain_model():
    return create_model(config=LOOP_CONFIG)


def with_dspark(depth=3, drafter=None):
    model = plain_model()
    mtp_serve.attach_dspark(model, random_drafter(model), depth=depth)
    model._mtp_config.drafter = drafter
    return model


def oracle(reference, prompt, right=3):
    """Drafts the true continuation (wrong after the first `right` drafts)."""
    def draft(so_far):
        n = len(so_far) - len(prompt)
        good = reference[n:n + 7]
        return [t if i < right else (t + 1) % 120 for i, t in enumerate(good)]
    return draft


def offsets(model, cache):
    """(tokens the target read, tokens each drafter context read) for a finished cache."""
    n = model._mtp_trunk_layers
    return cache[0][0].offset, [c.offset for c in cache[n:-1]]


def test_a_lone_sequence_with_a_random_head_decodes_exactly_as_plain():
    want, _, _ = generate(plain_model(), PROMPTS[:1], [30])
    model = with_dspark(depth=3)
    got, (cache,), _ = generate(model, PROMPTS[:1], [30])
    assert got == want
    assert model._mtp_config.stats.rounds > 0
    assert len(cache) == model._mtp_trunk_layers + len(MTP_CONFIG["local_layer_ids"]) + 2
    target, drafter = offsets(model, cache)
    assert target == len(PROMPTS[0]) + len(got[0]) and set(drafter) == {target}  # exactly what was returned


@pytest.mark.parametrize("right", [0, 1, 3])
@pytest.mark.parametrize("depth", [1, 3, 7])
def test_accepted_drafts_come_out_several_per_step_and_unchanged(right, depth):
    (want,), _, plain_steps = generate(plain_model(), PROMPTS[:1], [40])
    model = with_dspark(depth=depth, drafter=oracle(want, PROMPTS[0], right))
    (got,), (cache,), steps = generate(model, PROMPTS[:1], [40])
    assert got == want
    stats = model._mtp_config.stats
    if right >= depth:
        assert stats.acceptance == 1.0
        assert steps < plain_steps  # several tokens per call
    if right == 0:
        assert stats.accepted == 0
    target, drafter = offsets(model, cache)
    assert set(drafter) == {target}


def test_two_sequences_decode_plainly_and_the_survivor_then_speculates():
    want, _, _ = generate(plain_model(), PROMPTS, [30, 6])
    model = with_dspark(depth=3)
    got, caches, _ = generate(model, PROMPTS, [30, 6])
    assert got == want
    assert model._mtp_config.stats.rounds > 0  # the long one went on alone
    for cache in caches:
        target, drafter = offsets(model, cache)
        assert set(drafter) == {target}


@pytest.mark.parametrize("depth", [2, 5])
def test_the_drafters_contexts_hold_the_true_rows(depth):
    """The real drafter runs (contexts, observes) with correct drafts swapped in, so most rounds keep several tokens.
    Its contexts must then hold exactly what a fresh read of the whole sequence gives them."""
    (want,), _, _ = generate(plain_model(), PROMPTS[:1], [40])
    model = with_dspark(depth=depth)
    model._mtp_config.draft_override = oracle(want, PROMPTS[0], 7)
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [40])
    assert got == want and model._mtp_config.stats.acceptance > 0.5
    drafter = model._mtp_head
    read = PROMPTS[0] + got
    fresh = drafter.make_cache()
    _, taps = dspark.read(model, mx.array([read]), make_prompt_cache(plain_model()), drafter.config.target_layer_ids)
    drafter.observe(taps, fresh)
    n = model._mtp_trunk_layers
    for a, b in zip(cache[n:-1], fresh[:-1]):
        for x, y in zip(dspark.context_rows(a), dspark.context_rows(b)):
            assert x.shape == y.shape and mx.allclose(x, y, atol=1e-4)
    assert mx.allclose(cache[-1][0], fresh[-1][0], atol=1e-4)


def test_a_stop_token_inside_an_accepted_round_ends_it_there():
    (want,), _, _ = generate(plain_model(), PROMPTS[:1], [40])
    stop = want[20]  # rounds of 8 accepted tokens cover 1-8, 9-16, 17-24: 20 is mid-round
    end = want.index(stop) + 1
    (plain,), _, _ = generate(plain_model(), PROMPTS[:1], [40], stop_tokens=[[stop]])
    model = with_dspark(depth=7)
    model._mtp_config.draft_override = oracle(want, PROMPTS[0], 7)
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [40], stop_tokens=[[stop]])
    assert got == plain == want[:end]
    target, drafter = offsets(model, cache)
    assert target == len(PROMPTS[0]) + len(got) and set(drafter) == {target}


@pytest.mark.parametrize("max_tokens", [1, 2, 5, 11])
def test_max_tokens_inside_a_round_is_exact(max_tokens):
    (want,), _, _ = generate(plain_model(), PROMPTS[:1], [max_tokens])
    model = with_dspark(depth=7, drafter=oracle(want + [0] * 8, PROMPTS[0], 7))
    (got,), (cache,), _ = generate(model, PROMPTS[:1], [max_tokens])
    assert got == want and len(got) == max_tokens
    target, drafter = offsets(model, cache)
    assert target == len(PROMPTS[0]) + len(got) and set(drafter) == {target}


def test_a_second_turn_continues_from_the_first_turns_cache():
    (t1,), (c1,), _ = generate(plain_model(), PROMPTS[:1], [8])
    (want,), _, _ = generate(plain_model(), [[3, 3]], [16], caches=[c1])
    model = with_dspark(depth=3)
    (t2,), (c2,), _ = generate(model, PROMPTS[:1], [8])
    assert t2 == t1
    (got,), (c3,), _ = generate(model, [[3, 3]], [16], caches=[c2])
    assert got == want
    target, drafter = offsets(model, c3)
    assert target == len(PROMPTS[0]) + len(t2) + 2 + len(got) and set(drafter) == {target}


def test_kv_bits_leave_the_drafters_contexts_alone():
    model = with_dspark(depth=3)
    before = [type(c) for c in model.make_cache()]
    assert quantize_kv(model, 8) is False  # Inkling's own caches are CacheLists, and the drafter's are its own
    assert [type(c) for c in model.make_cache()] == before
    assert {KVCache, RotatingKVCache, ArraysCache} <= set(before)


def test_the_drafter_is_attached_when_an_inkling_model_loads(tmp_path, monkeypatch):
    model = plain_model()
    drafter = random_drafter(model)
    (tmp_path / "config.json").write_text(json.dumps({**LOOP_CONFIG, "mtp_config": MTP_CONFIG}))
    head = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(head), dict(tree_flatten(drafter.parameters())))
    provider = SimpleNamespace(model=None, _mtp_spec=SimpleNamespace(mtp=str(head), mtp_bits=0, mtp_depth=4))
    monkeypatch.setattr(mtp_serve, "_original_load", lambda self, *a, **k: setattr(self, "model", model))
    n = len(model.make_cache())
    mtp_serve._load_with_mtp(provider, str(tmp_path))
    assert isinstance(provider.model._mtp_head, dspark.Drafter)
    assert provider.model._mtp_config.depth == 4
    assert len(provider.model.make_cache()) == n + len(MTP_CONFIG["local_layer_ids"]) + 2
    mtp_serve._load_with_mtp(provider, str(tmp_path))  # loading again doesn't stack a second head
    assert len(provider.model.make_cache()) == n + len(MTP_CONFIG["local_layer_ids"]) + 2
    assert provider.model._mtp_head.nbytes > sum(v.nbytes for _, v in tree_flatten(drafter.parameters()))


def test_the_drafter_reads_a_batch_survivors_contexts_as_its_own():
    """In the generator a lone sequence's caches are batch caches. A survivor shorter than the drafter's window keeps
    its left padding there (mlx-lm's BatchRotatingKVCache.filter keeps it), and a round's commit then appends several
    rows: the drafter must still read exactly the rows a cache of that sequence alone holds, and draft the same."""
    model = plain_model()
    drafter = random_drafter(model)
    ids = drafter.config.target_layer_ids
    long, short, more = PROMPTS[0], PROMPTS[0][:3], [21, 40, 6]  # 3 tokens: under the 4-row window

    def observed(tokens):
        c = drafter.make_cache()
        _, taps = dspark.read(model, mx.array([tokens]), make_prompt_cache(model), ids)
        drafter.observe(taps, c)
        return c

    batch = [type(a).merge([a, b]) for a, b in zip(observed(long), observed(short))]
    for c in batch:
        c.filter([1])  # the short one survives, padded
    assert int(batch[0].left_padding[0].item()) > 0
    alone = observed(short)
    _, taps = dspark.read(model, mx.array([short + more]), make_prompt_cache(model), ids)
    for c in (batch, alone):  # a round's commit: several new rows at once
        drafter.observe([t[:, len(short):] for t in taps], c)
    view = dspark.one_sequence(batch)
    for a, b in zip(view[:-1], alone[:-1]):
        assert a.offset == b.offset == len(short + more)
        for x, y in zip(dspark.context_rows(a), dspark.context_rows(b)):
            assert x.shape == y.shape and mx.allclose(x, y, atol=1e-5)
    assert drafter.propose(model, 7, 7, view).tolist() == drafter.propose(model, 7, 7, alone).tolist()
