"""MTP speculative decoding on a tiny random Qwen 3.5 model (CPU): restoring the hybrid caches after a verify pass,
greedy output identical to plain decoding whatever the drafts are, and exact speculative sampling."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from aris_server import mtp
from tiny_models import tiny_hybrid_model

PROMPT = [5, 17, 3, 42, 8, 9, 23, 61, 2, 14, 30, 7]


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def plain_greedy(model, prompt, n):
    """Reference: read the prompt, then one token at a time, argmax each step."""
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    out = []
    for _ in range(n):
        token = int(mx.argmax(logits).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
    return out


def random_head(model):
    args = mtp.text_model(model).args
    mx.random.seed(1)
    head = mtp.MTPHead(args)
    mx.eval(head.parameters())
    return head


def prefilled(model, prompt):
    cache = make_prompt_cache(model)
    mx.eval(mtp.trunk_hidden(model, mx.array([prompt]), cache), *[c.state for c in cache])
    return cache


def cache_arrays(cache):
    out = []
    for c in cache:
        if hasattr(c, "offset"):  # KVCache: compare only the filled part
            out += [c.keys[..., :c.offset, :], c.values[..., :c.offset, :]]
        else:
            out += [c[0], c[1]]
    return out


def test_head_loader_adds_one_to_zero_centered_norms(tmp_path):
    model = tiny_hybrid_model()
    head = random_head(model)
    params = dict(tree_flatten(head.parameters()))
    raw = {}
    for name, value in params.items():
        if value.ndim == 1 and name.endswith(mtp._ZERO_CENTERED):
            value = value - 1.0
        raw["mtp." + name] = value
    mx.save_safetensors(str(tmp_path / "raw.safetensors"), raw)
    mx.save_safetensors(str(tmp_path / "mlx.safetensors"), {"mtp." + k: v for k, v in params.items()},
                        metadata={"mtp_format": "mlx"})

    for name in ("raw.safetensors", "mlx.safetensors"):
        loaded = dict(tree_flatten(mtp.load_mtp_head(tmp_path / name, mtp.text_model(model).args).parameters()))
        assert loaded.keys() == params.keys()
        for key, value in params.items():
            assert mx.allclose(loaded[key], value, atol=1e-6), (name, key)


def test_quantized_head_loads(tmp_path):
    model = tiny_hybrid_model()
    head = random_head(model)
    mx.save_safetensors(str(tmp_path / "h.safetensors"),
                        {"mtp." + k: v for k, v in tree_flatten(head.parameters())}, metadata={"mtp_format": "mlx"})
    q = mtp.load_mtp_head(tmp_path / "h.safetensors", mtp.text_model(model).args, bits=8, group_size=32)
    assert isinstance(q.fc, nn.QuantizedLinear)
    h = mx.random.normal((1, 1, 128))
    e = mx.random.normal((1, 1, 128))
    assert mx.allclose(q(e, h), head(e, h), atol=5e-2)


@pytest.mark.parametrize("kept", [1, 2, 3, 4])
def test_commit_leaves_the_caches_as_reading_only_the_kept_tokens(kept):
    model = tiny_hybrid_model()
    verify_tokens = [11, 29, 4, 50]

    cache = prefilled(model, PROMPT)
    hidden, _ = mtp.verify_forward(model, verify_tokens, cache)
    mtp.commit(cache, len(verify_tokens), kept)

    reference = prefilled(model, PROMPT)
    ref_hidden = []
    for t in verify_tokens[:kept]:  # one token at a time, as plain decoding reads them
        ref_hidden.append(mtp.trunk_hidden(model, mx.array([[t]]), reference))
    ref_hidden = mx.concatenate(ref_hidden, axis=1)

    assert mx.allclose(hidden[:, :kept], ref_hidden, atol=1e-4)
    for got, want in zip(cache_arrays(cache), cache_arrays(reference), strict=True):
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=1e-4)
    # And the next token reads the same from both.
    nxt = mx.array([[33]])
    assert mx.allclose(model(nxt, cache=cache), model(nxt, cache=reference), atol=1e-4)


def oracle(reference, right):
    """A drafter that proposes the true greedy continuation, wrong after the first `right` drafts of each round."""
    def draft(n_out):
        good = reference[n_out:n_out + 3]
        return [t if i < right else (t + 1) % 64 for i, t in enumerate(good)]
    return draft


@pytest.mark.parametrize("right", [0, 1, 2, 3])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_greedy_output_is_plain_greedy_whatever_the_drafts(right, depth):
    model = tiny_hybrid_model()
    reference = plain_greedy(model, PROMPT, 40)
    stats = mtp.MTPStats()
    out = mtp.generate_mtp(model, None, PROMPT, max_tokens=40, depth=depth, drafter=oracle(reference, right),
                           stats=stats)
    assert out == reference
    if right >= depth:
        assert stats.acceptance == 1.0
    if right == 0:
        assert stats.accepted == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_greedy_output_with_a_real_head_is_plain_greedy(depth):
    model = tiny_hybrid_model()
    head = random_head(model)
    reference = plain_greedy(model, PROMPT, 40)
    stats = mtp.MTPStats()
    assert mtp.generate_mtp(model, head, PROMPT, max_tokens=40, depth=depth, stats=stats) == reference
    assert stats.rounds > 0 and stats.drafted > 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_each_round_drafts_from_the_true_history(depth):
    """The head's cache holds exactly the true (hidden t, token t+1) rows: each round's first draft equals the head
    run fresh over the whole committed sequence. Draft rows left behind, or true rows missed, would change it."""
    model = tiny_hybrid_model()
    head = random_head(model)
    stats = mtp.MTPStats(log_drafts=True)
    out = mtp.generate_mtp(model, head, PROMPT, max_tokens=30, depth=depth, stats=stats)
    seq = PROMPT + out

    cache = make_prompt_cache(model)
    hidden = mtp.trunk_hidden(model, mx.array([seq]), cache)
    rows = head(mtp.embed(model, mx.array([seq[1:]])), hidden[:, :-1], head.make_cache())
    first_drafts = mx.argmax(mtp.lm_logits(model, rows), axis=-1)[0].tolist()  # row i predicts seq[i + 2]

    assert stats.drafts_log
    for n_out, drafts in stats.drafts_log:
        newest = len(PROMPT) + n_out - 1  # position of the newest committed token
        assert drafts[0] == first_drafts[newest - 1], (n_out, drafts)


def test_stop_token_ends_generation_even_as_an_accepted_draft():
    model = tiny_hybrid_model()
    reference = plain_greedy(model, PROMPT, 40)
    stop = reference[9]
    first = reference.index(stop)
    out = mtp.generate_mtp(model, None, PROMPT, max_tokens=40, depth=3, drafter=oracle(reference, 3),
                           stop_tokens=[stop])
    assert out == reference[:first + 1]


def test_processors_see_the_same_calls_as_plain_decoding():
    model = tiny_hybrid_model()
    reference = plain_greedy(model, PROMPT, 25)
    calls = []

    def recorder(tokens, logits):
        calls.append(tokens.tolist())
        return logits

    mtp.generate_mtp(model, None, PROMPT, max_tokens=25, depth=3, drafter=oracle(reference, 2),
                     processors=[recorder])
    assert calls == [PROMPT + reference[:i] for i in range(25)]


@pytest.mark.parametrize("settings", [dict(temperature=1.0), dict(temperature=0.6, top_p=0.95, top_k=20),
                                      dict(temperature=0.7, top_k=5), dict(temperature=0.8, top_p=0.5)])
def test_speculative_sampling_draws_the_target_distribution(settings):
    """Whatever the draft, the chosen token is distributed as the target (accept p(d), else resample without d)."""
    mx.random.seed(3)
    logits = mx.random.normal((1, 24)) * 2.0
    sampling = mtp.Sampling(**settings)
    target = sampling.probs(logits)[0]
    draft = int(mx.argsort(-target)[1].item())  # a likely but not top token
    n = 6000
    counts = mx.zeros((24,))
    keys = mx.random.split(mx.random.key(7), n)
    picks = [mtp._choose(logits, draft, sampling, keys[i])[0] for i in range(n)]
    counts = mx.array([picks.count(t) for t in range(24)], dtype=mx.float32) / n
    total_variation = 0.5 * mx.abs(counts - target).sum().item()
    assert total_variation < 0.04, total_variation
    assert all(counts[t].item() == 0 for t in range(24) if target[t].item() == 0)


@pytest.mark.parametrize("settings", [dict(temp=0.6, top_p=0.95, top_k=20), dict(temp=1.0, top_k=5),
                                      dict(temp=0.8, top_p=0.5)])
def test_target_distribution_matches_mlx_lm_sampler(settings):
    """Sampling.probs is the distribution mlx-lm's own sampler draws from (same filters, same order)."""
    mx.random.seed(5)
    logits = mx.random.normal((1, 24)) * 2.0
    ours = mtp.Sampling(temperature=settings["temp"], top_p=settings.get("top_p", 0.0),
                        top_k=settings.get("top_k", 0)).probs(logits)[0]
    sampler = make_sampler(**settings)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    n = 6000
    mx.random.seed(9)
    picks = [int(sampler(logprobs).item()) for _ in range(n)]
    theirs = mx.array([picks.count(t) for t in range(24)], dtype=mx.float32) / n
    assert 0.5 * mx.abs(theirs - ours).sum().item() < 0.04
    support = {t for t in range(24) if ours[t].item() > 0}
    assert {t for t in range(24) if theirs[t].item() > 0} <= support


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("head_hidden", ["post", "pre"])
def test_cheaper_drafts_and_pre_norm_rows_keep_the_output_exact(depth, head_hidden):
    """Drafts from a 4-bit copy of the output layer, or a head reading the trunk's pre-norm hidden state, change only
    which drafts are accepted: the trunk checks every one, so greedy output stays plain greedy."""
    model = tiny_hybrid_model()
    head = random_head(model)
    reference = plain_greedy(model, PROMPT, 40)
    draft = mtp.draft_lm_head(model, bits=4, group_size=32, rows_per_chunk=16)
    out = mtp.generate_mtp(model, head, PROMPT, max_tokens=40, depth=depth, draft_logits=draft,
                           head_hidden=head_hidden)
    assert out == reference
    norm = mtp.text_model(model).model.norm
    assert not isinstance(norm, mtp._NormTap), "the tap comes off after generating"


def test_pre_norm_rows_are_what_the_trunks_final_norm_reads():
    model = tiny_hybrid_model()
    text = mtp.text_model(model)
    seen = []
    original = text.model.norm
    tap = mtp._NormTap(original)
    text.model.norm = tap
    try:
        cache = make_prompt_cache(model)
        post = mtp.trunk_hidden(model, mx.array([PROMPT]), cache)
        seen.append(tap.last)
    finally:
        text.model.norm = original
    assert mx.allclose(original(seen[0]), post, atol=1e-6)
    assert not mx.allclose(seen[0], post, atol=1e-3)


def test_draft_head_is_built_a_block_at_a_time_and_tracks_the_real_one():
    model = tiny_hybrid_model()
    h = mx.random.normal((1, 1, mtp.text_model(model).args.hidden_size))
    whole = mtp.draft_lm_head(model, bits=4, group_size=32, rows_per_chunk=1 << 20)
    blocks = mtp.draft_lm_head(model, bits=4, group_size=32, rows_per_chunk=16)
    assert mx.array_equal(whole.weight, blocks.weight) and mx.array_equal(whole.scales, blocks.scales)
    assert whole(h).shape == mtp.lm_logits(model, h).shape
    assert mx.allclose(whole(h), mtp.lm_logits(model, h), atol=0.25)
    assert mtp.draft_lm_head(model)(h).shape == mtp.lm_logits(model, h).shape  # no bits: the model's own


def test_pre_mode_feeds_the_head_the_pre_norm_rows():
    model = tiny_hybrid_model()
    head = random_head(model)
    seen = []

    class Spy:
        make_cache = staticmethod(head.make_cache)

        def __call__(self, embeddings, hidden, cache=None):
            seen.append(hidden)
            return head(embeddings, hidden, cache)

    mtp.generate_mtp(model, Spy(), PROMPT, max_tokens=4, depth=1, head_hidden="pre")
    text = mtp.text_model(model)
    tap = mtp._NormTap(text.model.norm)
    text.model.norm = tap
    try:
        post = mtp.trunk_hidden(model, mx.array([PROMPT]), make_prompt_cache(model))
        pre = tap.last
    finally:
        text.model.norm = tap.inner
    assert mx.allclose(seen[0], pre[:, :len(PROMPT) - 1], atol=1e-5)  # the prompt's rows, as the head reads them
    assert not mx.allclose(seen[0], post[:, :len(PROMPT) - 1], atol=1e-3)
