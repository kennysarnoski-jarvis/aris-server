"""Inkling 2's draft head and speculative decoding with it: the drafter matches the checkpoint's layout, the target's
taps are the hidden states the head was trained on, and every cache rewinds exactly, so speculative output is the
target's own."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache

from aris_server import dspark
from aris_server.models.inkling2 import layers as inkling_layers
from aris_server.speculative import Sampling, SpeculativeStats, speculate
from test_inkling2_model import LOOP_CONFIG, UNPADDED_VOCAB, create_model

# The tiny looped target has 14 layer visits (8 layers, 3 of them looped 3 times) and an 8-token window.
VISITS = 14
MTP_CONFIG = {
    "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "intermediate_size": 64, "max_position_embeddings": 4096, "torch_dtype": "bfloat16", "q_bias": False,
    "o_bias": False, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0, 2],
    "sliding_window_size": 4, "block_size": 8, "markov_rank": 8, "markov_head_type": "rnn_hidden",
    "target_layer_ids": [2, 7, 12, VISITS],
}

# Inkling 2 Mini's mtp.safetensors, name -> shape, as its header lists them.
_LAYER = {
    "dflare_norm.weight": [2304], "input_layernorm.weight": [2304], "post_attention_layernorm.weight": [2304],
    "mlp.down_proj.weight": [2304, 9216], "mlp.gate_proj.weight": [9216, 2304], "mlp.up_proj.weight": [9216, 2304],
    "self_attn.k_norm.weight": [128], "self_attn.k_norm_ctx.weight": [128], "self_attn.q_norm.weight": [128],
    "self_attn.k_proj.weight": [1024, 2304], "self_attn.k_proj_ctx.weight": [1024, 2304],
    "self_attn.v_proj.weight": [1024, 2304], "self_attn.v_proj_ctx.weight": [1024, 2304],
    "self_attn.o_proj.weight": [2304, 3072], "self_attn.q_proj.weight": [3072, 2304],
}
MINI_HEADER = {
    **{f"layers.{i}.{k}": v for i in range(3) for k, v in _LAYER.items()},
    "fuse_weight": [4, 4], "mask_embed": [2304], "norm.weight": [2304],
    "markov_head.dflare_norm.weight": [2304], "markov_head.hidden_out.bias": [2304],
    "markov_head.hidden_out.weight": [2304, 256], "markov_head.markov_w1.weight": [201024, 256],
    "markov_head.rnn_proj.bias": [768], "markov_head.rnn_proj.weight": [768, 5120],
}
MINI_MTP_CONFIG = {
    "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8,
    "head_dim": 128, "intermediate_size": 9216, "max_position_embeddings": 1048576, "torch_dtype": "bfloat16",
    "q_bias": False, "o_bias": False, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "local_layer_ids": [0, 2],
    "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
    "target_layer_ids": [5, 23, 41, 48],
}


@pytest.fixture(scope="module")
def model():
    return create_model(config=LOOP_CONFIG)


def random_drafter(model, seed: int = 1) -> dspark.Drafter:
    drafter = dspark.Drafter(dspark.DraftConfig.from_target({"mtp_config": MTP_CONFIG}, model.args.text))
    mx.random.seed(seed)
    weights = []
    for k, v in tree_flatten(drafter.parameters()):
        if k.endswith("norm.weight"):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in ** -0.5
        weights.append((k, v))
    drafter.update(tree_unflatten(weights))
    mx.eval(drafter.parameters())
    return drafter.attach(model)


def plain_greedy(model, prompt, n):
    return [int(t) for t, _ in generate_step(mx.array(prompt), model, max_tokens=n)]


def prompt_of(length, seed=0):
    return mx.random.randint(0, UNPADDED_VOCAB, (length,), key=mx.random.key(seed)).tolist()


class _MiniText:
    """Inkling 2 Mini's text config, as far as the drafter reads it."""
    hidden_size, num_hidden_layers, vocab_size, unpadded_vocab_size = 2304, 24, 201024, 200058
    final_logit_softcapping = 30.0
    loop_config = {"core_start": 6, "core_end": 18, "num_passes": 3}
    loop_core = range(6, 18)


def test_drafter_has_the_checkpoints_layout():
    # Nothing is evaluated, so nothing of Mini's size is allocated.
    drafter = dspark.Drafter(dspark.DraftConfig.from_target({"mtp_config": MINI_MTP_CONFIG}, _MiniText))
    assert {k: list(v.shape) for k, v in tree_flatten(drafter.parameters())} == MINI_HEADER


def test_draft_config_is_checked(model):
    text = model.args.text
    with pytest.raises(ValueError, match="no mtp_config"):
        dspark.DraftConfig.from_target({}, text)
    for bad in ([2, 7, 12], [7, 2, VISITS], [2, 7, 13], [VISITS]):
        with pytest.raises(ValueError, match="target_layer_ids"):
            dspark.DraftConfig.from_target({"mtp_config": {**MTP_CONFIG, "target_layer_ids": bad}}, text)
    with pytest.raises(ValueError, match="Unsupported"):
        dspark.DraftConfig.from_target({"mtp_config": {**MTP_CONFIG, "hidden_size": 32}}, text)
    with pytest.raises(ValueError, match="Unsupported"):
        dspark.DraftConfig.from_target({"mtp_config": {**MTP_CONFIG, "markov_head_type": "mlp"}}, text)


def test_taps_are_the_inputs_of_those_visits(model, monkeypatch):
    seen = []
    call = inkling_layers.DecoderLayer.__call__

    def record(self, hidden_states, *args, **kwargs):
        seen.append(hidden_states)
        return call(self, hidden_states, *args, **kwargs)

    tokens = mx.array([prompt_of(6)])
    plain = model.model.language_model(tokens, cache=make_prompt_cache(model))
    monkeypatch.setattr(inkling_layers.DecoderLayer, "__call__", record)
    taps = (0, 2, 3, 7, 12, 13, VISITS)
    hidden, got = dspark.read(model, tokens, make_prompt_cache(model), taps)
    assert len(seen) == VISITS
    for visit, tap in zip(taps, got):
        expected = hidden if visit == VISITS else seen[visit]
        assert mx.array_equal(tap, expected), visit
    assert mx.allclose(hidden, plain, atol=1e-6)


@pytest.mark.parametrize("length", [1, 5, 9, 30])
@pytest.mark.parametrize("depth", [7, 3])
def test_greedy_output_is_the_targets(model, length, depth):
    drafter = random_drafter(model)
    prompt = prompt_of(length, seed=length)
    stats = SpeculativeStats()
    # A prefill step longer than the drafter's 4-token window makes its local layers skip rows.
    out = dspark.generate_dspark(model, drafter, prompt, max_tokens=24, depth=depth, prefill_step=7, stats=stats)
    assert out == plain_greedy(model, prompt, 24)
    assert stats.rounds > 0 and stats.drafted > 0


def _oracle(reference, plan):
    """Drafts of which exactly `right` are correct, round after round: (count, right) pairs from `plan`."""
    rounds = iter(plan * 100)

    def guesses(done):
        count, right = next(rounds)
        good = reference[done:done + count]
        wrong = [(t + 1) % UNPADDED_VOCAB for t in reference[done + right:done + count]]
        return good[:right] + wrong

    return guesses


PLAN = [(count, right) for count in range(1, 8) for right in range(count + 1)]


@pytest.mark.parametrize("length", [3, 12])
def test_every_rejection_depth_rewinds_exactly(model, length):
    prompt = prompt_of(length, seed=100 + length)
    reference = plain_greedy(model, prompt, 160)
    spec = dspark.DSparkSpeculator(model, None)
    stats = SpeculativeStats(log_drafts=True)
    out = speculate(spec, prompt, max_tokens=160, depth=7, drafter=_oracle(reference, PLAN), stats=stats)
    assert out == reference
    # Every round accepted exactly the drafts it was given right (the budget cuts the last rounds short).
    expected = sum(min(right, len(d)) for (count, right), (_, d) in zip(PLAN * 100, stats.drafts_log))
    assert stats.accepted == expected
    assert {len(d) for _, d in stats.drafts_log} >= set(range(1, 8))


def _rows(kv):
    """A visit's key and value rows in temporal order (a window keeps its last max_size)."""
    if isinstance(kv, RotatingKVCache):
        n = min(kv.offset, kv.max_size)
        return kv._temporal_order(kv.keys)[..., -n:, :], kv._temporal_order(kv.values)[..., -n:, :]
    return kv.keys[..., :kv.offset, :], kv.values[..., :kv.offset, :]


def test_rewound_cache_is_the_cache_of_the_tokens_kept(model):
    prompt = prompt_of(10, seed=7)
    reference = plain_greedy(model, prompt, 60)
    spec = dspark.DSparkSpeculator(model, None)
    out = speculate(spec, prompt, max_tokens=60, depth=7, drafter=_oracle(reference, [(7, 2), (5, 0), (3, 3)]))
    assert out == reference
    read = prompt + out[:-1]  # the newest token is chosen, not read yet
    plain = make_prompt_cache(model)
    model(mx.array([read]), cache=plain)
    for (kv, conv), (kv2, conv2) in zip(dspark._visits(spec.cache), dspark._visits(plain)):
        assert kv.offset == kv2.offset == len(read)
        for a, b in zip(_rows(kv), _rows(kv2)):
            assert a.shape == b.shape and mx.allclose(a, b, atol=1e-4)
        for a, b in zip(conv.cache, conv2.cache):
            assert a.shape == b.shape and mx.allclose(a, b, atol=1e-4)
        assert getattr(conv, "windows", None) is None


def test_drafter_context_follows_the_target(model):
    drafter = random_drafter(model)
    prompt = prompt_of(9, seed=3)
    spec = dspark.DSparkSpeculator(model, drafter)
    out = speculate(spec, prompt, max_tokens=30, depth=7)
    read = len(prompt) + len(out) - 1
    window = MTP_CONFIG["sliding_window_size"]
    for i, context in enumerate(spec.draft_cache.contexts):
        assert context.offset == read
        keys, _ = context.read()
        assert keys.shape[2] == (min(read, window) if i in MTP_CONFIG["local_layer_ids"] else read)

    # The context holds exactly what reading the committed tokens in one go would give it.
    fresh = drafter.make_cache()
    _, taps = dspark.read(model, mx.array([prompt + out[:-1]]), make_prompt_cache(model), drafter.config.target_layer_ids)
    drafter.observe(taps, fresh)
    for a, b in zip(spec.draft_cache.arrays(), fresh.arrays()):
        assert a.shape == b.shape and mx.allclose(a, b, atol=1e-4)


def test_proposals_are_tokens_of_the_vocabulary(model):
    drafter = random_drafter(model)
    cache = drafter.make_cache()
    _, taps = dspark.read(model, mx.array([prompt_of(5)]), make_prompt_cache(model), drafter.config.target_layer_ids)
    drafter.observe(taps, cache)
    for count in range(1, 8):
        guesses = drafter.propose(model, 3, count, cache).tolist()
        assert len(guesses) == count and all(0 <= g < UNPADDED_VOCAB for g in guesses)
    for count in (0, 8):
        with pytest.raises(ValueError):
            drafter.propose(model, 3, count, cache)


def test_load_drafter_from_a_checkpoint(model):
    drafter = random_drafter(model)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps({**LOOP_CONFIG, "mtp_config": MTP_CONFIG}))
        mx.save_safetensors(str(Path(d) / "mtp.safetensors"), dict(tree_flatten(drafter.parameters())))
        loaded = dspark.load_drafter(d, model)
        prompt = prompt_of(8, seed=5)
        assert dspark.generate_dspark(model, loaded, prompt, max_tokens=12) == \
            dspark.generate_dspark(model, drafter, prompt, max_tokens=12)
        weights = dict(tree_flatten(drafter.parameters()))
        weights.pop("mask_embed")
        mx.save_safetensors(str(Path(d) / "mtp.safetensors"), weights)
        with pytest.raises(ValueError):
            dspark.load_drafter(d, model)


def test_quantized_output_layer(model):
    quantized = create_model(config=LOOP_CONFIG)
    nn.quantize(quantized, group_size=32, bits=4, class_predicate=lambda path, m: path == "lm_head")
    assert isinstance(quantized.lm_head, nn.QuantizedLinear)
    drafter = random_drafter(quantized)
    rows = dspark._output_rows(quantized.lm_head, 0, UNPADDED_VOCAB)
    expected = mx.dequantize(quantized.lm_head.weight, quantized.lm_head.scales, quantized.lm_head.biases,
                             group_size=32, bits=4)[:UNPADDED_VOCAB]
    assert mx.array_equal(rows, expected)
    prompt = prompt_of(6, seed=9)
    assert dspark.generate_dspark(quantized, drafter, prompt, max_tokens=16) == plain_greedy(quantized, prompt, 16)


def test_sampling_is_seeded(model):
    drafter = random_drafter(model)
    prompt = prompt_of(7, seed=11)
    sampling = Sampling(temperature=1.0)
    runs = [dspark.generate_dspark(model, drafter, prompt, max_tokens=20, sampling=sampling, seed=s)
            for s in (4, 4, 5)]
    assert runs[0] == runs[1] and runs[0] != runs[2]
    assert all(0 <= t < UNPADDED_VOCAB for t in runs[0])


@pytest.mark.parametrize("bits", [8, 4])
def test_a_quantized_drafter_still_gives_the_targets_output(model, bits):
    config = {**MTP_CONFIG, "markov_rank": 32}  # the folded table quantizes in groups of 32 along its rank
    drafter = dspark.Drafter(dspark.DraftConfig.from_target({"mtp_config": config}, model.args.text))
    mx.random.seed(3)
    drafter.update(tree_unflatten([(k, 1 + 0.1 * mx.random.normal(v.shape) if k.endswith("norm.weight")
                                    else mx.random.normal(v.shape) * (v.shape[-1] if v.ndim > 1 else 16) ** -0.5)
                                   for k, v in tree_flatten(drafter.parameters())]))
    drafter.attach(model).quantize(bits, group_size=32)
    linears = [m for _, m in drafter.named_modules() if isinstance(m, (nn.Linear, nn.QuantizedLinear))]
    in_layers = [m for path, m in drafter.named_modules()
                 if path.startswith("layers.") and isinstance(m, (nn.Linear, nn.QuantizedLinear))]
    assert in_layers and all(isinstance(m, nn.QuantizedLinear) for m in in_layers)
    assert not isinstance(drafter.markov_head.rnn_proj, nn.QuantizedLinear)  # its weights feed float32 math
    assert drafter._fold_weight is None and drafter._fold_q is not None and len(linears) > len(in_layers)
    prompt = prompt_of(9, seed=21)
    assert dspark.generate_dspark(model, drafter, prompt, max_tokens=20) == plain_greedy(model, prompt, 20)
    drafter.attach(model)  # re-deriving after a weight change keeps the table quantized
    assert drafter._fold_weight is None and drafter._fold_q is not None


@pytest.mark.parametrize("rows", [2, 3, 5, 8])
def test_a_few_rows_of_full_attention_match_the_reference(rows):
    from aris_server.models.inkling2.layers import _attention_reference, _rows_attention

    S, H, Hkv, D, R = 21, 4, 2, 16, 8
    key = mx.random.key(rows)
    k1, k2, k3, k4, k5, k6 = mx.random.split(key, 6)
    q = mx.random.normal((1, H, rows, D), key=k1)
    k, v = mx.random.normal((1, Hkv, S, D), key=k2), mx.random.normal((1, Hkv, S, D), key=k3)
    bias, sinks = mx.random.normal((1, rows, H, R), key=k4), mx.random.normal((1, rows, H), key=k5)
    end = S  # the rows are the last `rows` keys, as after a cache update
    qpos, kpos = mx.arange(end - rows, end), mx.arange(S) + (end - S)
    expected = _attention_reference(q, k, v, bias, sinks, scale=D ** -0.5, window=0, query_positions=qpos,
                                    key_positions=kpos)
    got = _rows_attention(q, k, v, bias, sinks, scale=D ** -0.5, query_positions=qpos, key_positions=kpos)
    assert got.shape == expected.shape and mx.allclose(got, expected, atol=1e-5)
