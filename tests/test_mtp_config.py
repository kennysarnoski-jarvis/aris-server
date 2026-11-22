"""MTP settings: models.toml parsing, cache identity, the head attached at load, and samplers labelled for exact
speculative sampling."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm import server as mls

from aris_server import mtp, mtp_serve
from aris_server.config import _spec
from aris_server.disk_cache import model_id
from tiny_models import tiny_hybrid_model


def test_models_toml_mtp_settings_parse_and_validate(tmp_path):
    head = tmp_path / "mtp.safetensors"
    spec = _spec("m", {"path": "repo/x", "mtp": str(head), "mtp_depth": 3, "mtp_bits": 0}, "m")
    assert (spec.mtp, spec.mtp_depth, spec.mtp_bits) == (str(head), 3, 0)
    plain = _spec("m", {"path": "repo/x"}, "m")
    assert (plain.mtp, plain.mtp_depth, plain.mtp_bits) == (None, 2, 8)
    for bad in ({"mtp_depth": 9}, {"mtp_depth": -1}, {"mtp_depth": 1.5}, {"mtp_bits": 6}):
        with pytest.raises(ValueError):
            _spec("m", {"path": "repo/x", **bad}, "m")


def test_caches_made_with_a_head_are_kept_apart(tmp_path):
    head = tmp_path / "mtp.safetensors"
    head.write_bytes(b"x")
    base = model_id("repo/x")
    with_head = model_id("repo/x", mtp=str(head), mtp_bits=8)
    assert base != with_head
    assert with_head != model_id("repo/x", mtp=str(head), mtp_bits=4)
    assert with_head == model_id("repo/x", mtp=str(head), mtp_bits=8)


def test_the_head_is_attached_when_the_model_loads(tmp_path, monkeypatch):
    with mx.stream(mx.cpu):
        model = tiny_hybrid_model()
        head = mtp.MTPHead(mtp.text_model(model).args)
        mx.eval(head.parameters())
        path = tmp_path / "mtp.safetensors"
        mx.save_safetensors(str(path), {"mtp." + k: v for k, v in tree_flatten(head.parameters())},
                            metadata={"mtp_format": "mlx"})
        provider = SimpleNamespace(model=None, _mtp_spec=SimpleNamespace(mtp=str(path), mtp_bits=0, mtp_depth=3,
                                                                    mtp_adaptive=False))

        def fake_load(self, *a, **k):
            self.model = model

        monkeypatch.setattr(mtp_serve, "_original_load", fake_load)
        n = len(model.make_cache())
        mtp_serve._load_with_mtp(provider)
        assert provider.model._mtp_head is not None
        assert provider.model._mtp_config.depth == 3
        assert len(provider.model.make_cache()) == n + 2
        mtp_serve._load_with_mtp(provider)  # loading again doesn't stack a second head
        assert len(provider.model.make_cache()) == n + 2


def test_memory_sizing_counts_the_heads_weights():
    from aris_server.engine import model_bytes

    with mx.stream(mx.cpu):
        model = tiny_hybrid_model()
        before = model_bytes(model)
        head = mtp.MTPHead(mtp.text_model(model).args)
        mx.eval(head.parameters())
        mtp_serve.attach_mtp(model, head)
        assert model_bytes(model) == before + sum(v.nbytes for _, v in tree_flatten(head.parameters()))


def test_a_model_without_mtp_settings_loads_untouched(monkeypatch):
    with mx.stream(mx.cpu):
        model = tiny_hybrid_model()
        provider = SimpleNamespace(model=None, _mtp_spec=None)
        monkeypatch.setattr(mtp_serve, "_original_load", lambda self, *a, **k: setattr(self, "model", model))
        mtp_serve._load_with_mtp(provider)
        assert getattr(provider.model, "_mtp_head", None) is None
        assert mtp_serve.mtp_snapshot(provider.model) is None


def _args(**sampling):
    defaults = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, xtc_probability=0.0, xtc_threshold=0.0)
    return SimpleNamespace(sampling=SimpleNamespace(**{**defaults, **sampling}))


TOKENIZER = SimpleNamespace(encode=lambda s: [], eos_token_ids=[])


def test_samplers_carry_the_distribution_they_draw_from():
    s = mls._make_sampler(_args(), TOKENIZER).mtp_sampling
    assert (s.temperature, s.top_p, s.top_k, s.min_p) == (0.6, 0.95, 20, 0.0)
    greedy = mls._make_sampler(_args(temperature=0.0), TOKENIZER)
    assert greedy.mtp_sampling.temperature == 0.0
    assert mls._make_sampler(_args(temperature=0.0), TOKENIZER) is greedy  # still one shared greedy sampler


def test_xtc_requests_are_not_labelled_so_they_decode_without_mtp():
    assert not hasattr(mls._make_sampler(_args(xtc_probability=0.5, xtc_threshold=0.1), TOKENIZER), "mtp_sampling")
