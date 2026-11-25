"""The context limit a loaded model gets from this Mac's memory: for weights named by a Hugging Face repo id, as `pull`,
the built-in models and Aris Code's Aris Local load them, as well as for a local folder. Without it the server has no
"prompt is too long" limit, and `aris-server claude` can't tell Claude Code where to compact."""

import json
from types import SimpleNamespace

from mlx_lm.models.cache import KVCache

from aris_server import engine
from aris_server.config import ModelSpec

CONFIG = {"model_type": "llama", "num_hidden_layers": 4, "hidden_size": 256, "num_attention_heads": 8,
          "num_key_value_heads": 2, "head_dim": 32}


def loaded(path: str, layers: int = 4, kv_bits: int | None = None):
    """What _kv_bytes_per_token reads of an Engine: the loaded model's caches and its spec."""
    model = SimpleNamespace(make_cache=lambda: [KVCache() for _ in range(layers)])
    return SimpleNamespace(provider=SimpleNamespace(model=model), spec=ModelSpec(name="m", path=path, kv_bits=kv_bits))


def expected(layers: int, kv_bits: int | None = None) -> int:
    _, heads, head_dim = engine._kv_shape(CONFIG)
    return engine.kv_bytes_per_token(layers, heads, head_dim, kv_bits)


def test_weights_named_by_a_hub_repo_id_get_a_kv_size_per_token(monkeypatch):
    # The repo's config comes from the Hub, or from the cache `pull` filled; there's no folder at the repo id's path.
    monkeypatch.setattr(engine, "weights_and_config",
                        lambda path, adapter=None: (1 << 30, CONFIG) if path == "owner/model-4bit" else None)
    assert engine.Engine._kv_bytes_per_token(loaded("owner/model-4bit")) == expected(4) > 0
    assert engine.Engine._kv_bytes_per_token(loaded("owner/model-4bit", kv_bits=8)) == expected(4, 8) > 0


def test_a_local_folder_still_gets_one(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 64)
    assert engine.Engine._kv_bytes_per_token(loaded(str(tmp_path), layers=3)) == expected(3) > 0


def test_weights_nothing_can_size_get_none(monkeypatch):
    monkeypatch.setattr(engine, "weights_and_config", lambda path, adapter=None: None)
    assert engine.Engine._kv_bytes_per_token(loaded("owner/unknown")) is None
