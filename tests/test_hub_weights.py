"""Weights on the Hugging Face Hub are sized before anything big downloads: a Mac picks what it can hold first."""

import json
import types

GB = 1 << 30


def arete_config():
    """Arete-27B's attention shape: 16 of its 64 layers keep a KV cache."""
    layers = (["linear_attention"] * 3 + ["full_attention"]) * 16
    return {"text_config": {"num_hidden_layers": 64, "layer_types": layers, "num_attention_heads": 24,
                            "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 5120}}


class FakeHub:
    """The parts of huggingface_hub the server uses, for repos given as {repo: {file: size}}. `cached` maps a repo to
    a local snapshot folder, as the cache would have it after a download."""

    def __init__(self, monkeypatch, tmp_path, repos, online=True, cached=None):
        import huggingface_hub

        self.repos, self.online, self.cached, self.asked, self.downloads = repos, online, cached or {}, [], []
        hub = self

        class Api:
            def model_info(self, repo, files_metadata=False):
                hub.asked.append(repo)
                if not hub.online:
                    raise ConnectionError("offline")
                if repo not in hub.repos:
                    raise FileNotFoundError(f"no repo {repo}")  # stands in for RepositoryNotFoundError
                files = [types.SimpleNamespace(rfilename=f, size=s) for f, s in hub.repos[repo].items()]
                return types.SimpleNamespace(sha="0123abcd", siblings=files)

        def hf_hub_download(repo, filename, revision=None, **_):
            hub.downloads.append((repo, filename))
            path = tmp_path / "hub" / repo.replace("/", "--") / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(arete_config()))
            return str(path)

        def snapshot_download(repo, local_files_only=False, **_):
            if local_files_only and repo in hub.cached:
                return str(hub.cached[repo])
            raise FileNotFoundError(f"{repo} isn't in the cache")  # stands in for LocalEntryNotFoundError

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)


def arete_repo(weight_gb):
    """An MLX repo's listing: two weight shards, and the files that come with them but aren't weights."""
    half = int(weight_gb * GB) // 2
    return {"model-00001-of-00002.safetensors": half, "model-00002-of-00002.safetensors": int(weight_gb * GB) - half,
            "tokenizer.json": 19_989_339, "README.md": 6_390, "LICENSE": 11_544, "config.json": 4_085}


def test_a_hugging_face_repo_is_sized_from_the_hub_before_its_weights_download(monkeypatch, tmp_path):
    from aris_server.engine import weights_and_config

    hub = FakeHub(monkeypatch, tmp_path, {"YourAris/Aris-3.8-27B-MLX-8bit": arete_repo(26.6)})
    weights, config = weights_and_config("YourAris/Aris-3.8-27B-MLX-8bit")
    assert weights == int(26.6 * GB)  # the weight shards only: the tokenizer and the card don't take GPU memory
    assert config == arete_config()
    assert hub.downloads == [("YourAris/Aris-3.8-27B-MLX-8bit", "config.json")]  # a few kilobytes, nothing big

    # An adapter on top of Hub weights is a local folder: its bytes count too.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with open(adapter / "adapters.safetensors", "wb") as f:
        f.truncate(1 << 28)
    assert weights_and_config("YourAris/Aris-3.8-27B-MLX-8bit", str(adapter))[0] == int(26.6 * GB) + (1 << 28)


def test_a_model_on_the_hub_loads_the_best_weights_this_mac_holds_and_asks_the_hub_once(monkeypatch, tmp_path):
    from aris_server.config import ModelSpec
    from aris_server.engine import choose_weights, weights_and_config

    q8, q4 = "YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit"
    hub = FakeHub(monkeypatch, tmp_path, {q8: arete_repo(26.6), q4: arete_repo(15.1)})
    spec = ModelSpec(name="aris-3.8-27b", path=q8, choices=(q8, q4), kv_bits=8)
    assert choose_weights(spec, int(37.4 * GB)).path == q8  # a 48 GB Mac: about 125k tokens beside the 8-bit
    assert choose_weights(spec, int(21.3 * GB)).path == q4  # a 32 GB Mac: only the 4-bit fits
    weights_and_config(q8)  # the server sizes the chosen weights once more before loading them
    assert sorted(hub.asked) == [q4, q8]


def test_offline_a_downloaded_model_is_sized_from_the_cache(monkeypatch, tmp_path):
    from aris_server.config import ModelSpec
    from aris_server.engine import choose_weights, weights_and_config

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with open(snapshot / "model.safetensors", "wb") as f:
        f.truncate(int(15.1 * GB))  # sparse: no disk used
    (snapshot / "config.json").write_text(json.dumps(arete_config()))
    q8, q4 = "YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit"
    FakeHub(monkeypatch, tmp_path, {}, online=False, cached={q4: snapshot})
    assert weights_and_config(q4) == (int(15.1 * GB), arete_config())
    assert weights_and_config(q8) is None  # not downloaded, and the Hub can't be asked
    # The 8-bit can't be sized, so the 4-bit already here is what loads, even on a Mac that could hold the 8-bit.
    assert choose_weights(ModelSpec(name="m", path=q8, choices=(q8, q4), kv_bits=8), int(37.4 * GB)).path == q4


def test_weights_missing_from_the_hub_are_skipped(monkeypatch, tmp_path):
    from aris_server.config import ModelSpec
    from aris_server.engine import choose_weights, weights_and_config

    q8, q4 = "YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit"
    FakeHub(monkeypatch, tmp_path, {q8: arete_repo(26.6)})  # the 4-bit isn't published
    assert weights_and_config(q4) is None
    spec = ModelSpec(name="aris-3.8-27b", path=q8, choices=(q8, q4), kv_bits=8)
    assert choose_weights(spec, int(37.4 * GB)).path == q8
    # On a 32 GB Mac nothing that can be sized fits: the 8-bit is chosen, and its size refuses it before it downloads.
    assert choose_weights(spec, int(21.3 * GB)).path == q8


def test_a_repo_without_mlx_weight_files_isnt_sized_as_empty(monkeypatch, tmp_path):
    from aris_server.engine import weights_and_config

    FakeHub(monkeypatch, tmp_path, {"someone/gguf-only": {"model-Q4_K_M.gguf": 15 * GB, "README.md": 100}})
    assert weights_and_config("someone/gguf-only") is None
