"""Weights on the Hugging Face Hub are sized before anything big downloads (a Mac picks what it can hold first), and
`aris-server pull` downloads them ahead of serving, reporting progress for apps."""

import fnmatch
import json
import time
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

        def snapshot_download(repo, local_files_only=False, tqdm_class=None, allow_patterns=None, **_):
            if local_files_only:
                if repo in hub.cached:
                    return str(hub.cached[repo])
                raise FileNotFoundError(f"{repo} isn't in the cache")  # stands in for LocalEntryNotFoundError
            hub.pulled.append(repo)
            hub.patterns = allow_patterns
            if hub.fail:
                raise ConnectionError(hub.fail)
            wanted = {f: s for f, s in hub.repos[repo].items()
                      if not allow_patterns or any(fnmatch.fnmatch(f, p) for p in allow_patterns)}
            # Progress as huggingface_hub reports it: one summed bar for bytes received over the network, one for
            # bytes written to disk, and one counting files, which isn't about bytes.
            net = tqdm_class(desc="Downloading bytes", total=0, initial=0, unit="B", unit_scale=True)
            disk = tqdm_class(desc="Reconstructing (incomplete total...)", total=0, initial=0, unit="B",
                              unit_scale=True)
            files = tqdm_class(total=len(wanted), desc=f"Fetching {len(wanted)} files")
            for name, size in wanted.items():
                if (repo, name) in hub.cached_files:
                    continue  # already in the cache: huggingface_hub reports nothing for it
                time.sleep(hub.pace)
                net.update(size)  # Xet: received first, written in a later flush
                time.sleep(hub.pace)
                disk.update(size)
                files.update(1)
            folder = tmp_path / "snapshots" / repo.replace("/", "--")
            folder.mkdir(parents=True, exist_ok=True)
            return str(folder)

        def try_to_load_from_cache(repo, filename, revision=None, **_):
            return str(tmp_path / filename) if (repo, filename) in hub.cached_files else None

        self.pulled, self.fail, self.cached_files, self.pace = [], None, set(), 0.0
        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)


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


# --- the built-in model ---------------------------------------------------------------------------------------------


def test_aris_3_8_27b_is_built_in_with_the_settings_its_evals_ran_with(tmp_path):
    from aris_server.config import load_models, resolve

    spec = resolve("aris-3.8-27b", {})
    assert spec.choices == ("YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit")
    assert (spec.kv_bits, spec.effort_kwarg, spec.effort_map["high"]) == (8, "reasoning_effort", "medium")
    assert spec.budget_for("high") == 4096 and spec.budget_for("max") == 0

    # A models.toml entry of the same name replaces it; built-in models aren't listed as configured ones.
    (tmp_path / "models.toml").write_text('[models."aris-3.8-27b"]\npath = "~/models/my-own-build"\n')
    configured = load_models(tmp_path / "models.toml")
    assert resolve("aris-3.8-27b", configured).path.endswith("/models/my-own-build")
    assert load_models(tmp_path / "missing.toml") == {}


def test_the_models_command_lists_built_in_models_beside_configured_ones(tmp_path, monkeypatch, capsys):
    from aris_server.cli import main

    monkeypatch.setenv("ARIS_SERVER_HOME", str(tmp_path))
    (tmp_path / "models.toml").write_text('[models."mine"]\npath = "~/models/mine"\n')
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "mine" in out and "aris-3.8-27b" in out and "(built in)" in out


# --- pull ----------------------------------------------------------------------------------------------------------

Q8, Q4 = "YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit"


def pull_events(spec, limit, capsys):
    from aris_server.pull import pull

    code = pull(spec, int(limit * GB), as_json=True)
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def aris_spec():
    from aris_server.config import resolve

    return resolve("aris-3.8-27b", {})


def test_pull_downloads_the_weights_this_mac_would_load_and_says_where(monkeypatch, tmp_path, capsys):
    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1)})
    code, events = pull_events(aris_spec(), 21.3, capsys)  # a 32 GB Mac
    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

    assert code == 0 and hub.pulled == [Q4] and hub.patterns == DEFAULT_ALLOW_PATTERNS  # what loading reads
    chosen, done = events[0], events[-1]
    assert (chosen["event"], chosen["path"], chosen["bytes"]) == ("chosen", Q4, int(15.1 * GB))
    assert chosen["context"] > 32768
    assert (done["event"], done["path"]) == ("done", Q4) and done["folder"].endswith("YourAris--Aris-3.8-27B-MLX-4bit")
    # Everything loading needs, and only that: the weights, the tokenizer and the config, not the card or the license.
    total = int(15.1 * GB) + 19_989_339 + 4_085
    assert [e["done"] for e in events if e["event"] == "progress"][-1] == total
    assert all(e["total"] == total for e in events if e["event"] == "progress")


def test_pull_reports_progress_while_the_download_runs(monkeypatch, tmp_path, capsys):
    import aris_server.pull

    monkeypatch.setattr(aris_server.pull, "PROGRESS_EVERY", {True: 0.0, False: 0.0})
    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1)})
    hub.pace = 0.3  # each file takes a while to land
    code, events = pull_events(aris_spec(), 37.4, capsys)  # a 48 GB Mac: the 8-bit
    progress = [e["done"] for e in events if e["event"] == "progress"]
    total = events[-2]["total"]
    assert code == 0 and hub.pulled == [Q8]
    assert progress == sorted(progress)  # never goes backwards
    assert any(0 < p < total for p in progress)  # and is seen part way, not only at the end


def test_files_already_downloaded_count_as_done_from_the_start(monkeypatch, tmp_path, capsys):
    import aris_server.pull

    monkeypatch.setattr(aris_server.pull, "PROGRESS_EVERY", {True: 0.0, False: 0.0})
    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1)})
    hub.cached_files = {(Q4, "model-00001-of-00002.safetensors")}  # an earlier pull got this far
    hub.pace = 0.3
    code, events = pull_events(aris_spec(), 21.3, capsys)
    first = next(e for e in events if e["event"] == "progress")  # before any byte of this pull has moved
    assert code == 0 and first["done"] == int(15.1 * GB) // 2


def test_the_two_progress_bars_are_read_as_the_larger_of_bytes_received_and_bytes_written():
    from aris_server.pull import _Bytes

    moved = _Bytes()
    bar = moved.bar_class()
    net = bar(desc="Downloading bytes", total=0, unit="B")
    disk = bar(desc="Reconstructing (incomplete total...)", total=0, unit="B")
    files = bar(total=3, desc="Fetching 3 files")
    net.update(100)
    assert moved.moved() == 100  # Xet: received, not yet written
    disk.update(60)
    assert moved.moved() == 100
    disk.update(60)
    files.update(1)  # files, not bytes
    assert moved.moved() == 120  # HTTP files move only the disk bar


def test_pull_refuses_weights_this_mac_cant_hold_before_downloading_them(monkeypatch, tmp_path, capsys):
    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6)})  # no 4-bit published
    code, events = pull_events(aris_spec(), 21.3, capsys)
    assert code == 2 and hub.pulled == []
    assert events[-1]["event"] == "error" and "does not fit" in events[-1]["message"]


def test_pull_says_when_a_model_cant_be_found_or_downloaded(monkeypatch, tmp_path, capsys):
    from aris_server.config import resolve

    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6)})
    code, events = pull_events(resolve("nobody/no-such-model", {}), 37.4, capsys)
    assert code == 1 and hub.pulled == [] and events[-1]["event"] == "error"

    hub.fail = "connection reset by peer"
    code, events = pull_events(aris_spec(), 37.4, capsys)
    assert code == 1 and events[-1]["event"] == "error" and "connection reset" in events[-1]["message"]


def test_pull_of_a_local_folder_downloads_nothing(monkeypatch, tmp_path, capsys):
    from aris_server.config import resolve

    folder = tmp_path / "local-model"
    folder.mkdir()
    with open(folder / "model.safetensors", "wb") as f:
        f.truncate(int(15.1 * GB))
    (folder / "config.json").write_text(json.dumps(arete_config()))
    hub = FakeHub(monkeypatch, tmp_path, {})
    code, events = pull_events(resolve(str(folder), {}), 37.4, capsys)
    assert code == 0 and hub.pulled == [] and events[-1] == {"event": "done", "path": str(folder),
                                                             "folder": str(folder)}


def test_pull_speaks_plainly_to_people(monkeypatch, tmp_path, capsys):
    from aris_server.pull import pull

    FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1)})
    assert pull(aris_spec(), int(37.4 * GB)) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"aris-3.8-27b: {Q8} (28.6 GB), about ") and "downloaded" in out and "{" not in out
