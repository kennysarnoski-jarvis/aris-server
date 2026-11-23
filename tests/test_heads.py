"""Draft heads given as a Hugging Face repo's file ("owner/repo:file") are sized before they download, download with
the model on `aris-server pull`, and come from the Hub's cache after; local files are used as they are."""

import json
from dataclasses import replace

from aris_server import heads
from aris_server.config import ModelSpec, _spec
from test_hub_weights import GB, Q4, Q8, FakeHub, arete_config, arete_repo, aris_spec, pull_events

HEAD = "someone/a-draft-head"
REF = f"{HEAD}:heads/mtp.safetensors"
ARIS_TEXT = {"vocab_size": 248_320, "hidden_size": 5120}  # Aris 3.8-27B's output layer


def head_repo(size=int(0.85 * GB)):
    return {"heads/mtp.safetensors": size, "README.md": 1_000}


def test_a_head_is_a_hub_file_or_a_local_path(tmp_path, monkeypatch):
    assert heads.hub_file(REF) == (HEAD, "heads/mtp.safetensors")
    assert heads.hub_file("thinkingmachines/Inkling-2-Mini:mtp.safetensors") == (
        "thinkingmachines/Inkling-2-Mini", "mtp.safetensors")
    for local in ("~/models/qwen3.8-27b-mtp.safetensors", "/Volumes/models/mtp.safetensors", "mtp.safetensors", None):
        assert heads.hub_file(local) is None
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b:c.safetensors").write_bytes(b"x")  # a local file that reads like a repo's: the file wins
    assert heads.hub_file("a/b:c.safetensors") is None


def test_models_toml_keeps_a_hub_head_as_written_and_expands_a_local_one():
    assert _spec("m", {"path": "repo/x", "mtp": REF}, "m").mtp == REF
    local = _spec("m", {"path": "repo/x", "mtp": "~/h.safetensors"}, "m").mtp
    assert local.endswith("/h.safetensors") and not local.startswith("~")


def test_a_hub_head_downloads_once_then_comes_from_the_cache(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch, tmp_path, {HEAD: head_repo()})
    assert heads.local_head(REF).endswith("heads/mtp.safetensors")
    assert hub.pulled == [HEAD] and hub.patterns == ["heads/mtp.safetensors"]  # the head's file and nothing else
    hub.cached_files = {(HEAD, "heads/mtp.safetensors")}
    assert heads.local_head(REF) == str(tmp_path / "heads/mtp.safetensors")
    assert hub.pulled == [HEAD]  # not again
    assert heads.local_head("~/x.safetensors").endswith("/x.safetensors") and hub.pulled == [HEAD]


def test_a_heads_bytes_count_before_anything_loads(monkeypatch, tmp_path):
    FakeHub(monkeypatch, tmp_path, {HEAD: head_repo()})
    spec = replace(aris_spec(), mtp=REF)
    config = {"text_config": ARIS_TEXT}
    assert heads.head_bytes(REF) == heads.extra_bytes(spec, config) == int(0.85 * GB)
    # The 4-bit copy of the output layer its drafts go through: 0.72 GB beside Aris 3.8-27B, scales and biases in.
    copy = heads.extra_bytes(replace(spec, mtp_draft_bits=4), config) - int(0.85 * GB)
    assert copy == int(248_320 * 5120 * (4 / 8 + 4 / 64)) and 0.71e9 < copy < 0.72e9
    inkling = {"model_type": "inkling2_loop", "text_config": ARIS_TEXT}
    assert heads.extra_bytes(replace(spec, mtp_draft_bits=4), inkling) == int(0.85 * GB)  # Inkling keeps no copy
    assert heads.extra_bytes(replace(spec, mtp=None, mtp_draft_bits=4), config) == 0
    local = tmp_path / "head.safetensors"
    with open(local, "wb") as f:
        f.truncate(1 << 28)
    assert heads.head_bytes(str(local)) == 1 << 28


def test_a_head_counts_in_which_weights_this_mac_loads(monkeypatch, tmp_path):
    from aris_server.engine import choose_weights

    FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1), HEAD: head_repo()})
    plain, with_head = aris_spec(), replace(aris_spec(), mtp=REF)
    # Some Mac leaves the 8-bit just enough context on its own, and not beside its head.
    tipped = [gb / 10 for gb in range(300, 400)
              if choose_weights(plain, int(gb / 10 * GB)).path == Q8
              and choose_weights(with_head, int(gb / 10 * GB)).path == Q4]
    assert tipped


def test_pull_downloads_a_hub_head_with_the_model(monkeypatch, tmp_path, capsys):
    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1), HEAD: head_repo()})
    code, events = pull_events(replace(aris_spec(), mtp=REF), 37.4, capsys)  # a 48 GB Mac: the 8-bit
    chosen, done = events[0], events[-1]
    assert code == 0 and hub.pulled == [Q8, HEAD]  # the weights, then the head
    assert (chosen["path"], chosen["head"], chosen["head_bytes"]) == (Q8, REF, int(0.85 * GB))
    total = int(26.6 * GB) + 19_989_339 + 4_085 + int(0.85 * GB)  # what loading reads, and the head
    assert [e["done"] for e in events if e["event"] == "progress"][-1] == total
    assert all(e["total"] == total for e in events if e["event"] == "progress")
    assert done["event"] == "done" and done["head"].endswith("heads/mtp.safetensors")


def test_pull_of_a_local_model_still_downloads_its_hub_head(monkeypatch, tmp_path, capsys):
    hub = FakeHub(monkeypatch, tmp_path, {HEAD: head_repo()})
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(arete_config()))
    with open(folder / "model.safetensors", "wb") as f:
        f.truncate(int(15.1 * GB))  # sparse: sized, not written
    code, events = pull_events(ModelSpec(name="local", path=str(folder), kv_bits=8, mtp=REF), 37.4, capsys)
    assert code == 0 and hub.pulled == [HEAD]
    assert events[-1]["folder"] == str(folder) and events[-1]["head"].endswith("heads/mtp.safetensors")


def test_pull_says_which_download_failed(monkeypatch, tmp_path, capsys):
    hub = FakeHub(monkeypatch, tmp_path, {HEAD: head_repo()})
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(arete_config()))
    (folder / "model.safetensors").write_bytes(b"x")
    hub.fail = "connection reset"
    code, events = pull_events(ModelSpec(name="local", path=str(folder), kv_bits=8, mtp=REF), 37.4, capsys)
    assert code == 1 and events[-1]["event"] == "error" and REF in events[-1]["message"]
