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
    spec = replace(aris_spec(), mtp=REF, mtp_draft_bits=0)
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


ARIS_HEAD = "YourAris/Qwen3.8-27B-MTP-Head"
INKLING_HEAD = "thinkingmachines/Inkling-2-Mini"


def test_both_built_in_models_come_with_their_own_draft_heads():
    from aris_server.config import resolve

    aris, inkling = resolve("aris-3.8-27b", {}), resolve("inkling-2-mini", {})
    assert heads.hub_file(aris.mtp) == (ARIS_HEAD, "mtp.safetensors")
    assert heads.hub_file(inkling.mtp) == (INKLING_HEAD, "mtp.safetensors")
    assert (inkling.mtp_depth, inkling.mtp_adaptive, inkling.mtp_bits) == (7, True, 8)


def test_pulling_the_built_in_27b_fetches_its_head_with_its_weights(monkeypatch, tmp_path, capsys):
    from aris_server.config import resolve

    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1),
                                         ARIS_HEAD: {"mtp.safetensors": 849_400_563, "README.md": 4_238}})
    code, events = pull_events(resolve("aris-3.8-27b", {}), 37.4, capsys)  # a 48 GB Mac
    assert code == 0 and hub.pulled == [Q8, ARIS_HEAD]
    assert events[0]["head_bytes"] == 849_400_563 and events[-1]["head"].endswith("mtp.safetensors")


def test_the_4_bit_copy_is_counted_only_where_it_reads_less_than_the_layer():
    spec = replace(aris_spec(), mtp=REF, mtp_draft_bits=4)
    eight = {"quantization": {"group_size": 64, "bits": 8}, "text_config": ARIS_TEXT}
    four = {"quantization": {"group_size": 64, "bits": 4}, "text_config": ARIS_TEXT}
    override = {"quantization": {"group_size": 64, "bits": 4, "lm_head": {"group_size": 64, "bits": 8}},
                "text_config": ARIS_TEXT}
    assert (heads.output_bits(eight), heads.output_bits(four), heads.output_bits(override)) == (8, 4, 8)
    assert heads.output_bits({"text_config": ARIS_TEXT}) == 16  # full precision
    copy = int(248_320 * 5120 * (4 / 8 + 4 / 64))
    base = heads.head_bytes(REF)  # 0 here: no Hub to ask
    assert heads.extra_bytes(spec, eight) == base + copy
    assert heads.extra_bytes(spec, four) == base  # the 4-bit build drafts through its own layer
    assert heads.extra_bytes(spec, override) == base + copy


def test_a_head_that_cant_be_fetched_leaves_the_model_serving_without_it(monkeypatch, tmp_path, caplog):
    import logging

    from aris_server.config import resolve

    hub = FakeHub(monkeypatch, tmp_path, {})  # no head repo: offline, or one that moved
    hub.fail = "offline"
    spec = resolve("aris-3.8-27b", {})
    with caplog.at_level(logging.WARNING):
        served = heads.ready(spec)
    assert served.mtp is None and served.path == spec.path and "serving without it" in caplog.text
    local = replace(spec, mtp="/models/head.safetensors")
    assert heads.ready(local) is local  # a local file is the caller's to check
    hub.cached_files = {(ARIS_HEAD, "mtp.safetensors")}
    assert heads.ready(spec).mtp == str(tmp_path / "mtp.safetensors")  # the cached copy, offline too


def test_models_json_says_what_this_mac_would_load_and_what_is_downloaded(monkeypatch, tmp_path):
    from aris_server.config import resolve
    from aris_server.pull import describe

    hub = FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1),
                                         ARIS_HEAD: {"mtp.safetensors": 849_400_563}})
    spec = resolve("aris-3.8-27b", {})
    got = describe(spec, int(37.4 * GB), built_in=True)  # a 48 GB Mac: the 8-bit, beside its head
    assert (got["name"], got["built_in"], got["path"], got["bytes"]) == ("aris-3.8-27b", True, Q8, int(26.6 * GB))
    assert 32768 < got["context"] < describe(replace(spec, mtp=None), int(37.4 * GB), built_in=True)["context"]
    assert (got["downloaded"], got["head"], got["head_bytes"], got["head_downloaded"]) == (
        False, spec.mtp, 849_400_563, False)
    loading_reads = [f for f in arete_repo(26.6) if f.endswith((".json", ".safetensors"))]
    hub.cached_files = {(Q8, f) for f in loading_reads[:-1]}  # all but one file: not downloaded yet
    assert describe(spec, int(37.4 * GB), built_in=True)["downloaded"] is False
    hub.cached_files = {(Q8, f) for f in loading_reads} | {(ARIS_HEAD, "mtp.safetensors")}
    got = describe(spec, int(37.4 * GB), built_in=True)
    assert got["downloaded"] is True and got["head_downloaded"] is True
    assert describe(spec, int(21.3 * GB), built_in=True)["path"] == Q4  # a 32 GB Mac picks the 4-bit


def test_a_local_model_without_a_head_is_listed_as_on_disk(monkeypatch, tmp_path):
    from aris_server.pull import describe

    FakeHub(monkeypatch, tmp_path, {})
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(arete_config()))
    with open(folder / "model.safetensors", "wb") as f:
        f.truncate(int(15.1 * GB))
    got = describe(ModelSpec(name="mine", path=str(folder), kv_bits=8), int(37.4 * GB), built_in=False)
    assert (got["downloaded"], got["bytes"], got["head"], got["head_downloaded"]) == (True, int(15.1 * GB), None, None)


def test_the_models_command_prints_json_for_apps(monkeypatch, tmp_path, capsys):
    import mlx.core as mx

    from aris_server import cli

    FakeHub(monkeypatch, tmp_path, {Q8: arete_repo(26.6), Q4: arete_repo(15.1)})
    monkeypatch.setenv("ARIS_SERVER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": int(37.4 * GB)})
    assert cli.main(["models", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)["models"]
    assert [m["name"] for m in listed] == ["aris-3.8-27b", "inkling-2-mini"]
    assert listed[0]["path"] == Q8 and listed[0]["downloaded"] is False and listed[1]["bytes"] is None  # unknown here
