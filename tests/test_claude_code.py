"""`aris-server claude`: Claude Code started on the loaded model, told the window to compact by."""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aris_server import claude_code, cli


def test_claude_code_compacts_at_80_percent_of_a_small_window_and_33k_short_of_a_large_one():
    for window, told in ((49_152, 72_321), (100_352, 113_281), (200_000, 200_000)):
        assert claude_code.claude_code_window(window) == told
    # Where Claude Code then compacts, 33k short of what it's told: 80% of the 4-bit 27B's window on a 32 GB Mac
    # (not 16k, about every turn), and 33k short of a 200k one, as it always does.
    assert claude_code.claude_code_window(49_152) - claude_code.COMPACT_RESERVE == int(49_152 * 0.8)
    assert claude_code.claude_code_window(200_000) - claude_code.COMPACT_RESERVE == 167_000


MODELS = dict.fromkeys(("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                        "CLAUDE_CODE_SUBAGENT_MODEL"), "aris-3.8-27b")
LEFT_OUT = ("--disallowedTools=WebSearch,Workflow,ScheduleWakeup,ReportFindings,ListAgents,SendMessage,CronCreate,"
            "CronDelete,CronList,EnterWorktree,ExitWorktree")


def test_claude_code_starts_without_the_tools_a_local_session_has_no_use_for():
    # The same ones Aris Code's Aris Local leaves out, in one word attached with "=": given as a separate word, the
    # option takes a list and would take the prompt after it as more tool names.
    assert claude_code.arguments(["-p", "fix it"]) == [LEFT_OUT, "-p", "fix it"]
    assert claude_code.arguments(["fix it"]) == [LEFT_OUT, "fix it"]
    assert claude_code.arguments([]) == [LEFT_OUT]


def test_claude_code_is_pointed_at_the_server_and_keeps_a_window_the_user_set():
    env = claude_code.environment("http://127.0.0.1:8484", "local", "aris-3.8-27b", 100_352, {"PATH": "/bin"})
    assert env == {"PATH": "/bin", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8484", "ANTHROPIC_AUTH_TOKEN": "local",
                   "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "113281", "API_TIMEOUT_MS": "3000000",
                   "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1", **MODELS}
    mine = claude_code.environment("http://h:1", "k", "m", 100_352, {"CLAUDE_CODE_MAX_CONTEXT_TOKENS": "90000"})
    assert mine["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "90000"
    unsized = claude_code.environment("http://h:1", "k", "m", None, {})
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in unsized


def test_claude_code_waits_for_a_local_model_unless_the_user_said_otherwise():
    # Claude Code cuts a request off at 10 minutes, and a local model writing a whole file in one turn can take longer.
    env = claude_code.environment("http://h:1", "k", "m", None, {})
    assert env["API_TIMEOUT_MS"] == "3000000" and env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] == "1"
    mine = claude_code.environment("http://h:1", "k", "m", None, {"API_TIMEOUT_MS": "900000",
                                                                   "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "0"})
    assert mine["API_TIMEOUT_MS"] == "900000" and mine["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] == "0"


def test_claude_code_stays_on_the_server_whatever_the_shell_routes_it_to():
    shell = {"ANTHROPIC_API_KEY": "sk-ant-real", "CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_USE_VERTEX": "1",
             "CLAUDE_CODE_USE_FOUNDRY": "1", "ANTHROPIC_VERTEX_PROJECT_ID": "p", "CLAUDE_CODE_OAUTH_TOKEN": "t",
             "ANTHROPIC_CUSTOM_HEADERS": "x-team: a", "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
             "ANTHROPIC_MODEL": "claude-opus-5-5", "ANTHROPIC_DEFAULT_HAIKU_MODEL": "inkling-2-mini", "HOME": "/h"}
    env = claude_code.environment("http://127.0.0.1:8484", "local", "aris-3.8-27b", None, shell)
    assert env == {"HOME": "/h", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8484", "ANTHROPIC_AUTH_TOKEN": "local",
                   **claude_code.PATIENCE, **MODELS}


class _Models(BaseHTTPRequestHandler):
    status, body = 200, {"data": [{"type": "model", "id": "aris-3.8-27b", "context_window": 100_352},
                                  {"type": "model", "id": "inkling-2-mini"}]}
    keys = []

    def do_GET(self):
        type(self).keys.append(self.headers.get("x-api-key"))
        raw = json.dumps(self.body).encode() if isinstance(self.body, dict) else self.body
        self.send_response(self.status)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_):
        pass


@pytest.fixture
def server():
    def start(status=200, body=None):
        handler = type("Handler", (_Models,), {"status": status, "body": body or _Models.body, "keys": []})
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return httpd.server_address[1], handler

    started = []
    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """A `claude` on PATH, and os.execve recording what it would have started instead of starting it."""
    path = tmp_path / "claude"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    started = []
    monkeypatch.setattr(os, "execve", lambda file, argv, env: started.append((file, argv, env)))
    return str(path), started


def test_claude_code_starts_on_the_loaded_model_with_its_own_arguments(server, claude, monkeypatch, capsys):
    port, handler = server()
    path, started = claude
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert cli.main(["claude", "--port", str(port), "-p", "fix the failing test", "--continue"]) == 0
    [(file, argv, env)] = started
    assert file == path and argv == [path, LEFT_OUT, "-p", "fix the failing test", "--continue"]
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{port}" and env["ANTHROPIC_AUTH_TOKEN"] == "local"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "113281" and "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_MODEL"] == env["CLAUDE_CODE_SUBAGENT_MODEL"] == "aris-3.8-27b"
    assert handler.keys == ["local"]
    assert "Claude Code on aris-3.8-27b, with 100,352 tokens of context on this Mac" in capsys.readouterr().err


def test_arguments_after_a_double_dash_go_to_claude_code_whatever_they_are(server, claude):
    port, _ = server()
    path, started = claude
    assert cli.main(["claude", "--port", str(port), "--", "--port", "1"]) == 0
    assert started[0][1] == [path, LEFT_OUT, "--port", "1"]


def test_the_servers_key_is_passed_on(server, claude):
    port, handler = server()
    _, started = claude
    assert cli.main(["claude", "--port", str(port), "--api-key", "s3cret"]) == 0
    assert handler.keys == ["s3cret"] and started[0][2]["ANTHROPIC_AUTH_TOKEN"] == "s3cret"


def test_a_server_that_wants_a_key_says_so(server, claude, capsys):
    port, _ = server(status=401, body={"type": "error"})
    assert cli.main(["claude", "--port", str(port)]) == 2
    assert "pass --api-key" in capsys.readouterr().err and not claude[1]


def test_with_no_server_it_says_how_to_start_one(claude, capsys):
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert cli.main(["claude", "--port", str(port)]) == 2
    assert "aris-server serve --model" in capsys.readouterr().err and not claude[1]


def test_something_else_on_the_port_is_not_taken_for_aris_server(server, claude, capsys):
    port, _ = server(body=b"<html>not a model list</html>")
    assert cli.main(["claude", "--port", str(port)]) == 2
    assert "not with Aris Server's model list" in capsys.readouterr().err and not claude[1]


def test_without_claude_code_it_says_to_install_it(server, tmp_path, monkeypatch, capsys):
    port, _ = server()
    monkeypatch.setenv("PATH", str(tmp_path))  # no claude here
    assert cli.main(["claude", "--port", str(port)]) == 2
    assert "install Claude Code" in capsys.readouterr().err


def test_other_commands_still_refuse_arguments_they_dont_know():
    with pytest.raises(SystemExit):
        cli.main(["models", "--bogus"])
