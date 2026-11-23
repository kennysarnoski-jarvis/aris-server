"""`aris-server claude`: Claude Code on the model a running Aris Server has loaded, told the window to compact by."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request

# Claude Code compacts its conversation a fixed 33,000 tokens short of the window it's told. Told a small window as
# it is, it would compact about every turn (at 16k of a 49k window); told nothing, it takes the window as 200k and runs
# into the server's limit, where the summary request is refused too. Told claude_code_window's, it compacts at 80% of a
# small window, and where it always does on a large one. Aris Code's Aris Local tells it the same.
COMPACT_RESERVE = 33_000
COMPACT_AT_SHARE = 0.8


def claude_code_window(window: int) -> int:
    """The window to tell Claude Code (CLAUDE_CODE_MAX_CONTEXT_TOKENS) for a server that holds `window` tokens."""
    return max(int(window * COMPACT_AT_SHARE), window - COMPACT_RESERVE) + COMPACT_RESERVE


def served_model(base_url: str, api_key: str, timeout: float = 5.0) -> tuple[str, int | None]:
    """The loaded model's name and context window on this Mac, from the server's model list, where it comes first."""
    request = urllib.request.Request(f"{base_url}/v1/models", headers={"x-api-key": api_key})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        first = json.load(response)["data"][0]
    return first["id"], first.get("context_window")


def environment(base_url: str, api_key: str, window: int | None, env: dict[str, str]) -> dict[str, str]:
    """Claude Code's environment, pointed at the server. A window the user set stays theirs. A key for Anthropic's API
    is left out, so Claude Code doesn't send it to the server."""
    env = {k: v for k, v in env.items() if k != "ANTHROPIC_API_KEY"}
    env.update(ANTHROPIC_BASE_URL=base_url, ANTHROPIC_AUTH_TOKEN=api_key)
    if window and "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(claude_code_window(window))
    return env


def launch(host: str, port: int, api_key: str | None, args: list[str]) -> int:
    """Replace this process with Claude Code, on the model the server at host:port has loaded, passing it `args`.
    Returns 2, having said why, when there is no server to point it at or no Claude Code to start."""
    base_url, key = f"http://{host}:{port}", api_key or "local"
    try:
        name, window = served_model(base_url, key)
    except urllib.error.HTTPError as e:
        why = "wants its API key: pass --api-key" if e.code == 401 else f"answered {e.code} {e.reason}"
        print(f"Aris Server at {base_url} {why}", file=sys.stderr)
        return 2
    except OSError:  # refused, unreachable, timed out
        print(f"nothing answers at {base_url}. Start Aris Server first: `aris-server serve --model aris-3.8-27b` "
              "(a model takes a minute to load), or see `aris-server service status` if it runs as a service.",
              file=sys.stderr)
        return 2
    except (ValueError, LookupError, TypeError):
        print(f"{base_url} answered, but not with Aris Server's model list", file=sys.stderr)
        return 2
    claude = shutil.which("claude")
    if claude is None:
        print("no `claude` command on PATH: install Claude Code first", file=sys.stderr)
        return 2
    print(f"Claude Code on {name}" + (f", with {window:,} tokens of context on this Mac" if window else ""),
          file=sys.stderr)
    os.execve(claude, [claude, *args], environment(base_url, key, window, dict(os.environ)))
    return 0  # only where execve is stubbed out, as in the tests
