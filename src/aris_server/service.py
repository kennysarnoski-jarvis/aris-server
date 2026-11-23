"""Run Aris Server as a launchd agent: it starts at login and comes back by itself after any exit (the server exits
on purpose when the generation engine dies, e.g. out of GPU memory)."""

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from .config import home

LABEL = "dev.aris.server"


def plist_path() -> Path:
    return Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"


def build_plist(executable: str, serve_args: list[str], log: Path) -> dict:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"}
    if "ARIS_SERVER_HOME" in os.environ:
        env["ARIS_SERVER_HOME"] = os.environ["ARIS_SERVER_HOME"]
    return {
        "Label": LABEL,
        "ProgramArguments": [executable, "serve", *serve_args],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",  # the GPU-bound work should not be throttled like a background job
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
        "EnvironmentVariables": env,
    }


def _executable() -> str:
    found = shutil.which("aris-server") or str(Path(sys.argv[0]).resolve())
    return str(Path(found).resolve())


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(serve_args: list[str]) -> int:
    log = home() / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        subprocess.run(["launchctl", "bootout", f"{_domain()}/{LABEL}"], capture_output=True)
    path.write_bytes(plistlib.dumps(build_plist(_executable(), serve_args, log)))
    done = subprocess.run(["launchctl", "bootstrap", _domain(), str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        print(f"launchctl bootstrap failed: {done.stderr.strip()}", file=sys.stderr)
        return 1
    print(f"Aris Server runs as a service now ({LABEL}); log: {log}")
    return 0


def uninstall() -> int:
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{LABEL}"], capture_output=True)
    plist_path().unlink(missing_ok=True)
    print("Aris Server service removed")
    return 0


def service_fields(text: str) -> dict[str, str]:
    """The service's own state, pid, runs and last exit code from `launchctl print`: the top level of its block only.
    Blocks nested in it carry lines of their own (each resource and jetsam coalition has a `state = active`), and
    reading those reported a running service as "active"."""
    fields, depth = {}, 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith("{"):
            depth += 1
        elif stripped == "}":
            depth -= 1
        elif depth == 1:
            key, _, value = stripped.partition(" = ")
            if key in ("state", "pid", "runs", "last exit code"):
                fields.setdefault(key, value)
    return fields


def status() -> int:
    done = subprocess.run(["launchctl", "print", f"{_domain()}/{LABEL}"], capture_output=True, text=True)
    if done.returncode != 0:
        print("not installed")
        return 1
    fields = service_fields(done.stdout)
    print(", ".join(f"{k}: {v}" for k, v in fields.items()) or done.stdout[:400])
    return 0
