import subprocess
from pathlib import Path

import pytest

from aris_server import service
from aris_server.service import LABEL, build_plist, service_fields


def test_the_service_restarts_the_server_after_any_exit():
    plist = build_plist("/opt/aris/bin/aris-server", ["--model", "arete:27b", "--port", "8484"], Path("/tmp/s.log"))
    assert plist["Label"] == LABEL
    assert plist["ProgramArguments"] == ["/opt/aris/bin/aris-server", "serve", "--model", "arete:27b", "--port", "8484"]
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True and plist["ThrottleInterval"] == 10
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == "/tmp/s.log"


# `launchctl print gui/501/dev.aris.server` for a running service, trimmed: its coalition blocks have states of their own.
RUNNING = """gui/501/dev.aris.server = {
	active count = 1
	path = /Library/LaunchAgents/dev.aris.server.plist
	type = LaunchAgent
	state = running

	program = /opt/aris/bin/aris-server
	arguments = {
		/opt/aris/bin/aris-server
		serve
	}

	runs = 3
	pid = 4242
	last exit code = 1

	resource coalition = {
		ID = 1234
		type = resource
		state = active
		active count = 1
	}

	jetsam coalition = {
		ID = 1235
		type = jetsam
		state = active
	}
}
"""


def test_status_reads_the_services_own_state_not_its_coalitions():
    assert service_fields(RUNNING) == {"state": "running", "runs": "3", "pid": "4242", "last exit code": "1"}
    stopped = RUNNING.replace("\tstate = running", "\tstate = not running").replace("\tpid = 4242\n", "")
    assert service_fields(stopped) == {"state": "not running", "runs": "3", "last exit code": "1"}


class FakeLaunchctl:
    """`launchctl` as `service install` meets it: the old job stays loaded for `loaded_checks` prints after its bootout,
    and bootstrap answers "Bootstrap failed: 5" `busy` times (launchd still unloading), or `error` every time."""

    def __init__(self, loaded_checks=0, busy=0, error=None):
        self.calls, self.loaded_checks, self.busy, self.error = [], loaded_checks, busy, error

    def __call__(self, args, **kwargs):
        verb = args[1]
        self.calls.append(verb)
        if verb == "print":
            self.loaded_checks -= 1
            return subprocess.CompletedProcess(args, 0 if self.loaded_checks >= 0 else 113, "", "")
        if verb == "bootstrap":
            if self.error is not None:
                return subprocess.CompletedProcess(args, 1, "", self.error)
            if self.busy > 0:
                self.busy -= 1
                return subprocess.CompletedProcess(
                    args, 5, "", "Bootstrap failed: 5: Input/output error\nTry re-running the command as root for richer errors.\n"
                )
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def launchctl(monkeypatch, tmp_path):
    def make(**kwargs):
        fake = FakeLaunchctl(**kwargs)
        plist = tmp_path / "LaunchAgents" / f"{LABEL}.plist"
        plist.parent.mkdir()
        plist.write_bytes(b"old")  # a service is installed already, as when switching models
        monkeypatch.setattr(service.subprocess, "run", fake)
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        monkeypatch.setattr(service, "plist_path", lambda: plist)
        monkeypatch.setattr(service, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(service, "_executable", lambda: "/opt/aris/bin/aris-server")
        return fake

    return make


def test_install_waits_for_the_old_service_to_unload_before_loading_the_new_one(launchctl):
    fake = launchctl(loaded_checks=3)
    assert service.install(["--model", "inkling-2-mini"]) == 0
    assert fake.calls == ["bootout", "print", "print", "print", "print", "bootstrap"]


def test_install_loads_the_service_again_while_launchd_still_refuses_it(launchctl):
    fake = launchctl(busy=2)
    assert service.install(["--model", "inkling-2-mini"]) == 0
    assert fake.calls.count("bootstrap") == 3


def test_install_gives_up_after_ten_refusals(launchctl, capsys):
    fake = launchctl(busy=100)
    assert service.install(["--model", "inkling-2-mini"]) == 1
    assert fake.calls.count("bootstrap") == service.BOOTSTRAP_ATTEMPTS
    assert "Bootstrap failed: 5" in capsys.readouterr().err


def test_install_fails_at_once_on_any_other_launchctl_error(launchctl, capsys):
    fake = launchctl(error="Bootstrap failed: 122: Path had bad ownership/permissions\n")
    assert service.install(["--model", "inkling-2-mini"]) == 1
    assert fake.calls.count("bootstrap") == 1
    assert "bad ownership" in capsys.readouterr().err
