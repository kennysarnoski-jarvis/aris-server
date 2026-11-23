from pathlib import Path

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
