from pathlib import Path

from aris_server.service import LABEL, build_plist


def test_the_service_restarts_the_server_after_any_exit():
    plist = build_plist("/opt/aris/bin/aris-server", ["--model", "arete:27b", "--port", "8484"], Path("/tmp/s.log"))
    assert plist["Label"] == LABEL
    assert plist["ProgramArguments"] == ["/opt/aris/bin/aris-server", "serve", "--model", "arete:27b", "--port", "8484"]
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True and plist["ThrottleInterval"] == 10
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == "/tmp/s.log"
