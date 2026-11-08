import http.client
import json
import socket
import threading
import time

import pytest

from aris_server import http as ah
from aris_server.config import ModelSpec
from aris_server.engine import Piece, Run
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call


class FakeEngine:
    """Yields scripted pieces; `stopped` is set when the server stops the generation."""

    def __init__(self, script, delay=0.0):
        self.spec = ModelSpec(name="fake", path="fake")
        self.script, self.delay = script, delay
        self.stopped = threading.Event()

    def run(self, plan, on_progress=None):
        def pieces():
            n = 0
            for state, text in self.script:
                if self.stopped.is_set():
                    return
                time.sleep(self.delay)
                n += 1
                yield [Piece(state, text)], None, n
            while not self.stopped.is_set():  # a generation that would go on forever unless stopped
                time.sleep(self.delay)
                n += 1
                yield [Piece("tool", "x")], None, n

        return Run(prompt_tokens=10, cached_tokens=0, tool_parser=parse_tool_call, pieces=pieces(),
                   stop=self.stopped.set, budget=[])


@pytest.fixture
def serve():
    servers = []

    def start(engine):
        server = ah.Server(("127.0.0.1", 0), engine, {})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1]

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


BODY = {"model": "fake", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}


def test_long_tool_call_gets_heartbeat_pings(serve, monkeypatch):
    monkeypatch.setattr(ah, "HEARTBEAT_SECONDS", 0.02, raising=False)
    call = "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>"
    engine = FakeEngine([("tool", c) for c in call] + [("normal", "")], delay=0.01)
    engine.script.append(("normal", ""))
    port = serve(engine)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/messages", json.dumps({**BODY, "stream": True}), {"content-type": "application/json"})
    resp = conn.getresponse()
    events, buf = [], b""
    while not any(e["type"] == "content_block_stop" for e in events):
        buf += resp.read1(4096)
        *chunks, buf = buf.split(b"\n\n")
        events += [json.loads(c.split(b"data: ", 1)[1]) for c in chunks if b"data: " in c]
    conn.close()
    start = next(i for i, e in enumerate(events) if e["type"] == "message_start")
    tool = next(i for i, e in enumerate(events) if e["type"] == "content_block_start")
    assert sum(e["type"] == "ping" for e in events[start:tool]) >= 3
    assert events[tool]["content_block"]["name"] == "Bash"


def test_dropped_non_streaming_client_stops_generation(serve, monkeypatch):
    monkeypatch.setattr(ah, "DISCONNECT_CHECK_SECONDS", 0.01, raising=False)
    engine = FakeEngine([], delay=0.005)
    port = serve(engine)
    data = json.dumps(BODY).encode()
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\ncontent-type: application/json\r\n"
                 b"content-length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
    time.sleep(0.1)
    assert not engine.stopped.is_set()
    sock.close()
    assert engine.stopped.wait(5)


def test_dropped_stream_client_stops_generation_during_a_silent_tool_call(serve, monkeypatch):
    monkeypatch.setattr(ah, "HEARTBEAT_SECONDS", 0.01, raising=False)
    engine = FakeEngine([], delay=0.005)
    port = serve(engine)
    data = json.dumps({**BODY, "stream": True}).encode()
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\ncontent-type: application/json\r\n"
                 b"content-length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
    got = b""
    while b"message_start" not in got:
        got += sock.recv(65536)
    time.sleep(0.1)  # the generation is now inside a tool call that sends nothing
    assert not engine.stopped.is_set()
    sock.close()
    assert engine.stopped.wait(5)
