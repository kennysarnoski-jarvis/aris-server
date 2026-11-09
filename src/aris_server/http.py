"""HTTP front: the Anthropic Messages API over a threading HTTP server."""

import hmac
import json
import logging
import os
import select
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .config import ModelSpec, resolve
from .engine import Engine
from .stream import MessageBuilder, new_id, progress_note, sse
from .translate import RequestError, fit_context, plan


# Claude Code drops a stream that sends nothing for a few minutes, and a tool call is only sent once it closes, so a
# long Write could go silent; a ping goes out whenever nothing else has for this long.
HEARTBEAT_SECONDS = 5.0
# Claude Code doesn't count pings as activity either, and reading a long prompt can take longer than it waits, so a
# slow prefill reports its progress as a line of thinking (stripped again when the conversation comes back).
PROGRESS_SECONDS = 30.0
DISCONNECT_CHECK_SECONDS = 2.0


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine: Engine, models: dict[str, ModelSpec], api_key: str | None = None):
        super().__init__(address, Handler)
        self.engine, self.models, self.api_key = engine, models, api_key


class Handler(BaseHTTPRequestHandler):
    server: Server
    server_version = f"aris-server/{__version__}"

    def log_message(self, fmt, *args):
        logging.debug("%s %s", self.address_string(), fmt % args)

    # --- plumbing -------------------------------------------------------------------------------------------

    def _json(self, status: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("request-id", new_id("req"))
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, kind: str, message: str):
        self._json(status, {"type": "error", "error": {"type": kind, "message": message}})

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        given = self.headers.get("x-api-key") or self.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(given.encode(), key.encode())

    def _body(self) -> dict | None:
        try:
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._error(400, "invalid_request_error", "body must be JSON")
            return None
        if not isinstance(body, dict):
            self._error(400, "invalid_request_error", "body must be a JSON object")
            return None
        return body

    def _client_gone(self) -> bool:
        """True once the client has closed its end. A request that is still generating writes nothing (non-streaming,
        or a tool call in progress), so a dropped client would otherwise keep its generation running."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    @property
    def route(self) -> str:
        return urlsplit(self.path).path.rstrip("/")

    # --- routes ---------------------------------------------------------------------------------------------

    def do_GET(self):
        if self.route == "/health":
            return self._json(200, {"status": "ok", "model": self.server.engine.spec.name})
        if not self._authorized():
            return self._error(401, "authentication_error", "invalid x-api-key")
        if self.route == "/v1/models":
            names = [self.server.engine.spec.name] + [n for n in self.server.models if n != self.server.engine.spec.name]
            data = [{"type": "model", "id": n, "display_name": n, "created_at": "1970-01-01T00:00:00Z"} for n in names]
            return self._json(200, {"data": data, "has_more": False, "first_id": names[0], "last_id": names[-1]})
        self._error(404, "not_found_error", f"no route {self.route}")

    def do_HEAD(self):
        self.send_response(200 if self.route in ("", "/health") else 404)
        self.end_headers()

    def do_POST(self):
        if not self._authorized():
            return self._error(401, "authentication_error", "invalid x-api-key")
        if self.route not in ("/v1/messages", "/v1/messages/count_tokens"):
            return self._error(404, "not_found_error", f"no route {self.route}")
        body = self._body()
        if body is None:
            return
        if dump := os.environ.get("ARIS_SERVER_DUMP"):
            path = Path(dump) / f"{time.strftime('%Y%m%d-%H%M%S')}-{new_id('req')}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body, indent=1, ensure_ascii=False))
        engine = self.server.engine
        name = body.get("model") or engine.spec.name
        if name != engine.spec.name and name in self.server.models:
            return self._error(400, "invalid_request_error",
                               f"model {name!r} is configured but not loaded (loaded: {engine.spec.name!r})")
        spec = engine.spec if name not in self.server.models else resolve(name, self.server.models)
        try:
            p = plan(body, spec)
        except RequestError as e:
            return self._error(400, "invalid_request_error", str(e))
        if self.route.endswith("count_tokens"):
            return self._json(200, {"input_tokens": engine.count_tokens(p)})
        try:
            fit_context(p, engine.count_tokens(p))
        except RequestError as e:
            return self._error(400, "invalid_request_error", str(e))
        if body.get("stream"):
            self._stream(name, p)
        else:
            self._complete(name, p)

    # --- generation -----------------------------------------------------------------------------------------

    def _complete(self, name, p):
        t0 = time.time()
        run = self.server.engine.run(p)
        builder = MessageBuilder(run.tool_parser, p.tools)
        finish, n, checked = None, 0, time.time()
        for pieces, finish, n in run.pieces:
            for piece in pieces:
                builder.feed(piece.state, piece.text)
            if time.time() - checked > DISCONNECT_CHECK_SECONDS:
                checked = time.time()
                if self._client_gone():
                    run.stop()
                    logging.info("client disconnected after %d tokens; generation stopped", n)
                    return
        _, reason = builder.finish(finish, p.context_capped)
        msg = _message(name, builder.blocks, reason, run, n)
        self._json(200, msg)
        _log(name, p, run, n, builder, reason, t0)

    def _stream(self, name, p):
        t0 = time.time()
        self.send_response(200)
        self.send_header("request-id", new_id("req"))
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.flush()

        last_write = [time.time()]

        def write(events):
            for e in events:
                self.wfile.write(sse(e))
            self.wfile.flush()
            last_write[0] = time.time()

        holder: list[MessageBuilder] = []
        last_note = [time.time()]

        def progress(done, total):
            logging.debug("prompt %d/%d", done, total)
            if holder and time.time() - last_note[0] > PROGRESS_SECONDS:
                last_note[0] = time.time()
                write(holder[0].note(progress_note(done, total)))
            else:
                write([{"type": "ping"}])

        run = None
        try:
            write([{"type": "ping"}])
            run = self.server.engine.run(p, on_progress=progress)
            builder = MessageBuilder(run.tool_parser, p.tools)
            holder.append(builder)
            start = _message(name, [], None, run, 0)
            write([{"type": "message_start", "message": start}])
            finish, n = None, 0
            for pieces, finish, n in run.pieces:
                events = []
                for piece in pieces:
                    events += builder.feed(piece.state, piece.text)
                if not events and time.time() - last_write[0] > HEARTBEAT_SECONDS:
                    events = [{"type": "ping"}]
                if events:
                    write(events)
            events, reason = builder.finish(finish, p.context_capped)
            events.append({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": None},
                           "usage": _usage(run, n)})
            events.append({"type": "message_stop"})
            write(events)
            _log(name, p, run, n, builder, reason, t0)
        except (BrokenPipeError, ConnectionResetError):
            if run is not None:
                run.stop()
            logging.info("client disconnected; generation stopped")
            return
        except Exception as e:  # report engine failures in-stream; the status line is already sent
            logging.exception("generation failed")
            try:
                write([{"type": "error", "error": {"type": "api_error", "message": str(e)}}])
            except OSError:
                pass


def _usage(run, n):
    return {"input_tokens": run.prompt_tokens - run.cached_tokens, "cache_read_input_tokens": run.cached_tokens,
            "cache_creation_input_tokens": 0, "output_tokens": n}


def _message(name, blocks, reason, run, n):
    return {"id": new_id("msg"), "type": "message", "role": "assistant", "model": name, "content": blocks,
            "stop_reason": reason, "stop_sequence": None, "usage": _usage(run, n)}


def _log(name, p, run, n, builder, reason, t0):
    dt = time.time() - t0
    budget = run.budget[0] if run.budget else None
    think = f" think={budget.n}/{p.think_budget}{' FORCED' if budget.forced else ''}" if budget else ""
    logging.info("%s effort=%s prompt=%d cached=%d out=%d%s tools=%d stop=%s %.1fs (%.1f tok/s) %s", name, p.effort,
                 run.prompt_tokens, run.cached_tokens, n, think, builder.tool_calls, reason, dt, n / dt if dt else 0,
                 Engine.memory())
