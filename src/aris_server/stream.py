"""Turn the engine's classified text (reasoning / normal / tool) into Anthropic content blocks and SSE events."""

import json
import re
import uuid

from .inkling import MESSAGE, parse_tool_calls
from .tool_stream import FUNCTION, JsonToolCall, XmlToolCall

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}
PROGRESS_NOTE = re.compile(r"⟨reading prompt \d+/\d+⟩\n?")


def progress_note(done: int, total: int) -> str:
    return f"⟨reading prompt {done}/{total}⟩\n"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class MessageBuilder:
    """Feed (state, text) pieces in order; get back Anthropic stream events. `blocks` holds the final content."""

    def __init__(self, tool_parser=None, tools=None, exact=False):
        self.tool_parser, self.tools = tool_parser, tools
        # Thinking and text go to the client exactly as generated, not trimmed. Inkling's template writes a past
        # message's body back verbatim, so a body trimmed here (thinking that ends on the newline a think budget
        # closes it on, say) no longer matched the turn's cache, and the next turn read the whole reply again: 39 of
        # 59 cache rebuilds in a night of agent tasks on Inkling 2 Mini.
        self.exact = exact
        self.blocks: list[dict] = []
        self._open: str | None = None       # "thinking" | "text" while a block is open
        self._ws = ""                        # whitespace held back until real text arrives
        self._tool_text = ""
        self._xml: XmlToolCall | None = None  # set while an XML-style call streams
        self._json: JsonToolCall | None = None  # set while Inkling's JSON calls stream
        self._json_calls = tool_parser is parse_tool_calls
        self._buffer_tool = False             # this call's format can't stream: parse it whole when it ends
        self.tool_calls = 0

    @property
    def index(self) -> int:
        return len(self.blocks) - 1

    def _start(self, block: dict) -> list[dict]:
        events = self._close()
        self.blocks.append(block)
        shown = dict(block)
        if shown["type"] == "thinking":
            shown["thinking"], shown["signature"] = "", ""
        elif shown["type"] == "text":
            shown["text"] = ""
        events.append({"type": "content_block_start", "index": self.index, "content_block": shown})
        self._open = block["type"]
        return events

    def _close(self) -> list[dict]:
        if self._open is None:
            return []
        events = []
        if self._open == "thinking":
            # Clients echo thinking back with its signature; local thinking has nothing to verify.
            self.blocks[-1]["signature"] = "aris-local"
            events.append({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "signature_delta", "signature": "aris-local"}})
        events.append({"type": "content_block_stop", "index": self.index})
        self._open = None
        return events

    def feed(self, state: str, text: str) -> list[dict]:
        if state == MESSAGE:
            # A message of the model's own starts (an Inkling reply is a run of them). A block never runs across two:
            # a reply that thought twice in a row came back in the history as one thought, and missed its cache.
            # And a call ends with its message, so one cut short or malformed can't take the next one's text.
            events = self._flush_tool() if (self._tool_text or self._json) else []
            return events + (self._close() if self._open in ("thinking", "text") else [])
        events = []
        if state != "tool" and (self._tool_text or self._xml or self._json):
            events += self._flush_tool()
        if not text:
            return events
        if state == "tool":
            return events + self._feed_tool(text)
        kind = "thinking" if state == "reasoning" else "text"
        if self.exact:
            if self._open != kind:
                events += self._start({"type": kind, kind: ""})
            self.blocks[-1][kind] += text
            events.append({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "thinking_delta" if kind == "thinking" else "text_delta", kind: text}})
            return events
        text = self._ws + text
        if self._open != kind:
            text = text.lstrip()
        body = text.rstrip()
        # Trailing whitespace waits for more text, so blocks never end in blank lines.
        self._ws = text[len(body):]
        if not body:
            return events
        if self._open != kind:
            events += self._start({"type": kind, kind: ""})
        text = body
        self.blocks[-1][kind] += text
        events.append({"type": "content_block_delta", "index": self.index,
                       "delta": {"type": "thinking_delta" if kind == "thinking" else "text_delta", kind: text}})
        return events

    def note(self, text: str) -> list[dict]:
        """A line of thinking from the server itself (prefill progress): content the client counts as activity."""
        events = [] if self._open == "thinking" else self._start({"type": "thinking", "thinking": ""})
        self.blocks[-1]["thinking"] += text
        events.append({"type": "content_block_delta", "index": self.index,
                       "delta": {"type": "thinking_delta", "thinking": text}})
        return events

    def _feed_tool(self, text: str) -> list[dict]:
        if self._xml is not None:
            return self._apply(self._xml.feed(text))
        if self._json is not None:
            return self._feed_json(text)
        self._tool_text += text
        if self._buffer_tool:
            return []
        head = self._tool_text.lstrip()
        if head.startswith(FUNCTION):
            self._xml = XmlToolCall(self.tools)
            raw, self._tool_text = self._tool_text, ""
            return self._apply(self._xml.feed(raw))
        if self._json_calls and head.startswith("{"):
            self._json = JsonToolCall()
            raw, self._tool_text = self._tool_text, ""
            return self._feed_json(raw)
        if len(head) >= len(FUNCTION) or (head and not FUNCTION.startswith(head)):
            self._buffer_tool = True
        return []

    def _feed_json(self, text: str) -> list[dict]:
        events = self._apply(self._json.feed(text))
        if self._json.failed:  # not a call that streams: what it held is parsed whole when the calls end
            self._tool_text, self._buffer_tool, self._json = self._json.buf, True, None
        return events

    def _apply(self, ops) -> list[dict]:
        events = []
        for op, value in ops:
            if op == "start":
                events += self._start({"type": "tool_use", "id": new_id("toolu"), "name": value, "input": {}})
            elif op == "json":
                events.append({"type": "content_block_delta", "index": self.index,
                               "delta": {"type": "input_json_delta", "partial_json": value}})
            else:
                self.blocks[-1]["input"] = value
                events += self._close()
                self.tool_calls += 1
        return events

    def _flush_tool(self) -> list[dict]:
        if self._xml is not None:
            events = self._apply(self._xml.finish())
            self._xml = None
            return events
        if self._json is not None:
            call, self._json = self._json, None
            events = self._apply(call.finish())
            if call.failed and call.buf.strip():  # what never showed a call's head: parsed whole, as before
                self._tool_text = call.buf
                events += self._flush_tool()
            return events
        raw, self._tool_text, self._buffer_tool = self._tool_text, "", False
        calls = self._parse(raw)
        events = []
        for call in calls:
            block = {"type": "tool_use", "id": new_id("toolu"), "name": call["name"], "input": {}}
            events += self._start(block)
            block["input"] = call["arguments"]
            events.append({"type": "content_block_delta", "index": self.index,
                           "delta": {"type": "input_json_delta",
                                     "partial_json": json.dumps(call["arguments"], ensure_ascii=False)}})
            events += self._close()
            self.tool_calls += 1
        return events

    def _parse(self, raw: str) -> list[dict]:
        if self.tool_parser is None or not raw.strip():
            return []
        try:
            parsed = self.tool_parser(raw, self.tools)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return []
        parsed = parsed if isinstance(parsed, list) else [parsed]
        out = []
        for p in parsed:
            args = p.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if p.get("name"):
                out.append({"name": p["name"], "arguments": args if isinstance(args, dict) else {"value": args}})
        return out

    def finish(self, finish_reason: str | None, context_capped: bool = False) -> tuple[list[dict], str]:
        events = self._flush_tool() if (self._tool_text or self._xml or self._json) else []
        events += self._close()
        if self.tool_calls and finish_reason != "length":
            reason = "tool_use"
        elif finish_reason == "length" and context_capped:
            reason = "model_context_window_exceeded"
        else:
            reason = STOP_REASONS.get(finish_reason or "stop", "end_turn")
        return events, reason


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
