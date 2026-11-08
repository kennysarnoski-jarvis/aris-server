"""Turn the engine's classified text (reasoning / normal / tool) into Anthropic content blocks and SSE events."""

import json
import uuid

from .tool_stream import FUNCTION, XmlToolCall

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class MessageBuilder:
    """Feed (state, text) pieces in order; get back Anthropic stream events. `blocks` holds the final content."""

    def __init__(self, tool_parser=None, tools=None):
        self.tool_parser, self.tools = tool_parser, tools
        self.blocks: list[dict] = []
        self._open: str | None = None       # "thinking" | "text" while a block is open
        self._ws = ""                        # whitespace held back until real text arrives
        self._tool_text = ""
        self._xml: XmlToolCall | None = None  # set while an XML-style call streams
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
        events = []
        if state != "tool" and (self._tool_text or self._xml):
            events += self._flush_tool()
        if not text:
            return events
        if state == "tool":
            return events + self._feed_tool(text)
        kind = "thinking" if state == "reasoning" else "text"
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

    def _feed_tool(self, text: str) -> list[dict]:
        if self._xml is not None:
            return self._apply(self._xml.feed(text))
        self._tool_text += text
        if self._buffer_tool:
            return []
        head = self._tool_text.lstrip()
        if head.startswith(FUNCTION):
            self._xml = XmlToolCall(self.tools)
            raw, self._tool_text = self._tool_text, ""
            return self._apply(self._xml.feed(raw))
        if len(head) >= len(FUNCTION) or (head and not FUNCTION.startswith(head)):
            self._buffer_tool = True
        return []

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
        events = self._flush_tool() if (self._tool_text or self._xml) else []
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
