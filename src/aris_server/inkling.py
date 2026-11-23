"""Inkling 2's reply format: a sequence of messages, each a header and one kind of content.

    <|message_model|><|content_thinking|>...<|end_message|>                                 reasoning
    <|message_model|><|content_text|>...<|end_message|>                                     text
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>  a tool call
    <|content_model_end_sampling|>                                                          the end (EOS)

mlx-lm's state machine follows one start and one end marker per state, and its tool parsers look for one call
marker, so neither can read this. The model's own spec for it is the `response_template` in its
tokenizer_config.json, which this follows: thinking, text (also tool errors and plain-text tool calls) and JSON tool
calls, each closed by <|end_message|>.
"""

import json

HEADER = "<|message_model|>"
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
THINKING = "<|content_thinking|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
# Content markers, and the state of what follows each.
CONTENT = {
    THINKING: "reasoning",
    "<|content_text|>": "normal",
    "<|content_tool_error|>": "normal",
    "<|content_invoke_tool_text|>": "normal",
    TOOL_JSON: "tool",
}
# Markers that end a message or the reply: the loop guard never blocks them.
ENDING_TEXTS = (END, EOS)


def is_inkling(tokenizer) -> bool:
    """True for a tokenizer with Inkling's message markers as single tokens."""
    return all(_single_id(tokenizer, marker) is not None for marker in (HEADER, END, THINKING, TOOL_JSON))


# The headers that open a user, model or tool message. The prompt before the first of them is its system region: the
# tool declarations, the system prompt and the effort message.
CONVERSATION_HEADERS = ("<|message_user|>", HEADER, "<|message_tool|>")


def conversation_starts(tokenizer) -> frozenset[int]:
    """The ids of the headers that open a user, model or tool message: where a prompt's system region has ended."""
    return frozenset(i for i in (_single_id(tokenizer, h) for h in CONVERSATION_HEADERS) if i is not None)


def think_ids(tokenizer) -> tuple[int, int] | None:
    """The ids that open and close a thinking message, for the think budget; None for other models."""
    if not is_inkling(tokenizer):
        return None
    return _single_id(tokenizer, THINKING), _single_id(tokenizer, END)


def _single_id(tokenizer, text: str) -> int | None:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:  # noqa: BLE001  a tokenizer that can't encode it doesn't have it
        return None
    return ids[0] if len(ids) == 1 else None


class ReplyStream:
    """Feed a reply's text in order; get (state, text) pieces with every marker removed. States are "reasoning",
    "normal" and "tool". A message header's own text (the tool's name) is dropped: the call's JSON names it too.
    A marker split between two feeds waits for the rest."""

    def __init__(self):
        self.state = "normal"
        self._header = False
        self._held = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        buf, self._held = self._held + text, ""
        while buf:
            at = buf.find("<|")
            if at < 0:
                keep = 1 if buf.endswith("<") else 0  # it could start a marker
                self._emit(out, buf[:len(buf) - keep])
                self._held = buf[len(buf) - keep:]
                break
            self._emit(out, buf[:at])
            close = buf.find("|>", at + 2)
            if close < 0:
                self._held = buf[at:]
                break
            self._marker(buf[at:close + 2], out)
            buf = buf[close + 2:]
        return out

    def flush(self) -> list[tuple[str, str]]:
        """What a feed held back, as text: the reply ended in the middle of something that looked like a marker."""
        out: list[tuple[str, str]] = []
        self._emit(out, self._held)
        self._held = ""
        return out

    def discard(self) -> None:
        self._held = ""

    def _marker(self, marker: str, out: list) -> None:
        if marker == HEADER:
            self._header = True
        elif marker in CONTENT:
            self._header, self.state = False, CONTENT[marker]
        elif marker == END:
            if self.state == "tool":
                self._emit(out, "\n")  # calls in a row stay apart for the parser
            self._header, self.state = False, "normal"
        elif marker == EOS:
            pass
        else:
            self._emit(out, marker)  # not one of the format's markers: it's text

    def _emit(self, out: list, text: str) -> None:
        if not text or self._header:
            return
        if out and out[-1][0] == self.state:
            out[-1] = (self.state, out[-1][1] + text)
        else:
            out.append((self.state, text))


def parse_tool_calls(raw: str, tools=None) -> list[dict]:
    """Every call in the text of a run of JSON tool-call messages: [{"name", "arguments"}]. A call that doesn't
    parse ends the run; calls before it are kept."""
    calls, decoder, at = [], json.JSONDecoder(), 0
    while True:
        while at < len(raw) and raw[at].isspace():
            at += 1
        if at >= len(raw):
            break
        try:
            call, at = decoder.raw_decode(raw, at)
        except json.JSONDecodeError:
            if calls:
                break
            raise
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("an Inkling tool call names its tool")
        arguments = call.get("args", call.get("arguments", {}))
        calls.append({"name": call["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls
