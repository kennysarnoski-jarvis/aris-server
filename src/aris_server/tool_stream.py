"""Stream a tool call as Anthropic `input_json_delta` fragments while it is generated: an XML-style one (Qwen /
Qwen-coder: <function=NAME><parameter=KEY>value</parameter>...</function>), or Inkling's JSON one (JsonToolCall).

Claude Code abandons a response that produces no content for several minutes, and a model writing a whole file in
one Write call at a few tokens a second takes that long. Like the Anthropic API, the call's name goes out as soon as
it is known and string arguments stream as they are written. Non-string arguments are converted with the model's own
parser rules once complete. The concatenated fragments always form the same JSON the buffered parser would produce.
"""

import json
import re

from mlx_lm.tool_parsers.qwen3_coder import _convert_param_value, _get_arguments_config, _string_types

FUNCTION = "<function="
PARAM = "<parameter="
END_PARAM = "</parameter>"
END_FUNCTION = "</function>"
HOLD = len("\n" + END_PARAM)  # held back so a value's closing newline and tag are never streamed as content


def _escape(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


class XmlToolCall:
    """Feed text from inside <tool_call>...</tool_call>; get back ("start", name), ("json", fragment) and
    ("end", input_dict) operations."""

    def __init__(self, tools):
        self.tools = tools
        self.buf = ""
        self.phase = "name"
        self.config: dict = {}
        self.input: dict = {}
        self.key = None
        self.streaming = False
        self.emitted = ""      # streamed part of the current string value
        self.lead_checked = False

    def _close_value(self, raw: str, ops: list):
        if raw.endswith("\n"):
            raw = raw[:-1]
        prefix = ("{" if not self.input else ", ") + json.dumps(self.key) + ": "
        if self.streaming:
            ops.append(("json", _escape(raw) + '"'))
            self.input[self.key] = self.emitted + raw
        else:
            try:
                value = _convert_param_value(raw, self.key, self.config)
            except (ValueError, TypeError):
                value = raw
            ops.append(("json", prefix + json.dumps(value, ensure_ascii=False)))
            self.input[self.key] = value

    def feed(self, text: str) -> list:
        self.buf += text
        ops: list = []
        while True:
            if self.phase == "name":
                i = self.buf.find(FUNCTION)
                j = self.buf.find(">", i) if i >= 0 else -1
                if j < 0:
                    break
                name = self.buf[i + len(FUNCTION):j].strip()
                self.buf = self.buf[j + 1:]
                self.config = _get_arguments_config(name, self.tools)
                self.input = {}
                ops.append(("start", name))
                self.phase = "between"
            elif self.phase == "between":
                s = self.buf.lstrip()
                if s.startswith(PARAM):
                    j = s.find(">")
                    if j < 0:
                        self.buf = s
                        break
                    self.key = s[len(PARAM):j].strip()
                    self.buf = s[j + 1:]
                    kind = str(self.config.get(self.key, {}).get("type", "")).strip().lower()
                    self.streaming = kind in _string_types
                    self.emitted, self.lead_checked = "", False
                    if self.streaming:
                        ops.append(("json", ("{" if not self.input else ", ") + json.dumps(self.key) + ': "'))
                    self.phase = "value"
                elif s.startswith(END_FUNCTION):
                    self.buf = s[len(END_FUNCTION):]
                    ops.append(("json", "}" if self.input else "{}"))
                    ops.append(("end", dict(self.input)))
                    self.phase = "name"  # another call may follow in the same block
                elif not s or PARAM.startswith(s) or END_FUNCTION.startswith(s):
                    self.buf = s
                    break
                else:  # stray text between parameters: skip to the next tag
                    k = s.find("<", 1)
                    self.buf = s[k:] if k > 0 else ""
                    if k < 0:
                        break
            else:  # value
                if not self.lead_checked:
                    if not self.buf:
                        break
                    if self.buf.startswith("\n"):
                        self.buf = self.buf[1:]
                    self.lead_checked = True
                k = self.buf.find(END_PARAM)
                if k >= 0:
                    self._close_value(self.buf[:k], ops)
                    self.buf = self.buf[k + len(END_PARAM):]
                    self.phase = "between"
                    continue
                if self.streaming and len(self.buf) > HOLD:
                    chunk, self.buf = self.buf[:-HOLD], self.buf[-HOLD:]
                    self.emitted += chunk
                    ops.append(("json", _escape(chunk)))
                break
        return ops

    @property
    def open(self) -> bool:
        return self.phase != "name"

    def finish(self) -> list:
        """The tool text ended without </function> (cut off by max_tokens): close the JSON that was started."""
        ops: list = []
        if self.phase == "value":
            self._close_value(self.buf, ops)
        if self.phase in ("between", "value"):
            ops.append(("json", "}" if self.input else "{}"))
            ops.append(("end", dict(self.input)))
        self.phase, self.buf = "name", ""
        return ops


# Inkling's JSON tool calls: the head of one, up to the first character of its arguments.
NAME_ARGS = re.compile(r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")\s*,\s*"(?:args|arguments)"\s*:\s*')
FIRST_KEY = re.compile(r'\s*\{\s*"((?:[^"\\]|\\.)*)"\s*:')
HEAD_LIMIT = 512  # a call's head is short: text that hasn't shown one by then isn't a call this streams


class JsonToolCall:
    """Feed the text of Inkling's JSON tool-call messages ({"name":NAME,"args":{...}}, a newline after each); get back
    ("start", name), ("json", fragment) and ("end", input_dict) operations, as XmlToolCall gives. The arguments
    stream as the model writes them, so the fragments joined are its arguments' JSON as written. Held back whole, a
    call that wrote a file sent nothing for as long as the file took: Inkling 2 Mini writing one at 20-odd tokens a
    second was cut off by Claude Code at 10 minutes, twice in a night of agent tasks, and the file was lost.
    `failed` turns true for text that isn't such a call (another key order, arguments that aren't an object); what
    it hadn't streamed is then `buf`, for the buffered parser."""

    def __init__(self):
        self.buf = ""
        self.phase = "head"  # head -> args -> tail -> head: calls follow one another
        self.failed = False
        self.args: list[str] = []
        self.depth = 0
        self.in_string = self.escaped = False

    def _scan(self, text: str) -> int | None:
        """Index just past where the value being scanned closes (its depth back to 0), or None: the state carries
        over to the next text."""
        for i, ch in enumerate(text):
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
            elif ch == '"':
                self.in_string = True
            elif ch in "{[":
                self.depth += 1
            elif ch in "}]":
                self.depth -= 1
                if self.depth == 0:
                    return i + 1
        return None

    def _input(self) -> dict:
        try:
            value = json.loads("".join(self.args))
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def feed(self, text: str) -> list:
        self.buf += text
        ops: list = []
        while not self.failed:
            if self.phase == "head":
                s = self.buf.lstrip()
                if not s:
                    self.buf = ""
                    break
                m = NAME_ARGS.match(s) if s.startswith("{") else None
                if m is None or m.end() == len(s):  # the arguments' first character decides: an object streams
                    first = FIRST_KEY.match(s)
                    self.buf = s
                    if not s.startswith("{") or len(s) > HEAD_LIMIT or (first and first.group(1) != "name"):
                        self.failed = True
                    break
                if s[m.end()] != "{":
                    self.buf, self.failed = s, True
                    break
                ops.append(("start", json.loads(m.group(1))))
                self.buf = s[m.end():]
                self.phase, self.args, self.depth, self.in_string, self.escaped = "args", [], 0, False, False
            elif self.phase == "args":
                end = self._scan(self.buf)
                chunk = self.buf if end is None else self.buf[:end]
                if chunk:
                    self.args.append(chunk)
                    ops.append(("json", chunk))
                if end is None:
                    self.buf = ""
                    break
                self.buf = self.buf[end:]
                ops.append(("end", self._input()))
                self.phase, self.depth, self.in_string, self.escaped = "tail", 1, False, False
            else:  # tail: the rest of the call's own object, up to its closing brace
                end = self._scan(self.buf)
                if end is None:
                    self.buf = ""
                    break
                self.buf = self.buf[end:]
                self.phase = "head"
        return ops

    @property
    def open(self) -> bool:
        return self.phase != "head" or bool(self.buf.strip())

    def finish(self) -> list:
        """The text ended inside a call (cut off by max_tokens): end the one whose arguments were streaming, with what
        parsed of them, as the Anthropic API leaves a call it cut off."""
        ops = [("end", self._input())] if self.phase == "args" else []
        if self.phase == "head" and self.buf.strip():
            self.failed = True  # text whose head never showed: the buffered parser decides what it was
        self.phase = "head"
        return ops
