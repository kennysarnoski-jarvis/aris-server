"""Stream an XML-style tool call (Qwen / Qwen-coder: <function=NAME><parameter=KEY>value</parameter>...</function>)
as Anthropic `input_json_delta` fragments while it is generated.

Claude Code abandons a response that produces no content for several minutes, and a model writing a whole file in
one Write call at a few tokens a second takes that long. Like the Anthropic API, the call's name goes out as soon as
it is known and string arguments stream as they are written. Non-string arguments are converted with the model's own
parser rules once complete. The concatenated fragments always form the same JSON the buffered parser would produce.
"""

import json

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
