"""Guards against two ways a local model gets stuck in an agent loop, seen in hard agent tasks:

- a stuck tool call: the model makes the same call turn after turn, gets the same result, and learns nothing new
  (one run listed the same folder 20 turns in a row and never wrote a line of code). `stuck_call_repeats` finds it
  in the request; the server then adds a reminder before the model answers.
- looping output: a response that degenerates into one chunk repeated until the token cap (a test line written a
  thousand times, a run of zeros). `LoopGuard` steers the model off the loop, and gives up on a response that keeps
  falling back into it, so it ends the way the token cap would end it, just tens of thousands of tokens sooner.
"""

import json

import mlx.core as mx

STUCK_REPEATS = 3  # the same call with the same result this many times in a row gets a reminder

LOOP_SPAN = 400    # a loop is the last LOOP_SPAN generated tokens being one chunk repeated...
LOOP_CYCLES = 8    # ...at least this many times (so chunks of up to LOOP_SPAN / LOOP_CYCLES tokens)
LOOP_BREAKS = 3    # times the guard steers a response off a loop before it ends the response instead
# Generated tokens are read back from the GPU this many at a time. Reading each one as it comes would stall every
# decode step (the CPU couldn't queue the next step while the last finished); in batches they're long computed.
LOOP_READ_EVERY = 32

# Tokens the guard never blocks: they close what the model is doing (its thinking, a tool call, the turn), and
# blocking one could trap the model inside it.
ENDING_TEXTS = ("</think>", "</tool_call>", "<|im_end|>", "<|endoftext|>")


def stuck_call_note(repeats: int) -> str:
    return (f"You've made this exact tool call {repeats} times in a row and got the same result each time. Running "
            "it again won't tell you anything new: use the result you already have and take the next step.")


def _flat(content) -> str:
    """A tool result's content as text, for comparing two results exactly."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
                         else json.dumps(b, sort_keys=True) for b in content)
    return json.dumps(content, sort_keys=True)


def _typed(message) -> bool:
    """The user wrote something in this message besides tool results (Claude Code's reminders don't count)."""
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return any(isinstance(b, dict) and b.get("type") == "text"
               and not b.get("text", "").lstrip().startswith("<system-reminder>") for b in content or [])


def _step(assistant, user):
    """One step of the tool loop, as compared between turns: each call's name and input with its result. None when
    the pair isn't the model's tool calls answered by their results alone."""
    if assistant.get("role") != "assistant" or user.get("role") != "user":
        return None
    if not isinstance(assistant.get("content"), list) or not isinstance(user.get("content"), list) or _typed(user):
        return None
    calls = [b for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
    results = {b.get("tool_use_id"): b for b in user["content"] if isinstance(b, dict) and b.get("type") == "tool_result"}
    if not calls or any(c.get("id") not in results for c in calls):
        return None
    return tuple((c.get("name"), json.dumps(c.get("input") or {}, sort_keys=True), _flat(results[c["id"]].get("content")),
                  bool(results[c["id"]].get("is_error"))) for c in calls)


def stuck_call_repeats(messages) -> int:
    """How many times in a row the conversation's latest tool step was made: the same tool call(s), with the same
    input, answered with the same result(s). 0 unless the request answers the model's own tool calls."""
    turns = [m for m in messages or [] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    k, latest, repeats = len(turns) - 1, None, 0
    while k >= 1:
        step = _step(turns[k - 1], turns[k])
        if step is None or (latest is not None and step != latest):
            break
        latest, repeats, k = step, repeats + 1, k - 2
    return repeats


class LoopGuard:
    """Logits processor. Watches the tokens a response generates; once the last `span` of them are one chunk
    repeated at least `cycles` times, it blocks the token that would continue the chunk, so the model has to write
    something else. After `breaks` of those, the next loop sets `gave_up` instead, and the engine ends the response
    as the token cap would (there is no point generating thousands more tokens of the same chunk).

    A loop is found in O(span / cycles) per token: runs[p] counts how many of the latest tokens each equal the token
    p places before them, so the last `span` tokens repeat with period p once runs[p] >= span - p. Tokens are read
    `read_every` at a time (see LOOP_READ_EVERY), so a loop is caught up to that many tokens late."""

    def __init__(self, exempt: frozenset[int] = frozenset(), span: int = LOOP_SPAN, cycles: int = LOOP_CYCLES,
                 breaks: int = LOOP_BREAKS, read_every: int = LOOP_READ_EVERY):
        self.span, self.allowed, self.exempt, self.read_every = span, breaks, exempt, read_every
        self.max_period = max(1, span // cycles)
        self.recent: list[int] = []
        self.pending: list = []
        self.runs = [0] * (self.max_period + 1)
        self.breaks = 0
        self.gave_up = False

    def _period(self) -> int | None:
        for p in range(1, self.max_period + 1):
            if self.runs[p] >= self.span - p:
                return p
        return None

    def _push(self, token: int) -> None:
        recent = self.recent
        recent.append(token)
        if len(recent) > self.max_period + 1:
            del recent[0]
        runs = self.runs
        for p in range(1, len(recent)):
            runs[p] = runs[p] + 1 if recent[-1 - p] == token else 0

    def __call__(self, tokens, logits):
        if self.gave_up or tokens is None or len(tokens) == 0:
            return logits
        self.pending.append(tokens[-1])
        if len(self.pending) < self.read_every:
            return logits
        for token in mx.concatenate([t.reshape(-1) for t in self.pending]).tolist():
            self._push(int(token))
        self.pending.clear()
        p = self._period()
        if p is None:
            return logits
        self.runs = [0] * (self.max_period + 1)
        if self.breaks >= self.allowed:
            self.gave_up = True
            return logits
        self.breaks += 1
        continuing = self.recent[-p]  # what the chunk would put next
        if continuing in self.exempt:
            return logits
        return mx.where(mx.arange(logits.shape[-1]) == continuing, mx.array(-mx.inf, logits.dtype), logits)


def loop_exempt_ids(tokenizer) -> frozenset[int]:
    """The tokenizer's ids for the tokens `LoopGuard` never blocks: end of turn, of thinking, of a tool call."""
    ids = set(getattr(tokenizer, "eos_token_ids", None) or [])
    texts = set(ENDING_TEXTS)
    for attr in ("think_end", "tool_call_end"):
        try:
            if isinstance(text := getattr(tokenizer, attr, None), str):
                texts.add(text)
        except Exception:  # some tokenizer wrappers raise for a feature the model doesn't have
            pass
    for text in texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            continue
        if len(encoded) == 1:
            ids.add(encoded[0])
    return frozenset(ids)
