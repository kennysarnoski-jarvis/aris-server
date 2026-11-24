"""A cached conversation that its Claude Code session started over from (a compaction, a clear) is dropped, not saved
to disk on its way out of memory (MovingPromptCache.note_session), and only within one named session."""
import json
import logging

from aris_server.engine import MovingPromptCache
from aris_server.translate import claude_code_session


class Layer:
    def __init__(self, nbytes=1 << 20):
        self.nbytes = nbytes

    def is_trimmable(self):
        return False


class Disk:
    """What the cache asks of its disk tier, recording what it saves."""

    def __init__(self):
        self.index, self.saved = {}, []

    def has(self, tokens):
        return False

    def save(self, tokens, prompt_cache, cache_type):
        self.saved.append((len(tokens), cache_type))

    def nearest(self, tokens, longer_than=0):
        return None


MSG = 7                                    # the token that opens a message (Qwen's <|im_start|>)
SYSTEM = list(range(100, 160))             # the system prompt, through the first message's header
TASK = list(range(200, 240))               # the conversation's first message
TURN = SYSTEM + TASK + [MSG] + list(range(300, 400))   # its cached turn
SUMMARY = list(range(500, 530))            # the first message after a compaction: a summary in place of the task


def cache():
    c = MovingPromptCache(max_size=10, max_bytes=1 << 40)
    c.message_starts = frozenset({MSG})
    c.disk = Disk()
    return c


def tokenized(prompt, system=len(SYSTEM)):
    return prompt, [prompt[:system], prompt[system:]], ["system", "assistant"], "normal", None


def served(c, prompt, session, system=len(SYSTEM)):
    """A request through the engine's hook and its fetch; returns what it reused."""
    c.note_session("m", tokenized(prompt, system), session)
    layers, rest = c.fetch_nearest_cache("m", prompt)
    return len(prompt) - len(rest)


def keep_turn(c, prompt, session, turn=TURN):
    """A request whose finished turn is cached, as mlx-lm keeps it."""
    served(c, prompt, session)
    c.insert_cache("m", turn, [Layer()], cache_type="assistant")


def cached(c):
    return [list(k) for lru in c._lru._lrus.values() for _, k in lru]


def test_a_conversation_its_session_starts_over_from_is_dropped_not_saved(caplog):
    c = cache()
    keep_turn(c, TURN[:-30], "s1")
    with caplog.at_level(logging.INFO):
        served(c, SYSTEM + SUMMARY + [MSG, 1, 2], "s1")   # the request after a compaction
    assert TURN not in cached(c)
    assert "that its session started over from (a compaction or a clear)" in caplog.text
    c.trim_to(n_bytes=0)                                  # memory lets everything go: nothing of it is written
    assert (len(TURN), "assistant") not in c.disk.saved


def test_another_sessions_conversation_is_left_alone_and_saved_when_memory_lets_it_go():
    c = cache()
    keep_turn(c, TURN[:-30], "s1")
    served(c, SYSTEM + SUMMARY + [MSG, 1, 2], "s2")       # another session with the same system prompt
    assert TURN in cached(c)
    c.trim_to(n_bytes=0)
    assert (len(TURN), "assistant") in c.disk.saved


def test_a_client_that_names_no_session_ends_nothing():
    c = cache()
    keep_turn(c, TURN[:-30], None)
    served(c, SYSTEM + SUMMARY + [MSG, 1, 2], None)
    assert TURN in cached(c)


def test_a_subagent_with_its_own_system_prompt_ends_nothing():
    c = cache()
    keep_turn(c, TURN[:-30], "s1")
    own = SYSTEM[:30] + list(range(600, 630))             # the same session, another system prompt
    served(c, own + SUMMARY + [MSG, 1, 2], "s1")
    assert TURN in cached(c)


def test_a_fork_past_the_first_message_ends_nothing():
    c = cache()
    keep_turn(c, TURN[:-30], "s1")
    fork = TURN[:-50] + [MSG, 9, 9, 9]                    # a compaction's own request: the history, then its ask
    served(c, fork, "s1")
    assert TURN in cached(c)


def test_the_session_comes_from_claude_codes_header_or_metadata():
    body = {"metadata": {"user_id": json.dumps({"device_id": "d", "account_uuid": "", "session_id": "abc"})}}
    assert claude_code_session({"x-claude-code-session-id": "hdr"}, body) == "hdr"
    assert claude_code_session({}, body) == "abc"
    assert claude_code_session({}, {"metadata": {"user_id": "user_1f_account__session_def"}}) == "def"
    assert claude_code_session({}, {"metadata": {"user_id": "someone@example.com"}}) is None  # a user, not a session
    assert claude_code_session({}, {"metadata": {"user_id": json.dumps({"device_id": "d"})}}) is None
    assert claude_code_session({}, {}) is None and claude_code_session(None, {"metadata": "x"}) is None
