import json

from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

from aris_server.stream import MessageBuilder

TOOLS = [{"type": "function", "function": {"name": "Bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}}]


def run(pieces, finish="stop"):
    b = MessageBuilder(parse_tool_call, TOOLS)
    events = []
    for state, text in pieces:
        events += b.feed(state, text)
    tail, reason = b.finish(finish)
    return b, events + tail, reason


def test_thinking_then_text_then_tool_call():
    b, events, reason = run([
        ("reasoning", "Need to"), ("reasoning", " list.\n\n"), ("normal", "\n\n"), ("normal", "Listing."),
        ("normal", "\n"), ("tool", "<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
                                   "<parameter=timeout>\n30\n</parameter>\n</function>"), ("normal", ""),
    ])
    assert reason == "tool_use"
    assert [blk["type"] for blk in b.blocks] == ["thinking", "text", "tool_use"]
    assert b.blocks[0]["thinking"] == "Need to list."
    assert b.blocks[1]["text"] == "Listing."
    assert b.blocks[2]["name"] == "Bash" and b.blocks[2]["input"] == {"command": "ls -la", "timeout": 30}
    kinds = [e["type"] for e in events]
    assert kinds.count("content_block_start") == kinds.count("content_block_stop") == 3
    streamed = "".join(e["delta"]["partial_json"] for e in events
                       if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert json.loads(streamed) == {"command": "ls -la", "timeout": 30}
    sig = [e for e in events if e["type"] == "content_block_delta" and e["delta"]["type"] == "signature_delta"]
    assert len(sig) == 1 and sig[0]["index"] == 0


def test_indices_are_in_order_and_whitespace_only_text_makes_no_block():
    b, events, reason = run([("reasoning", "hmm"), ("normal", "\n\n"), ("normal", "  ")])
    assert reason == "end_turn"
    assert [blk["type"] for blk in b.blocks] == ["thinking"]
    assert all(e["index"] == 0 for e in events if "index" in e)


def test_paragraph_breaks_inside_text_survive():
    b, _, _ = run([("normal", "one"), ("normal", "\n\n"), ("normal", "two")])
    assert b.blocks[0]["text"] == "one\n\ntwo"


def test_length_stop_and_truncated_tool_call():
    b, _, reason = run([("reasoning", "thinking forever")], finish="length")
    assert reason == "max_tokens"
    b, _, reason = run([("tool", "<function=Bash>\n<parameter=command>\nls")], finish="length")
    assert reason == "max_tokens" and b.blocks[0]["input"] == {"command": "ls"}


def test_two_tool_calls_in_one_turn():
    call = "<function=Bash>\n<parameter=command>\n{}\n</parameter>\n</function>"
    b, _, reason = run([("tool", call.format("ls")), ("normal", "\n"), ("tool", call.format("pwd")), ("normal", "")])
    assert reason == "tool_use"
    assert [blk["input"]["command"] for blk in b.blocks] == ["ls", "pwd"]
    assert len({blk["id"] for blk in b.blocks}) == 2
