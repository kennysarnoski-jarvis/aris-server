import json
import random

import pytest
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

from aris_server.stream import MessageBuilder

TOOLS = [
    {"type": "function", "function": {"name": "Write", "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"}, "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}, "run_in_background": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"}, "edits": {"type": "array"}}}}},
]

RUST = '''pub fn satisfies(v: &str, r: &str) -> Result<bool, String> {
    // "quotes", back\\slashes, tabs\there, unicode ≥ ✓, and a fake tag: </param
    if r.is_empty() { return Err("empty range\\n".into()); }

    Ok(true)
}
'''

CALLS = [
    "<function=Write>\n<parameter=file_path>\n/tmp/p/src/lib.rs\n</parameter>\n<parameter=content>\n" + RUST
    + "\n</parameter>\n</function>",
    "<function=Bash>\n<parameter=command>\ncargo test 2>&1 | tail -20\n</parameter>\n<parameter=timeout>\n120\n"
    "</parameter>\n<parameter=run_in_background>\nfalse\n</parameter>\n</function>",
    "<function=Edit>\n<parameter=file_path>\na.rs\n</parameter>\n<parameter=edits>\n"
    '[{"old_string": "a", "new_string": "b"}]\n</parameter>\n</function>',
    "<function=Bash>\n</function>",
    "\n<function=Write>\n<parameter=content>\n\n\nblank lines around\n\n\n</parameter>\n</function>\n",
]


def stream(call, chunks):
    b = MessageBuilder(parse_tool_call, TOOLS)
    events, pos = [], 0
    for size in chunks:
        events += b.feed("tool", call[pos:pos + size])
        pos += size
    events += b.feed("tool", call[pos:])
    events += b.feed("normal", "")
    tail, reason = b.finish("stop")
    return b, events + tail, reason


def fragments(events):
    return "".join(e["delta"]["partial_json"] for e in events
                   if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("seed", range(25))
def test_streamed_json_matches_the_buffered_parser_for_any_chunking(call, seed):
    rng = random.Random(seed)
    chunks = [rng.randint(1, 7) for _ in range(len(call))] if seed else [1] * len(call)
    b, events, reason = stream(call, chunks)
    expected = parse_tool_call(call, TOOLS)
    assert reason == "tool_use"
    assert [blk["type"] for blk in b.blocks] == ["tool_use"]
    assert b.blocks[0]["name"] == expected["name"]
    assert json.loads(fragments(events)) == expected["arguments"] == b.blocks[0]["input"]


def test_a_long_write_streams_its_content_while_it_is_generated():
    content = "line\n" * 2000
    call = f"<function=Write>\n<parameter=file_path>\nbig.txt\n</parameter>\n<parameter=content>\n{content}"
    b = MessageBuilder(parse_tool_call, TOOLS)
    deltas = []
    for i in range(0, len(call), 4):
        deltas += [e for e in b.feed("tool", call[i:i + 4]) if e["type"] == "content_block_delta"]
    assert b.blocks[0]["name"] == "Write"            # the block opened before the call finished
    assert len(deltas) > 1000                        # content went out piece by piece
    streamed = fragments(deltas)
    assert streamed.startswith('{"file_path": "big.txt", "content": "line\\nline')


def test_a_call_cut_off_by_max_tokens_still_closes_its_json():
    call = "<function=Write>\n<parameter=file_path>\nx.rs\n</parameter>\n<parameter=content>\nfn main() {\n    print"
    b = MessageBuilder(parse_tool_call, TOOLS)
    events = b.feed("tool", call)
    tail, reason = b.finish("length")
    assert reason == "max_tokens"
    assert json.loads(fragments(events + tail)) == {"file_path": "x.rs", "content": "fn main() {\n    print"}


def test_non_xml_tool_formats_are_still_parsed_whole():
    def json_parser(text, tools):
        return json.loads(text)

    b = MessageBuilder(json_parser, TOOLS)
    events = b.feed("tool", '{"name": "Bash", ') + b.feed("tool", '"arguments": {"command": "ls"}}')
    assert events == []
    tail, reason = b.finish("stop")
    assert reason == "tool_use" and b.blocks[0]["input"] == {"command": "ls"}


# Inkling's JSON tool calls (JsonToolCall): one per message, a newline after each, as its ReplyStream hands them over.
INKLING_CALLS = [
    '{"name":"Bash","args":{"command":"cargo test 2>&1 | tail -20","timeout":120}}',
    '{"name":"Write","args":{"content":' + json.dumps(RUST, ensure_ascii=False) + ',"file_path":"/tmp/p/src/lib.rs"}}',
    '{"name":"Edit","args":{"file_path":"a.rs","old_string":"if a { b }","new_string":"if a {\\n    \\"b\\" \\\\ }"}}',
    '{"name":"TodoWrite","args":{"todos":[{"content":"x","status":"pending"},{"content":"{y}]","status":"done"}]}}',
    ' { "name" : "Bash" , "args" : { "command" : "ls" } } ',
    '{"name":"Bash","arguments":{"command":"ls -la"}}',
    '{"name":"Read","args":{}}',
    '{"name":"Write","args":{"content":"x\\"}{\\"y \\\\","file_path":"C:\\\\dir\\\\"}}',
]


def inkling_stream(text, chunks):
    from aris_server import inkling

    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    events, pos = [], 0
    for size in chunks:
        events += b.feed("tool", text[pos:pos + size])
        pos += size
    events += b.feed("tool", text[pos:])
    tail, reason = b.finish("stop")
    return b, events + tail, reason


@pytest.mark.parametrize("call", INKLING_CALLS)
@pytest.mark.parametrize("seed", range(25))
def test_inklings_streamed_calls_match_its_buffered_parser_for_any_chunking(call, seed):
    from aris_server import inkling

    text = call + "\n"
    rng = random.Random(seed)
    chunks = [rng.randint(1, 7) for _ in range(len(text))] if seed else [1] * len(text)
    b, events, reason = inkling_stream(text, chunks)
    expected = inkling.parse_tool_calls(text)
    assert reason == "tool_use"
    assert [(blk["type"], blk["name"], blk["input"]) for blk in b.blocks] == [
        ("tool_use", c["name"], c["arguments"]) for c in expected]
    assert json.loads(fragments(events)) == expected[0]["arguments"]


def test_inklings_calls_in_a_row_each_stream_as_their_own_block():
    from aris_server import inkling

    text = "".join(c + "\n" for c in INKLING_CALLS)
    b, events, _ = inkling_stream(text, [3] * (len(text) // 3))
    expected = inkling.parse_tool_calls(text)
    assert len(expected) == len(INKLING_CALLS)
    assert [(blk["name"], blk["input"]) for blk in b.blocks] == [(c["name"], c["arguments"]) for c in expected]
    starts = [e["index"] for e in events if e["type"] == "content_block_start"]
    stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
    assert starts == stops == list(range(len(INKLING_CALLS)))


def test_an_inkling_write_streams_its_content_while_it_is_generated():
    from aris_server import inkling

    content = "line\n" * 2000
    call = '{"name":"Write","args":{"file_path":"big.txt","content":' + json.dumps(content)
    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    deltas = []
    for i in range(0, len(call), 4):
        deltas += [e for e in b.feed("tool", call[i:i + 4]) if e["type"] == "content_block_delta"]
    assert b.blocks[0]["name"] == "Write"            # the block opened before the call finished
    assert len(deltas) > 1000                        # its content went out piece by piece
    assert fragments(deltas).startswith('{"file_path":"big.txt","content":"line\\nline')


@pytest.mark.parametrize("call,arguments", [
    ('{"args":{"command":"ls"},"name":"Bash"}', {"command": "ls"}),   # another key order
    ('{"name":"Bash","args":"ls"}', {}),                               # arguments that aren't an object
])
def test_an_inkling_call_that_cant_stream_is_parsed_whole_as_before(call, arguments):
    b, events, reason = inkling_stream(call + "\n", [5] * (len(call) // 5))
    assert reason == "tool_use"
    assert [(blk["name"], blk["input"]) for blk in b.blocks] == [("Bash", arguments)]
    assert json.loads(fragments(events)) == arguments


def test_an_inkling_call_cut_off_by_max_tokens_ends_where_it_stopped():
    from aris_server import inkling

    call = '{"name":"Write","args":{"file_path":"x.rs","content":"fn main() {\\n    print'
    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    events = b.feed("tool", call)
    tail, reason = b.finish("length")
    assert reason == "max_tokens"
    assert [(blk["name"], blk["input"]) for blk in b.blocks] == [("Write", {})]
    assert fragments(events + tail) == call[len('{"name":"Write","args":'):]  # as the Anthropic API leaves one


def test_an_inkling_call_with_another_key_before_its_arguments_is_parsed_whole():
    call = '{"name":"Bash","id":"call_1","args":{"command":"ls"}}'
    b, _, reason = inkling_stream(call + "\n", [4] * (len(call) // 4))
    assert reason == "tool_use" and [(blk["name"], blk["input"]) for blk in b.blocks] == [("Bash", {"command": "ls"})]


def test_a_malformed_inkling_call_ends_with_its_message_and_the_next_one_still_comes_through():
    from aris_server import inkling

    reply = ('<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"ls"}<|end_message|>'
             '<|message_model|>Read<|content_invoke_tool_json|>{"name":"Read","args":{"file_path":"a.rs"}}'
             "<|end_message|><|content_model_end_sampling|>")
    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    stream = inkling.ReplyStream()
    for i in range(0, len(reply), 3):
        for state, text in stream.feed(reply[i:i + 3]):
            b.feed(state, text)
    _, reason = b.finish("stop")
    assert reason == "tool_use"
    assert [(blk["name"], blk["input"]) for blk in b.blocks] == [("Bash", {"command": "ls"}), ("Read", {"file_path": "a.rs"})]


def test_an_inkling_call_cut_off_in_its_head_comes_to_nothing_as_before():
    from aris_server import inkling

    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    assert b.feed("tool", '{"name":"Ba') == []
    tail, reason = b.finish("length")
    assert reason == "max_tokens" and b.blocks == [] and tail == []


def test_text_after_a_malformed_inkling_call_ends_it_even_without_message_markers():
    from aris_server import inkling

    b = MessageBuilder(inkling.parse_tool_calls, TOOLS, exact=True)
    for state, text in [("tool", '{"name":"Bash","args":{"command":"ls"}\n'), ("normal", "Checking."),
                        ("tool", '{"name":"Read","args":{"file_path":"a.rs"}}\n')]:
        b.feed(state, text)
    b.finish("stop")
    assert [(blk["type"], blk.get("name")) for blk in b.blocks] == [("tool_use", "Bash"), ("text", None),
                                                                     ("tool_use", "Read")]
    assert [blk["input"] for blk in b.blocks if blk["type"] == "tool_use"] == [{"command": "ls"}, {"file_path": "a.rs"}]
