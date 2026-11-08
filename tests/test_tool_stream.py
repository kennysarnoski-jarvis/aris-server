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
