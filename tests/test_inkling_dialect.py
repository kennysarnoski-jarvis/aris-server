"""Inkling 2's reply format through Aris Server: the reply stream, tool calls, the think budget, effort and sizing."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from aris_server import config, inkling
from aris_server.budget import ThinkBudget
from aris_server.engine import _classify_inkling, full_attention_layers
from aris_server.kv_quant import quantize_kv
from aris_server.stream import MessageBuilder
from aris_server.translate import convert_messages, plan

REPLY = ("<|message_model|><|content_thinking|>The user wants the weather.\nCall the tool.<|end_message|>"
         "<|message_model|><|content_text|>Checking now.<|end_message|>"
         '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Phoenix"}}'
         "<|end_message|>"
         '<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{"tz":"MST"}}<|end_message|>'
         "<|content_model_end_sampling|>")


def merged(pieces):
    out = []
    for state, text in pieces:
        if out and out[-1][0] == state:
            out[-1] = (state, out[-1][1] + text)
        elif text:
            out.append((state, text))
    return out


def read(chunks):
    stream = inkling.ReplyStream()
    pieces = [p for chunk in chunks for p in stream.feed(chunk)] + stream.flush()
    return merged(pieces)


def test_reply_stream_splits_messages_and_drops_markers():
    assert read([REPLY]) == [
        ("reasoning", "The user wants the weather.\nCall the tool."),
        ("normal", "Checking now."),
        ("tool", '{"name":"get_weather","args":{"city":"Phoenix"}}\n{"name":"get_time","args":{"tz":"MST"}}\n'),
    ]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13])
def test_reply_stream_is_the_same_whatever_the_chunks(size):
    assert read([REPLY[i:i + size] for i in range(0, len(REPLY), size)]) == read([REPLY])


def test_reply_stream_keeps_text_that_only_looks_like_a_marker():
    assert read(["<|message_model|><|content_text|>a <|b|> c < d<|end_message|>"]) == [("normal", "a <|b|> c < d")]
    # A reply cut off inside something marker-like still shows it.
    assert read(["<|message_model|><|content_text|>x <|conte"]) == [("normal", "x <|conte")]


def test_tool_calls_parse_in_a_row_and_keep_what_parsed():
    raw = read([REPLY])[-1][1]
    assert inkling.parse_tool_calls(raw) == [{"name": "get_weather", "arguments": {"city": "Phoenix"}},
                                            {"name": "get_time", "arguments": {"tz": "MST"}}]
    assert inkling.parse_tool_calls('{"name":"a","args":{}}\n{"name":"b","ar') == [{"name": "a", "arguments": {}}]
    with pytest.raises(ValueError):
        inkling.parse_tool_calls('{"args":{}}')
    with pytest.raises(json.JSONDecodeError):
        inkling.parse_tool_calls('{"name":')


def test_a_reply_becomes_anthropic_blocks():
    builder = MessageBuilder(tool_parser=inkling.parse_tool_calls)
    for state, text in read([REPLY[i:i + 5] for i in range(0, len(REPLY), 5)]):
        builder.feed(state, text)
    _, reason = builder.finish("stop")
    assert reason == "tool_use"
    assert [b["type"] for b in builder.blocks] == ["thinking", "text", "tool_use", "tool_use"]
    assert builder.blocks[0]["thinking"] == "The user wants the weather.\nCall the tool."
    assert [(b["name"], b["input"]) for b in builder.blocks[2:]] == [("get_weather", {"city": "Phoenix"}),
                                                                     ("get_time", {"tz": "MST"})]


def test_an_inkling_turn_comes_back_in_the_history_as_it_was_generated():
    # Inkling's template writes a past message's body back verbatim (<|content_thinking|>{reasoning}<|end_message|>),
    # so what the client echoes has to be what was generated. Trimmed, a turn whose thinking ended on the newline a
    # think budget closes it on no longer matched its own cache, and the next turn read the whole reply again.
    thinking, text = "\nPlan: list the files.\n", "Listing them.\n"
    reply = (f"<|message_model|><|content_thinking|>{thinking}<|end_message|>"
             f"<|message_model|><|content_text|>{text}<|end_message|>"
             '<|message_model|>Bash<|content_invoke_tool_json|>{"name":"Bash","args":{"command":"ls"}}<|end_message|>'
             "<|content_model_end_sampling|>")
    builder = MessageBuilder(tool_parser=inkling.parse_tool_calls, exact=True)
    events = []
    for state, piece in read([reply[i:i + 3] for i in range(0, len(reply), 3)]):
        events += builder.feed(state, piece)
    events += builder.finish("stop")[0]
    streamed = {}  # what a client puts together from the deltas, which is what it echoes
    for e in events:
        if e["type"] == "content_block_delta" and e["delta"]["type"] in ("thinking_delta", "text_delta"):
            key = "thinking" if e["delta"]["type"] == "thinking_delta" else "text"
            streamed[e["index"]] = streamed.get(e["index"], "") + e["delta"][key]
    assert streamed == {0: thinking, 1: text}
    assert builder.blocks[0]["thinking"] == thinking and builder.blocks[1]["text"] == text
    echoed = [{"role": "user", "content": "List the files."}, {"role": "assistant", "content": builder.blocks}]
    turn = convert_messages(None, echoed)[-1]
    assert turn["reasoning_content"] == thinking and turn["content"] == text


def test_classify_ends_on_stop_and_shows_a_cut_reply():
    gens = [SimpleNamespace(text=t, finish_reason=None) for t in ["<|message_model|><|content_text|>Hel", "lo"]]
    stop = [SimpleNamespace(text="<|content_model_end_sampling|>", finish_reason="stop")]
    out = [(p.state, p.text) for pieces, _, _ in _classify_inkling(None, gens + stop) for p in pieces]
    assert merged(out) == [("normal", "Hello")]
    cut = gens[:1] + [SimpleNamespace(text="lo <|con", finish_reason="length")]
    out = [(p.state, p.text) for pieces, _, _ in _classify_inkling(None, cut) for p in pieces]
    assert merged(out) == [("normal", "Hello <|con")]


class FakeTokenizer:
    def __init__(self, markers):
        self.ids = {m: 1000 + i for i, m in enumerate(markers)}

    def encode(self, text, add_special_tokens=False):
        return [self.ids[text]] if text in self.ids else [1, 2]


INKLING_MARKERS = [inkling.HEADER, inkling.END, inkling.THINKING, inkling.TOOL_JSON, inkling.EOS]


def test_inkling_is_known_by_its_markers():
    tok = FakeTokenizer(INKLING_MARKERS)
    assert inkling.is_inkling(tok)
    assert inkling.think_ids(tok) == (tok.ids[inkling.THINKING], tok.ids[inkling.END])
    assert not inkling.is_inkling(FakeTokenizer(["</think>"]))
    assert inkling.think_ids(FakeTokenizer(["</think>"])) is None


START, END, NL, WORD = 7, 8, 9, 10
VOCAB = 16


def step(budget, token):
    logits = mx.zeros((1, VOCAB))
    out = budget(mx.array([token]), logits)
    return int(mx.argmax(out[0]).item()) if (out != logits).any().item() else None


def test_think_budget_counts_only_inside_inkling_thinking():
    budget = ThinkBudget(3, END, NL, frozenset(), {}, start_id=START)
    for token in [WORD] * 10 + [END]:  # a text message: never counted, never closed
        assert step(budget, token) is None
    assert budget.n == 0 and not budget.forced
    step(budget, START)
    assert [step(budget, t) for t in [WORD, WORD, WORD, WORD]] == [None, None, None, None]
    assert step(budget, NL) == END and budget.forced  # past the budget, the next line closes thinking
    step(budget, END)
    assert step(budget, WORD) is None  # text after it is free
    step(budget, START)
    assert step(budget, NL) == END  # a later thinking message gets what's left of the budget: nothing


def test_think_budget_without_a_start_token_is_unchanged():
    budget = ThinkBudget(2, END, NL, frozenset(), {})
    assert [step(budget, t) for t in [WORD, WORD, WORD]] == [None, None, None]
    assert step(budget, NL) == END and budget.done


INKLING_TEXT = {"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128,
                "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
                "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3}}


def test_kv_layers_count_each_pass_through_a_loop():
    assert full_attention_layers(INKLING_TEXT) == 8  # layer 5, layers 11 and 17 three times each, layer 23
    assert full_attention_layers({**INKLING_TEXT, "loop_config": None}) == 4
    assert full_attention_layers({"num_hidden_layers": 4,
                                  "layer_types": ["linear_attention", "full_attention"] * 2}) == 2
    assert full_attention_layers({"num_hidden_layers": 64, "full_attention_interval": 4}) == 16
    assert full_attention_layers({"num_hidden_layers": 12}) == 12


def test_kv8_leaves_a_model_with_nested_caches_alone():
    model = SimpleNamespace(make_cache=lambda: [CacheList(RotatingKVCache(max_size=8), ArraysCache(4)),
                                                CacheList(KVCache(), ArraysCache(4))])
    before = model.make_cache
    assert quantize_kv(model, 8) is False and model.make_cache is before
    plain = SimpleNamespace(make_cache=lambda: [KVCache(), ArraysCache(2)])
    assert quantize_kv(plain, 8) is True


def test_effort_reaches_inklings_template_and_off_means_none():
    spec = config.builtin_models()["inkling-2-mini"]
    body = {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"}}
    assert plan(body, spec).template_kwargs["reasoning_effort"] == "max"
    assert plan({**body, "thinking": {"type": "disabled"}}, spec).template_kwargs["reasoning_effort"] == "none"
    aris = config.builtin_models()["aris-3.8-27b"]
    assert "reasoning_effort" not in plan({**body, "thinking": {"type": "disabled"}}, aris).template_kwargs


def test_claude_codes_efforts_up_to_its_default_run_inkling_at_low():
    # At medium and above Inkling reasoned through a hard task's spec for most of an hour before writing; at low it
    # scored the same on agent tasks in a fraction of the time. The think budget still follows the effort asked for.
    spec = config.builtin_models()["inkling-2-mini"]
    body = {"messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "adaptive"}}
    for asked, runs in (("low", "low"), ("medium", "low"), ("high", "low"), ("xhigh", "xhigh"), ("max", "max")):
        assert plan({**body, "output_config": {"effort": asked}}, spec).template_kwargs["reasoning_effort"] == runs
    assert plan({**body, "output_config": {"effort": "high"}}, spec).think_budget == 8192


def test_context_sizing_sees_the_caches_inside_cache_lists():
    from aris_server.engine import _flat_caches
    caches = [CacheList(RotatingKVCache(max_size=8), ArraysCache(4)), CacheList(KVCache(), ArraysCache(4)),
              CacheList(KVCache(), ArraysCache(4)), KVCache()]
    assert sum(isinstance(c, KVCache) for c in _flat_caches(caches)) == 3


def test_the_false_no_tools_warning_is_dropped_only_where_aris_server_parses_tool_calls(caplog):
    import logging

    from aris_server import engine

    message = "Received tools but model does not support tool calling. If you think this is an error, file an issue"
    was = engine._OWN_TOOL_CALLS.active
    try:
        with caplog.at_level(logging.WARNING):
            engine._OWN_TOOL_CALLS.active = True
            logging.warning(message)
            assert message not in caplog.text
            logging.warning("something else")
            assert "something else" in caplog.text
            engine._OWN_TOOL_CALLS.active = False
            logging.warning(message)
            assert message in caplog.text
    finally:
        engine._OWN_TOOL_CALLS.active = was


@pytest.mark.parametrize("is_inkling", [True, False])
def test_the_engine_asks_for_untrimmed_text_only_for_inkling(is_inkling):
    from aris_server.engine import Engine

    engine = Engine.__new__(Engine)  # run() with stand-ins for what it calls: no model needed
    engine.guards, engine.inkling = False, is_inkling
    engine.budget_tokens = SimpleNamespace(make=lambda budget: None)
    ctx = SimpleNamespace(prompt=[1, 2], prompt_cache_count=0, tool_parser=None, stop=lambda: None)
    engine.generator = SimpleNamespace(generate=lambda request, args, progress_callback=None: (ctx, iter(())))
    engine.provider, engine.cache = SimpleNamespace(model=object()), SimpleNamespace(reading=[])
    plan_ = SimpleNamespace(think_budget=0, temperature=1.0, top_p=1.0, top_k=0, stop=[], max_tokens=8,
                            template_kwargs={}, messages=[], tools=None)
    assert engine.run(plan_).exact_text is is_inkling
