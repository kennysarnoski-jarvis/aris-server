import pytest

from aris_server.config import ModelSpec
from aris_server.translate import RequestError, parse_tools, plan

ARETE = ModelSpec(name="arete:27b", path="x", context=131072, effort_kwarg="reasoning_effort",
                  effort_map={"high": "medium", "xhigh": "xhigh"})


def body(**kw):
    """A turn of Claude Code's agent loop, which always asks for thinking."""
    return {"model": "arete:27b", "max_tokens": 32000, "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive", "display": "omitted"}, **kw}


def test_tool_loop_history_becomes_template_messages():
    p = plan(body(
        system=[{"type": "text", "text": "You are an agent."}, {"type": "text", "text": "Be brief."}],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "list files"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "use Bash", "signature": "s"},
                {"type": "text", "text": "Listing."},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.rs"}]},
                {"type": "text", "text": "<system-reminder>note</system-reminder>"}]},
        ]), ARETE)
    assert p.messages == [
        {"role": "system", "content": "You are an agent.\nBe brief."},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "Listing.", "reasoning_content": "use Bash",
         "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a.rs"},
        {"role": "user", "content": "<system-reminder>note</system-reminder>"},
    ]


def test_tool_error_is_marked_and_images_are_described():
    p = plan(body(messages=[{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "is_error": True, "content": "exit 1"},
        {"type": "image", "source": {"type": "base64", "data": ""}}]}]), ARETE)
    assert p.messages[0] == {"role": "tool", "tool_call_id": "t", "content": "Error: exit 1"}
    assert "image omitted" in p.messages[1]["content"]


def test_deferred_tools_stay_hidden_until_tool_search_loads_them():
    tools = [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}},
             {"name": "Monitor", "description": "watch", "input_schema": {"type": "object"}, "defer_loading": True},
             {"name": "Cron", "description": "cron", "input_schema": {"type": "object"}, "defer_loading": True},
             {"type": "web_search_20250305", "name": "web_search"}]
    before = plan(body(tools=tools), ARETE)
    assert [t["function"]["name"] for t in before.tools] == ["Bash"]
    assert before.hidden_tools == ["Monitor", "Cron"]
    after = plan(body(tools=tools, messages=[
        {"role": "user", "content": "watch it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "ToolSearch", "input": {"query": "monitor"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                      "content": [{"type": "tool_reference", "tool_name": "Monitor"}]}]}]), ARETE)
    # The listed tools never change mid-conversation (they head the prompt, so a change re-reads all of it); a
    # loaded tool's definition goes where it was loaded, and the parser still knows it.
    assert [t["function"]["name"] for t in after.tools] == ["Bash"]
    assert after.messages[-1]["content"].startswith("Tool Monitor is now available. Its definition:\n")
    assert '"name": "Monitor"' in after.messages[-1]["content"]
    assert [t["function"]["name"] for t in parse_tools(after)] == ["Bash", "Monitor"]


@pytest.mark.parametrize("req,kwargs,budget", [
    ({"output_config": {"effort": "high"}}, {"enable_thinking": True, "reasoning_effort": "medium"}, 8192),
    ({"output_config": {"effort": "xhigh"}}, {"enable_thinking": True, "reasoning_effort": "xhigh"}, 16384),
    ({"output_config": {"effort": "low"}}, {"enable_thinking": True, "reasoning_effort": "low"}, 1024),
    ({"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}, {"enable_thinking": False}, 0),
    ({"thinking": {"type": "enabled", "budget_tokens": 2000}}, {"enable_thinking": True, "reasoning_effort": "medium"}, 2000),
    ({"output_config": {"effort": "max"}}, {"enable_thinking": True, "reasoning_effort": "max"}, 0),
])
def test_effort_and_thinking(req, kwargs, budget):
    p = plan(body(**req), ARETE)
    assert p.template_kwargs == kwargs
    assert p.think_budget == budget


def test_a_request_that_does_not_ask_for_thinking_does_not_think():
    # Claude Code's side requests (here summarizing a page WebFetch read) carry an effort but no `thinking`; Arete
    # thought through each one for minutes.
    side = {"model": "arete:27b", "max_tokens": 32000, "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "Web page content:\n---\nExample Domain\n---\nWhat is the title?"}]}
    p = plan(side, ARETE)
    assert p.template_kwargs == {"enable_thinking": False}
    assert p.think_budget == 0


def test_model_without_effort_control_gets_budgets_only():
    p = plan(body(output_config={"effort": "low"}), ModelSpec(name="m", path="m"))
    assert p.template_kwargs == {"enable_thinking": True}
    assert p.think_budget == 1024


def test_sampling_defaults_and_overrides():
    p = plan(body(), ARETE)
    assert (p.temperature, p.top_p, p.top_k) == (0.6, 0.95, 20)
    p = plan(body(temperature=0.2, top_p=0.5, stop_sequences=["END"]), ARETE)
    assert (p.temperature, p.top_p, p.stop) == (0.2, 0.5, ["END"])


def test_system_messages_inside_the_conversation():
    p = plan(body(system="base", messages=[
        {"role": "system", "content": "more base"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": [{"type": "text", "text": "mode changed"}]},
        {"role": "user", "content": "go"}]), ARETE)
    assert p.messages == [
        {"role": "system", "content": "base\n\nmore base"},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "<system-reminder>\nmode changed\n</system-reminder>"},
        {"role": "user", "content": "go"},
    ]


@pytest.mark.parametrize("bad", [{"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
                                 {"output_config": {"effort": "ultra"}}])
def test_bad_requests(bad):
    with pytest.raises(RequestError):
        plan(body(**bad), ARETE)


TOOLS = [{"name": n, "description": "", "input_schema": {"type": "object"}} for n in ("Bash", "Write")]


@pytest.mark.parametrize("choice,names,note", [
    (None, ["Bash", "Write"], None),
    ({"type": "auto"}, ["Bash", "Write"], None),
    ({"type": "none"}, None, None),
    ({"type": "any"}, ["Bash", "Write"], "calling one of the available tools"),
    ({"type": "tool", "name": "Write"}, ["Write"], "calling the Write tool"),
])
def test_tool_choice(choice, names, note):
    p = plan(body(tools=TOOLS, tool_choice=choice), ARETE)
    assert ([t["function"]["name"] for t in p.tools] if p.tools else None) == names
    assert (note in p.messages[-1]["content"]) if note else p.messages[-1] == {"role": "user", "content": "hi"}


def test_tool_choice_naming_an_unknown_tool_is_rejected():
    with pytest.raises(RequestError):
        plan(body(tools=TOOLS, tool_choice={"type": "tool", "name": "Nope"}), ARETE)


def test_output_is_cut_to_fit_the_context_and_overlong_prompts_get_the_api_s_wording():
    from aris_server.translate import PromptTooLong, fit_context

    p = plan(body(max_tokens=32000), ARETE)
    fit_context(p, 120000)
    assert (p.max_tokens, p.context_capped) == (131072 - 120000, True)
    p = plan(body(max_tokens=32000), ARETE)
    fit_context(p, 1000)
    assert (p.max_tokens, p.context_capped) == (32000, False)
    with pytest.raises(PromptTooLong, match=r"^prompt is too long: 130600 tokens > 130048 maximum$"):
        fit_context(plan(body(), ARETE), 130600)


def test_prefill_progress_lines_never_reach_the_model():
    p = plan(body(messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "⟨reading prompt 512/9000⟩\n⟨reading prompt 1024/9000⟩\nreal plan", "signature": "s"},
            {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "⟨reading prompt 512/9000⟩\n", "signature": "s"}, {"type": "text", "text": "ok"}]},
    ]), ARETE)
    assert p.messages[1]["reasoning_content"] == "real plan"
    assert "reasoning_content" not in p.messages[2]


@pytest.mark.parametrize("last,budget", [
    ({"role": "user", "content": "build it"}, 8192),
    ({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}, 4096),
    ({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                                  {"type": "text", "text": "<system-reminder>todo</system-reminder>"}]}, 4096),
    ({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                                  {"type": "text", "text": "actually use a BTreeMap"}]}, 8192),
])
def test_turns_that_continue_a_tool_loop_get_a_smaller_think_budget(last, budget):
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]}, last]
    assert plan(body(messages=msgs), ModelSpec(name="m", path="m")).think_budget == budget


def test_followup_budget_has_a_floor_and_explicit_budgets_win():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]
    assert plan(body(messages=msgs, output_config={"effort": "low"}), ModelSpec(name="m", path="m")).think_budget == 512
    explicit = plan(body(messages=msgs, thinking={"type": "enabled", "budget_tokens": 3000}), ModelSpec(name="m", path="m"))
    assert explicit.think_budget == 3000


def test_models_toml_reads_an_adapter(tmp_path, monkeypatch):
    from aris_server.config import load_models

    (tmp_path / "models.toml").write_text('[models."arete-trained:27b"]\npath = "~/models/arete"\nadapter = "~/adapters/expert1"\n')
    spec = load_models(tmp_path / "models.toml")["arete-trained:27b"]
    assert spec.adapter.endswith("/adapters/expert1") and not spec.adapter.startswith("~")


def test_claude_codes_per_request_billing_line_never_reaches_the_prompt():
    # Its suffix changes between requests mid-system-prompt, which would re-read everything after it.
    for suffix in ("5de", "eed"):
        p = plan(body(system=[{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.319.{suffix}; "
                                                       "cc_entrypoint=t3-code;"},
                              {"type": "text", "text": "You are Claude Code."}]), ARETE)
        assert p.messages[0] == {"role": "system", "content": "You are Claude Code."}
