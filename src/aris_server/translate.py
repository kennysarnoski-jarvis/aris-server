"""Anthropic Messages request -> chat-template input and generation settings."""

import json
from dataclasses import dataclass, field

from .config import EFFORTS, ModelSpec
from .stream import PROGRESS_NOTE


class RequestError(ValueError):
    """The request is malformed; reported to the client as invalid_request_error."""


@dataclass
class Plan:
    messages: list[dict]
    tools: list[dict] | None
    template_kwargs: dict
    max_tokens: int
    stop: list[str]
    temperature: float
    top_p: float
    top_k: int
    think_budget: int
    thinking: bool
    effort: str
    hidden_tools: list[str] = field(default_factory=list)
    # Deferred tools loaded mid-conversation: described where they were loaded, not in `tools` (see convert_tools).
    loaded_tools: list[dict] = field(default_factory=list)
    context: int | None = None
    context_capped: bool = False  # max_tokens was cut to fit the context window


def parse_tools(p: "Plan") -> list[dict] | None:
    """Every tool the model may call: the listed ones plus deferred ones loaded so far."""
    return (p.tools or []) + p.loaded_tools or None


class PromptTooLong(RequestError):
    """Worded exactly as the Anthropic API words it: Claude Code matches the text to start compacting."""

    def __init__(self, tokens: int, maximum: int):
        super().__init__(f"prompt is too long: {tokens} tokens > {maximum} maximum")


MIN_OUTPUT_TOKENS = 1024  # a prompt that leaves less room than this is too long to answer usefully


def fit_context(p: "Plan", prompt_tokens: int) -> None:
    """Shrink max_tokens so prompt + output fits the context window, as the Anthropic API does."""
    if not p.context:
        return
    room = p.context - prompt_tokens
    if room < MIN_OUTPUT_TOKENS:
        raise PromptTooLong(prompt_tokens, p.context - MIN_OUTPUT_TOKENS)
    if p.max_tokens > room:
        p.max_tokens, p.context_capped = room, True


def _text(content, defs: dict | None = None) -> str:
    """Flatten a string or a list of content blocks to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "image":
            parts.append("[image omitted: this model cannot see images]")
        elif kind == "tool_reference":
            name = block.get("tool_name", "")
            spec = (defs or {}).get(name)
            parts.append(f"Tool {name} is now available." + (f" Its definition:\n{json.dumps(spec)}" if spec else ""))
        elif kind == "document":
            parts.append(_text(block.get("source", {}).get("content"), defs) or "[document omitted]")
    return "\n".join(p for p in parts if p)


def _loaded_tools(messages) -> set[str]:
    """Deferred tools that a tool result has referenced (Claude Code's tool search loads them this way)."""
    names = set()
    for m in messages:
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                    names.update(b.get("tool_name") for b in block["content"] if b.get("type") == "tool_reference")
    return names


def convert_messages(system, messages, defs: dict | None = None) -> list[dict]:
    out = []
    sys_text = _text(system)
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            # Claude Code sends mid-conversation system messages; most chat templates only allow a system
            # message first, so later ones become reminders inside the conversation.
            text = _text(content)
            if not out:
                out.append({"role": "system", "content": text})
            elif out[-1]["role"] == "system" and len(out) == 1:
                out[0]["content"] += "\n\n" + text
            elif text:
                out.append({"role": "user", "content": f"<system-reminder>\n{text}\n</system-reminder>"})
            continue
        if role not in ("user", "assistant"):
            raise RequestError(f"unsupported message role: {role!r}")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise RequestError("message content must be a string or a list of blocks")
        if role == "user":
            pending = []
            for block in content:
                if block.get("type") == "tool_result":
                    if pending:
                        out.append({"role": "user", "content": "\n".join(pending)}); pending = []
                    text = _text(block.get("content"), defs)
                    if block.get("is_error") and "error" not in text[:40].lower():
                        text = f"Error: {text}"
                    out.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": text})
                else:
                    t = _text([block])
                    if t:
                        pending.append(t)
            if pending:
                out.append({"role": "user", "content": "\n".join(pending)})
        else:
            msg = {"role": "assistant", "content": ""}
            reasoning, texts, calls = [], [], []
            for block in content:
                kind = block.get("type")
                if kind == "thinking":
                    thought = PROGRESS_NOTE.sub("", block.get("thinking", ""))
                    if thought.strip():
                        reasoning.append(thought)
                elif kind == "text":
                    texts.append(block.get("text", ""))
                elif kind == "tool_use":
                    calls.append({"id": block.get("id"), "type": "function",
                                  "function": {"name": block.get("name"), "arguments": block.get("input") or {}}})
            msg["content"] = "\n\n".join(t for t in texts if t)
            if reasoning:
                msg["reasoning_content"] = "\n\n".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def convert_tools(tools, messages) -> tuple[list[dict] | None, list[str], list[dict]]:
    """(listed tools, names of deferred tools not loaded, deferred tools loaded). Loaded deferred tools stay out of
    the listed ones: chat templates render the list at the top of the prompt, so a tool joining it mid-conversation
    changes the prompt's first tokens and the whole conversation is re-read (20k tokens, minutes, on a local model).
    Their definitions go where tool search loaded them instead."""
    if not tools:
        return None, [], []
    loaded = _loaded_tools(messages)
    out, hidden, late = [], [], []
    for t in tools:
        if "input_schema" not in t:  # server-side tools (web search etc.) can't run locally
            continue
        fn = {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t["input_schema"]}}
        if t.get("defer_loading"):
            if t["name"] in loaded:
                late.append(fn)
            else:
                hidden.append(t["name"])
            continue
        out.append(fn)
    return out or None, hidden, late


def plan(body: dict, spec: ModelSpec, default_max: int = 32000) -> Plan:
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise RequestError("messages must be a non-empty list")
    thinking_cfg = body.get("thinking") or {}
    thinking = thinking_cfg.get("type", "enabled") != "disabled"
    effort = (body.get("output_config") or {}).get("effort") or "high"
    if effort not in EFFORTS:
        raise RequestError(f"unknown effort {effort!r}")

    kwargs = dict(spec.template_kwargs)
    kwargs["enable_thinking"] = thinking
    if spec.effort_kwarg and thinking:
        kwargs[spec.effort_kwarg] = spec.effort_map.get(effort, effort)

    budget = spec.budget_for(effort) if thinking else 0
    if thinking and thinking_cfg.get("budget_tokens"):
        budget = int(thinking_cfg["budget_tokens"])
    elif budget and _continues_tool_loop(body["messages"]):
        budget = max(int(budget * spec.followup_budget), MIN_FOLLOWUP_BUDGET)

    max_tokens = int(body.get("max_tokens") or default_max)
    tools, hidden, late = convert_tools(body.get("tools"), body["messages"])
    messages = convert_messages(body.get("system"), body["messages"], {f["function"]["name"]: f for f in late})
    tools, messages = _apply_tool_choice(body.get("tool_choice"), tools, messages, late)
    return Plan(
        messages=messages,
        tools=tools,
        template_kwargs=kwargs,
        max_tokens=max_tokens,
        stop=list(body.get("stop_sequences") or []),
        temperature=float(body["temperature"]) if body.get("temperature") is not None else spec.temperature,
        top_p=float(body["top_p"]) if body.get("top_p") is not None else spec.top_p,
        top_k=int(body["top_k"]) if body.get("top_k") is not None else spec.top_k,
        think_budget=budget,
        thinking=thinking,
        effort=effort,
        hidden_tools=hidden,
        loaded_tools=late,
        context=spec.context,
    )


MIN_FOLLOWUP_BUDGET = 512


def _continues_tool_loop(messages) -> bool:
    """The request answers the model's own tool calls (the last message holds tool results and nothing the user
    typed), as opposed to starting work on something the user just asked."""
    last = messages[-1]
    if last.get("role") != "user" or not isinstance(last.get("content"), list):
        return False
    kinds = {b.get("type") for b in last["content"]}
    if "tool_result" not in kinds:
        return False
    typed = [b for b in last["content"] if b.get("type") == "text"
             and not b.get("text", "").lstrip().startswith("<system-reminder>")]
    return not typed


def _apply_tool_choice(choice, tools, messages, late=()):
    """Open models can't be forced to call a tool, so "any" and a named tool become an instruction (and a named
    tool narrows the list); "none" removes the tools."""
    kind = (choice or {}).get("type", "auto")
    if kind == "auto" or not tools:
        return tools, messages
    if kind == "none":
        return None, messages
    if kind == "tool":
        named = [t for t in [*tools, *late] if t["function"]["name"] == choice.get("name")]
        if not named:
            raise RequestError(f"tool_choice names unknown tool {choice.get('name')!r}")
        tools, note = named, f"Respond by calling the {choice['name']} tool."
    elif kind == "any":
        note = "Respond by calling one of the available tools."
    else:
        raise RequestError(f"unknown tool_choice type {kind!r}")
    return tools, messages + [{"role": "user", "content": f"<system-reminder>\n{note}\n</system-reminder>"}]
