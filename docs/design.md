# Aris Server design

Aris Server runs open-weight models locally and serves them over the Anthropic Messages API, so
Claude Code (and Aris Code, and anything else that speaks `/v1/messages`) can use a local model as
if it were Claude. It replaces Ollama for that job, without Ollama's model zoo.

## Why it exists

Running a local model behind Claude Code through Ollama breaks in ways no setting fixes:

- **Effort never reaches the model.** Claude Code sends `output_config.effort`; Ollama drops it,
  so a model with trained effort levels always runs at its template default.
- **Silence until the tool call ends.** Ollama's `/v1/messages` sends nothing, not even headers,
  while a tool call is generated. Claude Code times out after a few minutes of silence and retries.
- **Someone else's template.** A GGUF carries whatever chat template its converter baked in. When
  that template rejects a system message or contradicts the Modelfile, the model breaks and the
  only fix is a different build.
- **No way to stop a model that won't stop thinking.** Reasoning models that overthink burn the
  whole `max_tokens` inside `<think>` and never answer.

## What it is

One process, one HTTP port (default `127.0.0.1:8484`), one model resident at a time.

```
Claude Code ──/v1/messages──▶ Anthropic layer ──▶ mlx-lm engine ──▶ Metal
                              (translate, effort,     (batching, prompt cache,
                               think budget, SSE)      tool and thinking parsing)
```

- **Engine:** mlx-lm's `ResponseGenerator` and `ModelProvider` (continuous batching, an LRU
  prompt cache that keeps the system-prompt segment across conversations, thinking detection and
  tool-call parsers for most open model families). mlx-lm is pinned to an exact version because
  these are internal APIs.
- **Anthropic layer (ours):** translates Messages requests into chat-template input, runs them on
  the engine, and streams Anthropic SSE events back.
- **Model settings (ours):** a TOML file maps a model name to its weights and its effort and
  thinking behavior.

## API

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/v1/messages` | Streaming and non-streaming. Text, thinking and `tool_use` blocks. |
| POST | `/v1/messages/count_tokens` | Exact count with the model's tokenizer. |
| GET | `/v1/models` | Configured models, Anthropic list shape. |
| GET | `/health` | Liveness and the loaded model. |

### Request translation

- `system` (string or text blocks) → one system message.
- `user` text → user message; `tool_result` blocks → `tool` messages, in order, their content
  flattened to text (images dropped with a note).
- `assistant` text and `tool_use` blocks → an assistant message with `tool_calls`; `thinking`
  blocks → `reasoning_content` so templates that keep reasoning in the current tool loop can.
- `tools` → OpenAI-style function schemas for the template. **Deferred tools**
  (`defer_loading: true`) are hidden from the model until a `tool_reference` to them appears in a
  tool result, which is how Claude Code's tool search loads them.
- `temperature`, `top_p`, `top_k`, `stop_sequences`, `max_tokens` → sampling and stop settings.
- Ignored: `cache_control`, `metadata`, `context_management`, `betas` (the prompt cache is
  automatic).

### Effort and thinking

| Request | Model sees |
| --- | --- |
| `thinking.type: "disabled"` | Thinking off (the template's `enable_thinking=False`). |
| `output_config.effort` low / medium / high / xhigh / max | The model's own effort control if it has one (for example Arete's `reasoning_effort`), else nothing, plus that effort's think budget. |
| `thinking.budget_tokens` | Overrides the think budget. |

**Think budget:** a logits processor counts thinking tokens. Past the budget it waits for the next
paragraph break outside a code fence and forces the end-of-thinking token, so the model closes its
reasoning on a complete thought and answers. Budgets per effort are set per model, with defaults.

### Streaming

Headers and `message_start` go out at once. While a long prompt is read, `ping` events go out
with progress, so clients never see a silent connection. Thinking streams as `thinking_delta`,
text as `text_delta`. A tool call streams as one `input_json_delta` when its closing tag arrives
(parsers need the whole call). `message_delta` carries `stop_reason` (`end_turn`, `tool_use`,
`max_tokens`, `stop_sequence`) and usage, including cached input tokens. A client that disconnects
stops its generation.

## Model settings

`~/.aris-server/models.toml`:

```toml
[models."arete:27b"]
path = "~/models/arete-27b-mlx-8bit"     # local MLX folder or a Hugging Face repo id
context = 131072
effort_kwarg = "reasoning_effort"         # template variable for effort, if the model has one
effort_map = { low = "low", medium = "medium", high = "xhigh", xhigh = "xhigh", max = "xhigh" }
think_budget = { low = 1024, medium = 4096, high = 8192, xhigh = 16384, max = 0 }  # 0 = no limit
```

Models without an entry still load by path or repo id with defaults (no effort kwarg, budgets
only).

## Command line

```
aris-server serve [--model NAME] [--host 127.0.0.1] [--port 8484]
aris-server models          # list configured models
```

## Not in scope (for now)

Model registry or downloads beyond what Hugging Face's cache does, GGUF, vision input, the
OpenAI API, multi-model residency, Windows and Linux (a llama.cpp backend behind the same API is
the plan; Ollama remains the fallback there).

## Security

Binds to loopback by default. Binding elsewhere requires `--host` and prints a warning; there is
no authentication beyond an optional `--api-key` that clients send as `x-api-key`.
