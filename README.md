# Aris Server

Run open-weight models on your Mac and use them from Claude Code, Aris Code, or anything else that
speaks the Anthropic Messages API.

```bash
uv tool install git+https://github.com/kennysarnoski-jarvis/aris-server
aris-server serve --model mlx-community/Qwen3.5-9B-MLX-8bit
```

Then point Claude Code at it:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_AUTH_TOKEN=local claude
```

## Why not Ollama?

Aris Server does one job: serve a local model to coding agents properly.

- **Native `/v1/messages`.** Thinking, tool calls and text stream as real Anthropic events from
  the first byte, with pings while a long prompt is read or a long tool call is written, so clients
  never time out on silence, and a client that hangs up stops its generation.
- **Effort that works.** Claude Code's effort setting reaches models that have trained effort
  levels.
- **A think budget.** Reasoning models that overthink get their thinking closed at the end of
  the current line once they pass the budget for the requested effort, so they answer instead of
  burning every token deliberating.
- **Claude Code's tool search.** Deferred tools stay out of the prompt until the agent loads them.
- **Fast on Apple Silicon.** Built on [MLX](https://github.com/ml-explore/mlx) and mlx-lm:
  continuous batching and a prompt cache that keeps the system prompt warm across conversations.

## Models

Any MLX model folder or Hugging Face repo that mlx-lm can load works by name. To give a model a
short name, effort mapping or think budgets, add it to `~/.aris-server/models.toml`:

```toml
[models."arete:27b"]
path = "~/models/arete-27b-mlx-8bit"
context = 131072
effort_kwarg = "reasoning_effort"
effort_map = { low = "low", medium = "medium", high = "medium", xhigh = "xhigh", max = "xhigh" }
think_budget = { low = 1024, medium = 4096, high = 8192, xhigh = 16384, max = 0 }
```

```bash
aris-server serve --model arete:27b
```

A fine-tuned model can ship as a LoRA adapter on top of its base: add `adapter = "~/adapters/expert1"`
(a folder with `adapter_config.json` and `adapters.safetensors`, as mlx-lm's trainer writes them) to
a model entry, or pass `--adapter`. Switching between a base model and its adapter is a restart.

## Run it as a service

```bash
aris-server service install -- --model arete:27b
aris-server service status
aris-server service uninstall
```

The service starts at login and restarts the server within seconds of any exit (it exits on purpose
if the GPU runs out of memory). Saved prompt caches mean the conversation picks up where it left off.
Logs go to `~/.aris-server/server.log`.

## Status

Early. macOS on Apple Silicon only. See [docs/design.md](docs/design.md) for what's in and out.

## License

Apache-2.0
