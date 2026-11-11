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

Claude Code assumes a 200k-token window and compacts its conversation about 33k tokens short of the
window it's given. A local model's window is whatever fits beside its weights, so tell Claude Code
what it is: the server logs it at startup, and `/v1/models` lists it as `context_window`. With
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` set to it, the conversation compacts before the server's limit:

```bash
CLAUDE_CODE_MAX_CONTEXT_TOKENS=124928 ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_AUTH_TOKEN=local claude
```

## Aris 3.8-27B

Aris 3.8-27B is built in as `aris-3.8-27b`. It's Aris's coding-agent fine-tune of
[Arete-27B](https://huggingface.co/NousResearch/Arete-27B) by Nous Research, which is built on
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team at Alibaba Cloud. `pull`
picks the best build this Mac can hold, before anything big downloads, and downloads it:

```bash
aris-server pull aris-3.8-27b
aris-server serve --model aris-3.8-27b
```

The 8-bit build takes 27 GB, so it needs a Mac with 48 GB of memory or more, where it has about
125k tokens of context. Its [model card](https://huggingface.co/YourAris/Aris-3.8-27B-MLX-8bit) has
the details and evals.

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

`kv_bits = 8` in a model entry (or `--kv-bits 8`) keeps the KV cache at 8 bits: about twice the context
in the same memory. `--kv-bits 16` turns it back to full precision.

To serve the best weights each Mac can hold, give `path` a list, best first. Before loading, the server
works out how much context each would leave on this Mac and loads the first that leaves `min_context`
tokens (65536 unless set), or else the one that leaves the most. Hugging Face repos are sized from the
Hub before anything downloads, so a Mac only ever fetches weights it can run:

```toml
[models."arete:27b"]
path = ["~/models/arete-27b-mlx-8bit", "~/models/arete-27b-mlx-4bit"]
kv_bits = 8
```

With the 8-bit KV cache, Arete-27B's 8-bit weights leave about 124k tokens of context on a 48 GB Mac and
more than the model's whole context on a 64 GB one. A 32 GB Mac can't hold them and gets the 4-bit
weights, with about 63k.

## Run it as a service

```bash
aris-server service install -- --model aris-3.8-27b
aris-server service status
aris-server service uninstall
```

`aris-server pull --json` reports each step of a download as a JSON line, for apps that set the
server up for their users. `aris-server models` lists configured and built-in models.

The service starts at login and restarts the server within seconds of any exit (it exits on purpose
if the GPU runs out of memory). Saved prompt caches mean the conversation picks up where it left off;
they take up to 20 GB of disk, all models together (`--disk-cache-gb`). Logs go to
`~/.aris-server/server.log`.

## Status

Early. macOS on Apple Silicon only. See [docs/design.md](docs/design.md) for what's in and out.

## License

Apache-2.0
