# Aris Server

Run open-weight models on your Mac and use them from Claude Code, Aris Code, or anything else that
speaks the Anthropic Messages API.

```bash
uv tool install aris-server
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

## Inkling 2 Mini

Inkling 2 Mini is built in as `inkling-2-mini`: [Thinking Machines'](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
mixture-of-experts model, 26.4B parameters with 3.4B active per token. `pull` downloads
[mlx-community's 4-bit build](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit) (17 GB):

```bash
aris-server pull inkling-2-mini
aris-server serve --model inkling-2-mini
```

On a 48 GB M4 Pro it writes about 32 tokens a second and reads prompts at about 370, with room for about
320k tokens of context. Its thinking and tool calls stream as Anthropic events, like any other model's. Aris
Server serves its text; images aren't supported yet. mlx-lm doesn't run Inkling yet, so Aris Server carries
the model code from mlx-lm's open [pull request](https://github.com/ml-explore/mlx-lm/pull/2086) until a
release does.

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
- **Loop guards.** A model that makes the same tool call three times in a row, and gets the same
  result each time, gets a reminder to use what it already has. A response that degenerates into one
  chunk repeated over and over is steered off it, and ended early if it keeps falling back in, instead
  of running on to the token cap. `--no-guards` turns both off.
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

### Draft heads

Some models ship a small head that guesses the next few tokens. The model checks the guesses in one pass, which costs
about as much as writing one token, and keeps each one it agrees with, so the output is still the model's own. Aris
Server runs two kinds: the MTP head of Qwen3.8-27B (which Arete-27B and Aris 3.8-27B are trained from) and Inkling 2's
own draft head. Point `mtp` at the head's weights:

```toml
[models."arete:27b"]
path = "~/models/arete-27b-mlx-8bit"
mtp = "~/models/qwen3.8-27b-mtp.safetensors"  # the mtp.* tensors of Qwen3.8-27B's checkpoint
mtp_depth = 3

[models."inkling-2-mini"]
mtp = "~/models/inkling-2-mini-mtp/mtp.safetensors"  # from thinkingmachines/Inkling-2-Mini
mtp_depth = 3
```

`mtp_depth` is the most tokens a round guesses. Each round guesses as many as pay there, learned as the reply goes,
since text that copies takes far more guesses than prose; where guessing doesn't pay, the server decodes plainly.
`mtp_adaptive = false` guesses `mtp_depth` every round instead. `mtp_bits` quantizes the head (8 unless set; 0 keeps it
as shipped). `mtp_draft_bits = 4` gives an MTP head's guesses a 4-bit copy of the model's output layer, which each guess
reads in full: faster guesses for about 0.7 GB more memory on the 8-bit 27B (0, the model's own layer, unless set;
Inkling's head scores with its own tables and ignores it). `--mtp`, `--mtp-depth`, `--mtp-fixed`, `--mtp-draft-bits` and
`--no-mtp` do the same on the command line. A request speculates while it decodes alone; several at once decode as
before. On a 48 GB M4 Pro, Claude Code finished a coding task on the 27B in 150 s with its head and 241 s without, its
replies streaming at about 16 tokens a second instead of 7. Inkling's head is a smaller win, because each extra token
Inkling checks reads more of its experts: 1.1-1.3 times mlx-lm's own decoding on code and long contexts, and about even
on prose.

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
