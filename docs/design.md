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
- `temperature`, `top_p`, `top_k`, `stop_sequences` → sampling and stop settings.
- `max_tokens` is cut so prompt plus output fits the model's `context` (stop reason
  `model_context_window_exceeded` if that cut is what ends the answer). A prompt that leaves under
  1024 tokens of room gets the API's `prompt is too long: N tokens > M maximum`, worded exactly:
  Claude Code matches that text to compact the conversation.
- `tool_choice`: `none` drops the tools; `any` and a named tool (which narrows the list) become an
  instruction, since an open model can't be forced to call one.
- Ignored: `cache_control`, `metadata`, `context_management`, `betas` (the prompt cache is
  automatic).

### Effort and thinking

| Request | Model sees |
| --- | --- |
| `thinking.type: "enabled"` or `"adaptive"` | Thinking on. Claude Code asks for `"adaptive"` on every turn of its agent loop. |
| `thinking.type: "disabled"`, or no `thinking` | Thinking off (the template's `enable_thinking=False`). As in Anthropic's API, a request thinks only when it asks to: Claude Code leaves `thinking` out of its side requests, such as summarizing a page WebFetch read. |
| `output_config.effort` low / medium / high / xhigh / max | The model's own effort control if it has one (for example Arete's `reasoning_effort`), else nothing, plus that effort's think budget. |
| `thinking.budget_tokens` | Overrides the think budget. |

**Think budget:** a logits processor counts thinking tokens. Past the budget, outside a code fence,
a paragraph break the model would take becomes a single newline, and the end-of-thinking token is
forced after the next newline, so the model closes its reasoning on a complete line and answers.
1024 tokens past the budget it closes regardless: inside a code fence the model can write code for
thousands of tokens, and a model can loop on one line that never ends. It closes on one newline, not a
paragraph break, because chat templates re-render past reasoning as `text\n</think>`: a turn closed
as `text\n\n</think>` stops matching its own prompt cache, and the next turn re-reads from there.
Budgets per effort are set per model, with defaults.

### Context that fits

Reading a prompt peaks above the KV cache it builds, and the peak grows with the context: prefill's scratch, and the
copy mlx-lm makes when it grows a cache (in 4096-token blocks here, not 256, so rarely). Measured on Arete-27B at
8 bits on a 48 GB Mac, cold prompts of 30k and 53k tokens peaked at 31.8 and 34.3 GB: 1.8 bytes of peak for each
byte of cache. At startup the server works out how many tokens fit at that rate beside the weights and 3.5 GB of
overhead (about 64k for Arete-27B at 8 bits on that Mac) and answers "prompt is too long" past it, below the model's
own context if need be: Claude Code compacts the conversation instead of the server running out of memory.

### Quantized KV cache

`kv_bits = 8` in a model's settings (or `--kv-bits 8`) keeps keys and values at 8 bits in groups of 64: 34 KB per token
for Arete-27B instead of 64 KB, so about twice the context fits in the same memory (`kv_bits = 4`: 18 KB). mlx-lm
quantizes the cache only outside batching (its server stops batching when `kv_bits` is set), and its quantized attention
suits neither job here: it holds the whole score matrix (about 1.5 GB per layer to read a 512-token chunk of prompt
against 60k tokens of context) and reads a KV head's cache once for each query head sharing it (six times for Arete).
Aris Server's cache batches like mlx-lm's own. A chunk of prompt attends on a dequantized copy of one layer's cache
through the fused kernel (about 250 MB at 60k tokens for Arete), and a decode step attends on the quantized cache
directly, all of a KV head's query heads in one matmul. Saved caches are kept apart by KV precision. The peak keeps
the same 1.8 bytes per byte of cache: with 8-bit Arete plus an adapter (27.0 GB) on a 48 GB Mac, cold prompts of 30k,
60k, 100k and 115k tokens peaked at 30.2, 33.4, 34.8 and 35.6 GB, so about 118k tokens fit there instead of 62k.
Reading a cold 115k-token prompt took 23 minutes (82 tokens/s); a conversation that grows turn by turn reads only
what's new.

Saved caches make room when a request takes its own, before its prompt is read. mlx-lm trims them to the budget
minus what active sequences hold at that moment, which for a cold prompt is next to nothing, so the cache the prompt
is about to build is counted as well. Without that, a cold 53k-token prompt built 3.2 GB of cache beside 3.5 GB of
saved ones and came within 0.5 GB of the limit on a 48 GB Mac.

### Claude Code's billing line

Claude Code's system prompt carries an `x-anthropic-billing-header: cc_version=…` line whose
suffix changes between requests. It is for Anthropic's usage attribution, not the model, and
midway through the system prompt it made everything after it miss the cache. It is dropped.

### Deferred tools

Claude Code sends tools marked `defer_loading` and loads them with tool search mid-conversation.
They stay out of the template's tool list, which heads the prompt: a tool joining it changes the
prompt's first tokens and the whole conversation is re-read (30k tokens took 6.5 minutes on a 27B
model). A loaded tool's definition goes into the tool result that loaded it instead, and the
stream parser still gets every loaded tool's parameter types.

### Streaming

Headers and `message_start` go out at once. Claude Code doesn't count pings as activity, so when
reading the prompt takes more than 30 seconds its progress goes out as thinking (a
`⟨reading prompt N/M⟩` line every 30 seconds, stripped when the conversation comes back, so the
model never sees it); re-reading a long conversation on a small Mac can take longer than Claude
Code waits. A `ping` also goes out whenever nothing else has been sent for five seconds (a tool call is only sent once it closes, so
writing a long file would otherwise be minutes of silence, and Claude Code abandons a silent
stream and retries). Thinking streams as `thinking_delta`,
text as `text_delta`. Tool calls in the XML style most open models use
(`<function=...><parameter=...>`) stream like the Anthropic API's: the `tool_use` block opens when
the name is known and string arguments stream as `input_json_delta` while they are written (Claude
Code abandons a response that produces no content for minutes, and pings don't count). Other tool
formats are parsed whole when the call ends. `message_delta` carries `stop_reason` (`end_turn`, `tool_use`,
`max_tokens`, `stop_sequence`) and usage, including cached input tokens. A client that disconnects
stops its generation, also when nothing is being written to it (non-streaming requests, or a tool
call in progress): the socket is checked every two seconds. Abandoned generations otherwise keep
running, pile up, and exhaust GPU memory.

## Prompt cache

mlx-lm's LRU cache, with three changes. A conversation's cache moves to the turn that continues
it instead of being copied (only the shared system-prompt snapshot is copied), and a snapshot of
part of a prompt still being answered goes to disk only (the request holds its own copy; two copies
of a 51k-token conversation ran a 48 GB Mac out of memory), and when a conversation moves from
reading its prompt to generating, its batch cache is handed over rather than deep-copied (mlx-lm
copies the whole batch there), so a long context is in memory once. A conversation that finishes in a batch with
others is copied out on its own: mlx-lm cuts it out lazily, and its recurrent state as a view of the batch's rows, so
every kept conversation held the whole batch's cache alive (13 GB for eight kept from one 4-request batch on a 27B
model). One alone in its batch is not copied; it is the batch. A turn that just finished is never the entry evicted to
make room: its conversation's next turn continues it within seconds, and evicting it sent that
turn to disk while mlx-lm still held the evicted copy. Evictions are logged. The cache's
budget is what the GPU holds beside the weights less a reserve for prefill scratch. Before a prompt is read, saved
entries make room for the peak reading it reaches (1.8 times the cache it builds, as for the context limit), and the
system-prompt snapshot, which mlx-lm copies while the rest of the prompt is still being read, is kept only where it
fits beside that: a 100k-token system prompt's copy beside its read ran a 48 GB Mac's GPU out of memory. A system
prompt too long to keep is read again by the next conversation that starts from it. And entries go to
`~/.aris-server/cache/<setup>/` when memory lets them go: a finished turn when the LRU evicts it or
at a graceful stop (usually its next turn takes it from memory first and it is never written), the
shared system prompt at once. Writing every turn put 445 GB on an SSD in one day. Each setup (weights, adapter, KV
precision) has its own folder, and one cap, 20 GB by default, covers them all: a server reads only its own folder, so
at startup and whenever a save goes over the cap, the caches of setups no running server holds go first, oldest first,
then its own oldest. A cap per folder let a day of trying seven setups on one Mac fill 116 GB, 96 of them for setups
no longer served. A running server holds a shared lock on its folder, and another server sharing the directory leaves
it alone. A cache that can't be written (a full disk, say) costs a warning, not the server: mlx-lm calls the cache from
its generation loop with no handler, and MLX raises RuntimeError, not OSError. A folder deleted while serving is made
and held again at the next save. A model sometimes
generates a non-canonical tokenization, so its finished turn no longer matches the history the
client sends back; instead of a multi-gigabyte snapshot of each prompt, the recurrent layers' state
at the prompt's end (fixed-size, megabytes) is kept, and such a turn is rebuilt to its prompt's end
in memory: its attention caches trimmed and that state put back. A conversation keeps only its
longest earlier snapshot. Caches are loaded at startup, so a restart or crash resumes a 40k-token conversation
in seconds instead of re-reading it for minutes. A turn that finds memory holding less of its
prompt than disk does (at least 1024 tokens less) loads the disk copy: on a Mac where one long
conversation plus the system prompt doesn't fit the memory budget, the conversation is evicted
every turn, and re-reading 40k tokens costs minutes where the file costs a second. A cache that can't be loaded
(deleted by hand, or damaged) is dropped and the turn looks again, so it reads on from what memory holds rather than
from its first token; one whose file is gone is forgotten before the turn gives up what memory holds. A cache file
without its metadata file, or with its metadata cut off, is a write that was cut off and is deleted: metadata that
couldn't be read stopped the server at every start. A folder that can't be changed never stops startup. Loaded caches are evaluated before generation uses them: MLX binds
a lazy load to the loading thread's stream.

Decoding evaluates the batch's caches every 64 steps. Each step, every linear-attention layer counts its cache's
padding down, lazily, and only the first such layer's mask reads the count; mlx-lm evaluates cache state after each
chunk of a prompt but never while decoding. Unread, the counts grew a graph per layer that kept one Metal buffer alive
per step until the batch emptied: with four requests always in flight, Arete-27B (48 linear-attention layers) reached
Metal's 499,000-buffer limit after about 10,600 steps (some 40k tokens) and the engine died. MLX's memory count barely
moved, a few bytes a buffer, while the process held over a GB more.

## Model settings

`~/.aris-server/models.toml`:

```toml
[models."arete:27b"]
path = "~/models/arete-27b-mlx-8bit"     # local MLX folder or a Hugging Face repo id
context = 131072
effort_kwarg = "reasoning_effort"         # template variable for effort, if the model has one
effort_map = { low = "low", medium = "medium", high = "xhigh", xhigh = "xhigh", max = "xhigh" }
think_budget = { low = 1024, medium = 4096, high = 8192, xhigh = 16384, max = 0 }  # 0 = no limit
kv_bits = 8                               # KV cache precision: 8 or 4 bits; leave out for full precision
```

`adapter` applies a LoRA adapter folder on top of `path` (mlx-lm's format). One configuration is
resident at a time; base and adapted weights don't fit side by side on most Macs, so switching is a
restart. Prompt caches are keyed on the base and adapter files, so they never cross between them.

`path` can also list several weights, best first: `path = ["~/models/arete-27b-mlx-8bit",
"~/models/arete-27b-mlx-4bit"]`. Before loading, the server works out from each one's files how much
context it would leave on this Mac (the same reckoning as the context limit above, at the entry's
`kv_bits`) and loads the first that leaves `min_context` tokens (65536 unless set, or the model's own
`context` if that is less); failing that, the one that leaves the most. A coding agent's conversation
outgrows a small window and pays a long re-read at every compaction, so a 4-bit model with room beats
an 8-bit one squeezed. One `models.toml` then fits every Mac. Weights given by Hugging Face repo id
can't be sized before they load and are skipped while any local folder can.

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
