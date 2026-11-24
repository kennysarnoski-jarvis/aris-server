"""mlx-lm's server engine (batching, prompt cache, thinking and tool parsing), driven by Aris Server."""

import argparse
import copy
import fnmatch
import functools
import gc
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from mlx_lm import server as mls
from mlx_lm.generate import DEFAULT_QUANTIZED_KV_START, GenerationBatch, PromptProcessingBatch, TextStateMachine
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    CacheList,
    KVCache,
    LRUPromptCache,
    PromptTrieResult,
    QuantizedKVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
)

from . import heads as draft_heads
from . import inkling
from . import models as _models
from .budget import BudgetTokens, ThinkBudget
from .config import ModelSpec, home
from .disk_cache import DiskCache
from .guards import LoopGuard, loop_exempt_ids
from .kv_quant import BatchQuantizedKVCache, kv_bytes_per_token, quantize_kv
from .mtp_serve import mtp_snapshot
from .translate import Plan

# Architectures served before mlx-lm releases them (models/), where mlx-lm's loader looks.
_models.register()


class _OwnToolCalls(logging.Filter):
    """mlx-lm warns, on every request with tools, that a model whose tokenizer names no tool-call markers takes no
    tools. It passes them to the chat template all the same, and Aris Server parses some models' tool calls itself
    (Inkling's): for those the warning is false."""

    active = False

    def filter(self, record):
        return not (self.active and record.getMessage().startswith(
            "Received tools but model does not support tool calling"))


_OWN_TOOL_CALLS = _OwnToolCalls()
logging.getLogger().addFilter(_OWN_TOOL_CALLS)  # mlx-lm logs through the root logger


@dataclass
class _Args(mls.GenerationArguments):
    extra_processors: Optional[Callable[[], list]] = None


_orig_processors = mls._make_logits_processors


def _processors(args):
    procs = _orig_processors(args)
    extra = getattr(args, "extra_processors", None)
    return procs + (extra() if extra else [])


mls._make_logits_processors = _processors


_orig_split = PromptProcessingBatch.split


def _split(self, indices):
    """mlx-lm deep-copies the whole batch's cache when sequences move from prompt reading to generation, then
    discards the rest. With one conversation in flight, that's a second copy of a long context (4.5 GB at 66k tokens
    for a 27B model) at the moment generation starts: it ran a 48 GB Mac out of memory. When every sequence moves,
    hand the cache over instead."""
    if sorted(indices) != list(range(len(self.uids))):
        return _orig_split(self, indices)
    moved = self.__class__.__new__(self.__class__)
    moved.__dict__.update(self.__dict__)
    self.prompt_cache = []
    self.filter([])
    return moved


PromptProcessingBatch.split = _split


def _kept_alone(extract):
    def extract_cache(self, idx):
        """mlx-lm cuts a finished conversation's cache out of its batch lazily, and its recurrent state as a view of
        the batch's rows, so a conversation kept from a shared batch held the whole batch's cache alive: every
        sequence's rows, in 4096-token blocks. Eight kept conversations held 13 GB of one 4-request batch's caches for
        a 27B model. Copied out on its own, it holds its own. A conversation alone in its batch is the batch, and a
        copy would only hold its cache twice at the end of every turn."""
        caches = extract(self, idx)
        if len(self.uids) > 1:
            for c in _flat_caches(caches):
                if isinstance(c, ArraysCache):
                    c.cache = [a if a is None else mx.contiguous(a) for a in c.cache]
            mx.eval([x for c in caches for _, x in tree_flatten(c.state) if isinstance(x, mx.array)])
        return caches
    return extract_cache


PromptProcessingBatch.extract_cache = _kept_alone(PromptProcessingBatch.extract_cache)
GenerationBatch.extract_cache = _kept_alone(GenerationBatch.extract_cache)

# Each decode step, every linear-attention layer counts its cache's padding down (ArraysCache.advance), lazily, and
# only the first such layer's mask reads the count. Unread, the others grow a graph that keeps every step's scalar
# alive, one Metal buffer a step per layer, until the batch empties. mlx-lm evaluates the caches' state after each
# chunk of a prompt but never while decoding, so with requests always in flight Arete-27B (48 linear-attention
# layers) reached Metal's 499,000-buffer limit after about 10,600 steps, some 40k tokens at four in flight, and the
# engine died. MLX's own memory count barely moves (a few bytes a buffer); the process held over a GB more.
STATE_EVAL_STEPS = 64

_orig_generation_next = GenerationBatch.next


def _next_evaluating_cache_state(self):
    responses = _orig_generation_next(self)
    self._steps_unevaluated = getattr(self, "_steps_unevaluated", 0) + 1
    if self._steps_unevaluated >= STATE_EVAL_STEPS:
        self._steps_unevaluated = 0
        mx.async_eval([x for c in self.prompt_cache for _, x in tree_flatten(c.state) if isinstance(x, mx.array)])
    return responses


GenerationBatch.next = _next_evaluating_cache_state

# mlx-lm grows a KV cache by concatenating a new block onto the old one, a full copy of the conversation's keys and
# values, every 256 tokens. At 60k tokens on a 27B model that is a 4 GB copy each time (both halves alive while it
# runs): the spikes behind running out of memory, and memory bandwidth taken from generation. Grow in larger blocks.
KV_GROWTH_TOKENS = 4096
KVCache.step = BatchKVCache.step = QuantizedKVCache.step = BatchQuantizedKVCache.step = KV_GROWTH_TOKENS


# Reading a prompt peaks above the cache it builds (prefill scratch, the cache's growth copies), and the peak grows
# with the context: cold prompts of 30k and 53k tokens on Arete-27B at 8 bits peaked at 31.8 and 34.3 GB on a 48 GB
# Mac, 1.8 bytes of peak for each byte of KV cache.
PEAK_PER_KV_BYTE = 1.8
# Beside the weights and that peak: the shared system-prompt cache, kept states, MLX's buffer pool, and headroom.
CONTEXT_OVERHEAD = int(3.5 * (1 << 30))


def context_that_fits(limit: int, weights: int, kv_bytes_per_token: int, overhead: int) -> int:
    """Tokens of context this machine can read and hold beside the weights, at the peak reading a prompt reaches."""
    room = limit - weights - overhead
    return max(0, int(room // (PEAK_PER_KV_BYTE * kv_bytes_per_token))) // 1024 * 1024


# Claude Code's own system prompt and tools take 9-15k tokens, and it compacts at 80% of a small window (as
# `aris-server claude` and Aris Code tell it to), so below about 32k it has no room left to work in.
MIN_USEFUL_CONTEXT = 32768


class NotEnoughMemory(RuntimeError):
    """The model can't run in this Mac's GPU memory."""


def weights_on_disk(*folders: str | None) -> int | None:
    """The bytes a model's weights (and adapter) will take once loaded: their safetensors files. None for a model that
    isn't a local folder (a Hugging Face repo id: see weights_and_config)."""
    total = 0
    for folder in folders:
        if folder is None:
            continue
        path = Path(folder)
        if not path.is_dir():
            return None
        # mlx-lm loads only adapters.safetensors from an adapter folder; the trainer leaves its checkpoints beside it.
        files = [path / "adapters.safetensors"] if (path / "adapter_config.json").is_file() else path.glob("*.safetensors")
        total += sum(p.stat().st_size for p in files if p.is_file())
    return total


def check_memory(name: str, limit: int, weights: int, fits: int | None = None) -> None:
    """Refuse a model this Mac can't hold rather than serve it without a context limit, which runs out of memory on
    the first long prompt (and under the service, again after every restart)."""
    gb = 1 << 30
    if limit - weights - CONTEXT_OVERHEAD <= 0 or fits == 0:
        raise NotEnoughMemory(
            f"{name} does not fit this Mac's GPU memory: its weights take {weights / gb:.1f} GB, the GPU can use "
            f"{limit / gb:.1f} GB, and reading a prompt needs {CONTEXT_OVERHEAD / gb:.1f} GB more beside the weights. "
            "Use a smaller quantization of the model: 4-bit weights take about half the memory of 8-bit.")
    if fits is not None and fits < MIN_USEFUL_CONTEXT:
        logging.warning("only about %d tokens of context fit beside %s's weights, and Claude Code needs at least %d: "
                        "a smaller quantization of the model, kv_bits = 8, or --no-mtp (no draft head) leaves more "
                        "room, where the model has them", fits, name, MIN_USEFUL_CONTEXT)


def _kv_shape(config: dict) -> tuple[dict, int, int]:
    """A model config's text settings, its KV heads and its head size."""
    text = config.get("text_config", config)
    heads = text.get("num_key_value_heads") or text["num_attention_heads"]
    return text, heads, text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]


# The weight files mlx-lm loads from a Hugging Face repo, and everything it downloads with them.
HUB_WEIGHTS = "model*.safetensors"


@functools.lru_cache(maxsize=32)
def hub_info(repo: str):
    """A Hugging Face repo's file listing, with sizes, from the Hub (None when it can't be reached or doesn't know the
    repo: offline, misspelled, or private without a login). Asked once per process."""
    import huggingface_hub

    try:
        return huggingface_hub.HfApi().model_info(repo, files_metadata=True)
    except Exception as e:  # noqa: BLE001  offline, a missing or private repo: the caller falls back to the cache
        logging.debug("could not ask the Hugging Face Hub about %s: %s", repo, e)
        return None


def _hub_weights_and_config(repo: str) -> tuple[int, dict] | None:
    """A Hugging Face repo's weight bytes and config before its weights download: their sizes from the Hub (the config
    is a few kilobytes), or else from the local cache when the Hub can't be reached."""
    import huggingface_hub

    if (info := hub_info(repo)) is not None:
        weights = sum(s.size or 0 for s in info.siblings or [] if fnmatch.fnmatch(s.rfilename, HUB_WEIGHTS))
        try:
            config = Path(huggingface_hub.hf_hub_download(repo, "config.json", revision=info.sha))
            return (weights, json.loads(config.read_text())) if weights else None
        except Exception as e:  # noqa: BLE001  the listing came back but the download didn't: try the cache
            logging.debug("could not read %s's config from the Hub: %s", repo, e)
    try:
        folder = huggingface_hub.snapshot_download(repo, local_files_only=True,
                                                   allow_patterns=["*.json", HUB_WEIGHTS])
    except Exception:  # noqa: BLE001  not in the cache either
        return None
    weights, config = weights_on_disk(folder), Path(folder) / "config.json"
    return (weights, json.loads(config.read_text())) if weights and config.is_file() else None


def weights_and_config(path: str, adapter: str | None = None) -> tuple[int, dict] | None:
    """The bytes a model's weights (and adapter) will take once loaded, and its config, known before anything loads or
    downloads: from a local folder's files, or for a Hugging Face repo id from the Hub (or the local cache when the
    Hub can't be reached). None when neither can say."""
    if Path(path).is_dir():
        weights, config = weights_on_disk(path, adapter), Path(path) / "config.json"
        return (weights, json.loads(config.read_text())) if weights is not None and config.is_file() else None
    hub = _hub_weights_and_config(path)
    extra = weights_on_disk(adapter) if adapter else 0
    if hub is None or extra is None:
        return None
    return hub[0] + extra, hub[1]


def context_before_loading(path: str, adapter: str | None, kv_bits: int | None, limit: int,
                           spec: ModelSpec | None = None) -> int | None:
    """Tokens of context the weights in `path` would leave on this Mac, known before they load (or download), beside
    `spec`'s draft head when it has one. None for weights that can't be sized: see weights_and_config."""
    sized = weights_and_config(path, adapter)
    if sized is None:
        return None
    weights, config = sized
    weights += draft_heads.extra_bytes(spec, config) if spec is not None else 0
    text, heads, head_dim = _kv_shape(config)
    layers = full_attention_layers(text)
    return context_that_fits(limit, weights, kv_bytes_per_token(layers, heads, head_dim, kv_bits), CONTEXT_OVERHEAD)


def full_attention_layers(text: dict) -> int:
    """How many KV caches grow with the context: one per full-attention layer (a hybrid model's linear-attention
    layers and sliding-window layers keep a fixed size), and one per pass through a looped model's core."""
    n = text["num_hidden_layers"]
    if types := text.get("layer_types"):
        full = [i for i, t in enumerate(types) if t == "full_attention"]
    elif interval := text.get("full_attention_interval"):
        return n // interval
    elif (local := text.get("local_layer_ids")) is not None:
        full = [i for i in range(n) if i not in set(local)]
    else:
        full = list(range(n))
    loop = text.get("loop_config") or {}
    if loop.get("num_passes"):
        core = range(loop["core_start"], loop["core_end"])
        return sum(loop["num_passes"] if i in core else 1 for i in full)
    return len(full)


def _flat_caches(caches):
    """A model's layer caches with any CacheList (a layer keeping several caches) opened up."""
    for c in caches:
        if isinstance(c, CacheList):
            yield from _flat_caches(c.caches)
        else:
            yield c


_NOT_KEPT = "not kept"  # a layer cache whose state at a prompt's end can't be kept (see _end_state)


def _end_state(c):
    """What a layer's cache must keep from the end of a prompt to be put back there after it read on (see
    _rebuild_diverged): nothing for attention, which cuts back to any token; a recurrent layer's arrays, which hold
    only its latest state; a sliding window's last rows, which decode overwrites in its ring (168 MB for Inkling 2
    Mini's 40 window visits); and each part of a CacheList."""
    if isinstance(c, CacheList):
        return [_end_state(x) for x in c.caches]
    if isinstance(c, ArraysCache):
        return list(c.cache)
    if isinstance(c, RotatingKVCache):
        if c.keep or c.keys is None:
            return _NOT_KEPT
        n = min(c.offset, c.max_size)
        # Copied out, so neither later writes into the live ring nor a batch it is a view of reach it.
        keys = mx.contiguous(c._temporal_order(c.keys)[..., -n:, :])
        values = mx.contiguous(c._temporal_order(c.values)[..., -n:, :])
        mx.eval(keys, values)
        return keys, values, c.offset
    return None


def _can_put_back(c, state, n: int) -> bool:
    """Whether _put_back can return this layer's cache to token n from its state there."""
    if isinstance(c, CacheList):
        return isinstance(state, list) and len(state) == len(c.caches) and all(
            _can_put_back(x, s, n) for x, s in zip(c.caches, state))
    if isinstance(c, ArraysCache):
        return isinstance(state, list)
    if isinstance(c, RotatingKVCache):
        return isinstance(state, tuple) and state[2] == n
    return state is None and hasattr(c, "trim") and getattr(c, "offset", -1) >= n


def _put_back(c, state, n: int) -> None:
    if isinstance(c, CacheList):
        for x, s in zip(c.caches, state):
            _put_back(x, s, n)
    elif isinstance(c, ArraysCache):
        c.cache = list(state)
    elif isinstance(c, RotatingKVCache):
        c.keys, c.values, c.offset = state
        c._idx = c.keys.shape[2]  # in temporal order, as a multi-token read leaves it
    else:
        c.trim(c.offset - n)


def choose_weights(spec: ModelSpec, limit: int) -> ModelSpec:
    """Of a model's weights, best first, the first that leaves `min_context` tokens of context on this Mac (or the
    model's whole context, if that is less); failing that, the one that leaves the most."""
    if not spec.choices:
        return spec
    fits = {p: context_before_loading(p, spec.adapter, spec.kv_bits, limit, spec) for p in spec.choices}
    want = min(spec.min_context, spec.context or spec.min_context)
    sized = [p for p in spec.choices if fits[p] is not None]
    # Among weights that leave the same (no context at all, say), the later ones are the smaller.
    most = max(sized, key=lambda p: (fits[p], spec.choices.index(p)), default=spec.choices[0])
    chosen = next((p for p in sized if fits[p] >= want), most)
    logging.info("memory: %s; loading %s", "; ".join(
        f"{p} leaves about {fits[p]} tokens of context" if fits[p] is not None else f"{p} can't be sized before loading"
        for p in spec.choices), chosen)
    return replace(spec, path=chosen)


_orig_generate = mls.ResponseGenerator._generate


def _generate_then_stop_hook(self):
    """MLX arrays can only be evaluated on the thread whose streams made them, so work that touches the prompt
    cache at shutdown (writing it to disk) runs here, on the generation thread, as it exits."""
    try:
        _orig_generate(self)
    finally:
        hook = getattr(self, "on_stop", None)
        if hook is not None:
            try:
                hook()
            except Exception as e:  # noqa: BLE001  a failed flush costs one re-read after the restart
                logging.warning("could not write prompt caches before stopping: %s", e)


mls.ResponseGenerator._generate = _generate_then_stop_hook


DISK_MIN_GAIN = 1024  # fewer extra tokens than this are as quick to read as the cache file
MAX_KEPT_STATES = 2  # prompt-end recurrent states kept per kind (requests in flight; finished turns)

# A hybrid model's linear-attention layers carry a running state that can't be cut back to an earlier token, and a
# sliding window's ring holds only the last rows it read, so a request that leaves every cached entry part-way through
# reused none of what it still shared with them. A coding agent's plugin that rewrote one line near the end of its
# system prompt after each check it ran sent a 57k-token conversation back to its first token, 630 s on a 48 GB Mac,
# to change a line at token 11k. Checkpoints of that state, taken as prompts are read, let such a request resume from
# the last one before it leaves, with the attention cache of an entry that shares that much cut back to the same token.
CHECKPOINT_GRID = 512      # taken this far apart over a prompt's system region (its system prompt and tools, where a
                           # request's first change usually is), then thinned evenly to their share of the budget.
                           # mlx-lm reads a prompt 512 tokens at a time, so cuts on this grid add at most one read.
CHECKPOINT_SPAN = 32768    # the grid's reach
CHECKPOINT_SHARE = 8       # checkpoints hold at most this fraction of the prompt cache's budget (157 MB each for
                           # Arete-27B, 168 MB for Inkling 2 Mini)
MAX_CHECKPOINTS = 64
LINE_BACKOFF = 512         # how far back from where a request left the cache to look for the start of its line
HISTORY_LOOKBACK = 64      # how far back from a prompt's end to look for the start of the turn it asks for
RESYNC_BYTES = 1024        # how far two tokenizations of one text may run apart before they meet at a token boundary


def _state_arrays(state):
    """The arrays in a layer state as _end_state keeps it."""
    if isinstance(state, mx.array):
        yield state
    elif isinstance(state, (list, tuple)):
        for s in state:
            yield from _state_arrays(s)


def _all_kept(state) -> bool:
    if state is _NOT_KEPT:
        return False
    return not isinstance(state, list) or all(_all_kept(s) for s in state)


def _checkpointable(layers) -> bool:
    """Whether a model's caches hold state that can't be cut back to an earlier token (recurrent layers' arrays, a
    sliding window's ring), all of it state a checkpoint can keep, beside attention that cuts back to any token: such a
    model's prompt cache takes checkpoints."""
    flat = list(_flat_caches(layers))
    return any(isinstance(c, (ArraysCache, RotatingKVCache)) for c in flat) and all(
        isinstance(c, ArraysCache) or type(c) in (KVCache, QuantizedKVCache)
        or (isinstance(c, RotatingKVCache) and not c.keep) for c in flat)


def _resume_layer(layer, state, n: int):
    if isinstance(layer, CacheList):
        if not isinstance(state, list) or len(state) != len(layer.caches):
            return None
        parts = [_resume_layer(c, s, n) for c, s in zip(layer.caches, state)]
        return None if any(p is None for p in parts) else CacheList(*parts)
    resumed = copy.copy(layer)
    if isinstance(layer, ArraysCache) and isinstance(state, list):
        resumed.cache, resumed.left_padding, resumed.lengths = list(state), None, None
    elif isinstance(layer, RotatingKVCache) and isinstance(state, tuple) and state[2] == n:
        resumed.keys, resumed.values, resumed.offset = state
        resumed._idx = resumed.keys.shape[2]  # in temporal order, as a multi-token read leaves it
    elif state is None and type(layer) in (KVCache, QuantizedKVCache) and layer.offset >= n:
        resumed.keys, resumed.values = tree_map(lambda x: x[..., :n, :], (layer.keys, layer.values))
        resumed.offset = n
    else:
        return None
    return resumed


def _resume(layers, states, n: int):
    """A cache at token n: an entry's attention layers cut back to n, and a checkpoint's state at n for the rest (a
    recurrent layer's arrays, a sliding window's last rows; each part of a CacheList). The cut layers are views of the
    entry's arrays, which stays as it was, and the checkpoint stays too: mlx-lm copies a cache into its batch before
    reading on. None when a layer is neither."""
    if len(layers) != len(states):
        return None
    out = [_resume_layer(layer, state, n) for layer, state in zip(layers, states)]
    return None if any(c is None for c in out) else out


@dataclass
class _Checkpoint:
    states: list         # each layer's state at the checkpoint, as _end_state keeps it
    nbytes: int
    touched: int         # the latest fetch whose prompt runs along it (or the fetch it was taken under)
    learned: bool = False  # taken where a request left the cache, rather than on the grid


def _split_segments(segments: list, types: list, at, history: int | None = None) -> None:
    """Cut a prompt's segments (in place) at these token positions: mlx-lm hands the prompt cache a copy of the cache
    at the end of every segment but the last, so each cut is where a "checkpoint" is taken. The cut at `history`,
    where the prompt's history ends, hands it the state a rebuild resumes from instead (see _history_end)."""
    cuts = sorted(set(at) | ({history} if history is not None else set()))
    out_segments, out_types, start = [], [], 0
    for segment, kind in zip(segments, types):
        end, done = start + len(segment), start
        for p in cuts:
            if done < p < end:
                out_segments.append(segment[done - start:p - start])
                out_types.append("history" if p == history else "checkpoint")
                done = p
        out_segments.append(segment[done - start:])
        out_types.append(kind)
        start = end
    segments[:], types[:] = out_segments, out_types


class _EvictionTrie:
    """mlx-lm's prompt trie, reporting entries the LRU evicts (and not ones a request takes) so they can go to disk
    on the way out."""

    def __init__(self, trie, on_evict):
        self._trie, self._on_evict = trie, on_evict
        self.taking = False

    def __getattr__(self, name):
        return getattr(self._trie, name)

    def pop(self, model, tokens):
        entry = self._trie.pop(model, tokens)
        if not self.taking:
            self._on_evict(tokens, entry)
        return entry

    def search(self, model, tokens) -> PromptTrieResult:
        """mlx-lm's search, same results, in time linear in the entries' length. Looking for the shortest entry past
        where the prompt leaves the trie, mlx-lm's copies the path at every step down: quadratic. A prompt that leaves
        a long conversation early, the one after a compaction leaving an 88k-token conversation after its 5k-token
        system prompt, took 5.4 s on an M5 to search, and a request searches twice: the 12 and 19 s Claude Code waited
        after each compaction on Aris 3.8-27B."""
        trie = self._trie._trie
        if model not in trie:
            return PromptTrieResult(model, None, None, None, 0)
        current = trie[model]
        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)
        last_index, index = -1, 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)
        shorter = tokens[:last_index + 1] if last_index > 0 else None
        longer = list(tokens[:index]) + _shortest_below(current) if index > 0 else None
        return PromptTrieResult(model, None, shorter, longer, index)


def _shortest_below(node) -> list:
    """The tokens from a trie node to the nearest entry below it: mlx-lm's depth-first search in its order (so a tie
    goes the same way), keeping each step's token and the step before it in place of a copy of the path so far."""
    steps: list[tuple[int, int]] = []  # (the step before, token)
    best = None  # (depth, its last step)
    stack = [(node, 0, -1)]
    while stack:
        current, depth, step = stack.pop()
        if "__value__" in current:
            if best is None or depth < best[0]:
                best = (depth, step)
        elif best is None or depth < best[0]:
            for tok in current:
                steps.append((step, tok))
                stack.append((current[tok], depth + 1, len(steps) - 1))
    path = []
    step = best[1] if best is not None else -1
    while step >= 0:
        step, tok = steps[step]
        path.append(tok)
    return path[::-1]


class _RecencyOrder(LRUPromptCache.CacheOrder):
    """The order the prompt cache lets entries go in: the least recently used first, whatever its type. mlx-lm's own
    order takes a finished turn before any system prompt while there are at least as many turns as user-prompt
    snapshots, which this cache never keeps. Each Claude Code session's system prompt is its own (it carries the
    working directory and git status), so once eight sessions' system prompts filled the cache's entries, the turn
    that had just finished was the one to go, every turn: written to disk, and read back by the next turn. That wrote
    367 GB in a night of agent tasks on a 48 GB Mac."""

    def __init__(self):
        super().__init__()
        self._order: deque = deque()  # (model, tokens, type), least recently used first

    def push(self, model, tokens, cache_type="assistant"):
        super().push(model, tokens, cache_type)
        self._order.append((model, tokens, cache_type))

    def _at(self, model, tokens):
        return next((i for i, (m, t, _) in enumerate(self._order) if m == model and t == tokens), None)

    def remove(self, model, tokens):
        super().remove(model, tokens)
        i = self._at(model, tokens)
        if i is not None:
            del self._order[i]

    def touch(self, model, tokens):
        """An entry a request just started from goes last."""
        i = self._at(model, tokens)
        if i is not None:
            self._order.append(self._order[i])
            del self._order[i]

    def pop(self):
        model, tokens, cache_type = self._order.popleft()
        self._lrus[cache_type].remove((model, tokens))
        return model, tokens


class MovingPromptCache(LRUPromptCache):
    """mlx-lm copies a saved cache for the request that continues it, so a long conversation sits in memory twice
    for the whole turn (2.4 GB extra at 37k tokens on a 27B model, enough to run a 48 GB Mac out of memory). A
    conversation's cache is only ever continued by its next turn, so it moves to that request instead; the shared
    system-prompt segment is still copied for every conversation that starts from it."""

    disk = None  # DiskCache: entries go to disk when memory lets them go (see insert_cache)
    # Full snapshots of a prompt being answered go to disk only from this length (off by default: the in-memory
    # rebuild below covers what they were for, and at 2-4 GB each they wrote ~45 GB an hour).
    snapshot_min_tokens = 1 << 62
    light_min_tokens = 1024  # shorter prompts are quick to re-read
    kv_bytes_per_token = None  # set by the engine: the cache one token of context adds to a sequence
    _pending_bytes = 0  # the cache the request just handed its own is about to build (see trim_to)
    _read_peak = 0  # what reading the latest prompt needs at its peak, while its system-prompt copy may arrive
    # Checkpoints (see CHECKPOINT_GRID): their spacing over a prompt's system region, set by the engine for a model
    # with state that can't be cut back (None: none are taken; an attention-only cache is cut back as it is).
    checkpoint_every = None
    checkpoint_span = CHECKPOINT_SPAN
    checkpoint_min_gain = DISK_MIN_GAIN  # where a request left the cache: a checkpoint there must add this many tokens
    # Tokens at which a prompt's system region has ended: Qwen's end of turn, which closes its system turn (system
    # prompt and tools); Inkling's user, model and tool headers, which open the conversation after its tool
    # declarations, system prompt and effort message.
    system_ends: frozenset = frozenset()
    # The token that opens the turn a request asks for (the first of the chat template's generation prompt, set by the
    # engine beside checkpoint_every): the last one near a prompt's end is where its history ends (see _history_end).
    # Empty for a template that adds no generation prompt, whose prompts end where their history does.
    turn_starts: frozenset = frozenset()
    tokenized = None  # set by the engine as it tokenizes a request: its segments, which fetching it cuts at checkpoints
    hold = None  # and where the messages the server added to it start, if it added any (appended_start)
    _leaving = ((), 0)  # the cached entry the request being fetched runs along furthest, and how far
    _resuming = None  # a checkpoint's cache for the request being fetched, until something covers more

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reading: list[list[int]] = []  # prompts of the requests in flight
        self._trie = _EvictionTrie(self._trie, self._evicted)
        self._lru = _RecencyOrder()
        self._budget_lock = threading.Lock()  # see lower_budget
        # A checkpoint's tokens -> each layer's state after reading them, least recently run along first.
        self._checkpoints: dict[tuple, _Checkpoint] = {}
        self._fetches = 0  # requests fetched so far: a checkpoint's recency
        self._learn: dict[tuple, None] = {}  # checkpoints planned where a request left the cache, until taken
        self._pieces: dict[int, str] = {}  # each token's text, for finding where lines start
        # Recurrent state of requests in flight where their history ends (or at their prompt's end), then of the turns
        # they finished as. A next request that parts from a finished turn there or later (its history re-rendered
        # with other tokens than were generated, a compaction or a retry in its place) gets it rebuilt from these: its
        # attention caches trimmed (they hold the prompt's tokens first) and its recurrent layers' state put back. The
        # recurrent state is small and fixed-size, so this costs megabytes where a full snapshot costs gigabytes.
        self._prompt_state: dict[tuple, list] = {}
        self._turn_state: dict[tuple, tuple[int, list]] = {}
        # Recent requests cut where their history ends -> their prompt's length: that state stands in for their end's.
        self._history_ends: dict[tuple, int] = {}
        self._own_seen: dict[tuple, int] = {}  # see _note_own_tokens
        # Requests in flight -> (session, system prompt length), and the entries their turns left, for note_session.
        self._asking: list[tuple[list, str | None, int | None]] = []
        self._sessions: dict[tuple, tuple[str, int]] = {}
        self._ended: set[tuple] = set()  # entries a session started its conversation over from: dropped, not saved

    def _save(self, tokens, prompt_cache, cache_type):
        if self.disk is None or self.disk.has(tokens):
            return
        try:
            self.disk.save(tokens, prompt_cache, cache_type)
        except Exception as e:  # noqa: BLE001  a full disk, or a folder that can't be written
            # MLX raises RuntimeError there, not OSError, and mlx-lm calls the prompt cache from its generation loop
            # with no handler: an error here stopped that loop, and the server with it.
            logging.warning("could not save a prompt cache to disk: %s", e)

    def _evicted(self, tokens, entry):
        self._save(tokens, entry.prompt_cache, entry.cache_type)

    def flush(self):
        """Write what is only in memory, for a restart to resume from (a graceful stop calls this)."""
        for kind, lru in self._lru._lrus.items():
            for model, key in list(lru):
                self._save(key, self._trie.get(model, key).prompt_cache, kind)

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        with self._budget_lock:
            self._insert(model, tokens, prompt_cache, cache_type)

    def lower_budget(self, shrink: Callable[[int], int]) -> int | None:
        """Lower the budget to shrink(budget) if that is lower, from any thread: the new budget, or None. Keeping a turn
        moves the budget aside for a moment on the generation thread (see _insert), and putting it back undid a
        lowering made from a request's thread in between: 12.1 GB logged on a 48 GB Mac, 13.9 GB in force two hours
        later."""
        with self._budget_lock:
            cap = shrink(self.max_bytes)
            if cap >= self.max_bytes:
                return None
            self.max_bytes = cap
            return cap

    def _insert(self, model, tokens, prompt_cache, cache_type):
        if cache_type == "checkpoint":  # a cut the request's fetch planned: only the recurrent state is kept
            self._keep_checkpoint(tokens, prompt_cache)
            return
        # A snapshot of part of a prompt still being answered: the request holds its own copy, so keeping this one
        # in memory too doubles the conversation (3.5 GB more at 51k tokens) and ran a 48 GB Mac out of memory.
        # Disk is enough for the retry it exists for. The shared system prompt stays in memory.
        mid_request = cache_type != "system" and any(
            len(tokens) < len(p) and p[:len(tokens)] == list(tokens) for p in self.reading)
        if mid_request and len(tokens) >= self.light_min_tokens:
            key = tuple(tokens)
            # One state per request: where its history ends when it was cut there (see _history_end), which serves
            # every next request the one at its prompt's end serves, and a compaction or retry besides. mlx-lm's own
            # snapshot of that request, a token before its prompt's end, is then left out.
            if cache_type == "history" or not any(len(key) == n - 1 and key[:len(k)] == k
                                                  for k, n in self._history_ends.items()):
                # ~150 MB each for a 27B model: keep only what a request in flight or a conversation's latest turn needs.
                self._prompt_state[key] = [_end_state(c) for c in prompt_cache]
                while len(self._prompt_state) > MAX_KEPT_STATES:
                    self._prompt_state.pop(next(iter(self._prompt_state)))
        elif cache_type != "system":
            key = tuple(tokens)
            self._tag_session(tokens)
            at = max((k for k in self._prompt_state if len(k) < len(key) and key[:len(k)] == k), key=len, default=None)
            for older in [k for k in self._turn_state if len(k) < len(key) and key[:len(k)] == k]:
                self._turn_state.pop(older)  # this conversation's earlier turns: superseded
            if at is not None:
                self._turn_state[key] = (len(at), self._prompt_state.pop(at))
                while len(self._turn_state) > MAX_KEPT_STATES:
                    self._turn_state.pop(next(iter(self._turn_state)))
        cap, size = self.max_bytes, sum(c.nbytes for c in prompt_cache)
        room = cap - self._checkpoint_bytes()  # what checkpoints leave of the budget
        if cache_type == "system" and size + self._read_peak > room:
            # mlx-lm copies the cache at the end of the system prompt, while the rest of that prompt is still being
            # read. Kept beside what the read needs, a long system prompt's copy ran a 48 GB Mac's GPU out of memory
            # just after it read a 100k-token one (3.7 GB of copy beside a 6.1 GB read, with a 4.8 GB budget).
            # It still goes to disk, the copy already made: the conversation after a compaction starts from this system
            # prompt, and with memory that tight it read it cold, 19k tokens in 166 s on Aris 3.8-27B, where loading
            # it from disk takes a tenth of a second.
            gb = 1 << 30
            self._save(tokens, prompt_cache, cache_type)
            logging.info("memory cache: not keeping the %d-token system prompt: its %.1f GB beside the %.1f GB reading "
                         "its prompt needs is more than the %.1f GB budget%s", len(tokens), size / gb,
                         self._read_peak / gb, room / gb, "; it's on disk" if self.disk is not None else "")
            return
        if not mid_request:
            before = {k for k in self._entry_keys()}
            # The turn that just finished is the cache its conversation's next turn continues, often within seconds:
            # when it's larger than the budget, other entries go and it stays. Evicting it sent that next turn to
            # disk while mlx-lm still briefly held the evicted copy, putting two copies of the conversation in memory.
            keep = size + self._n_bytes_by_type.get("system", 0)
            if cache_type != "system" and keep > room:
                self.max_bytes = keep
            elif cache_type == "system":  # room for it beside the read still under way: other saved entries go
                self.max_bytes = room - self._read_peak
            else:
                self.max_bytes = room
            try:
                super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
            finally:
                self.max_bytes = cap
            gone = before - {k for k in self._entry_keys()}
            if gone:
                gb = 1 << 30
                logging.info("memory cache: inserting %d tokens evicted %s; %d left, %.1f of %.1f GB", len(tokens),
                             ", ".join(f"{n}-token {kind}" for kind, n in sorted(gone)), len(self),
                             self.nbytes / gb, self.max_bytes / gb)
        # Disk writes are kept to what is needed: a cache of 40k tokens is 3 GB, and writing every turn put 445 GB on
        # an SSD in one day. A finished turn is written only when memory evicts it (or at shutdown): usually its next
        # turn takes it from memory first. The shared system prompt is written at once, and a snapshot of a long
        # prompt being answered too: it is what a retry, or a next turn whose re-rendered history tokenizes
        # differently from what was generated, resumes from.
        if cache_type == "system" or (mid_request and len(tokens) >= self.snapshot_min_tokens):
            self._save(tokens, prompt_cache, cache_type)

    def _entry_keys(self):
        """(type, length) of each entry, for logging what an insert evicted."""
        return [(kind, len(key)) for kind, lru in self._lru._lrus.items() for _, key in lru] \
            if hasattr(self._lru, "_lrus") else []

    decode = None  # the tokenizer's decode, for logging where a turn and its re-rendered history part ways
    token_bytes = None  # a token's bytes (see token_bytes), for own_tokens: None leaves requests' tokens as they are

    def own_tokens(self, model, tokenized):
        """The request's tokenized prompt with a cached entry's tokens in place where the two spell the same text
        with other tokens. A reply comes back in the next request's history as text, and tokenizing it doesn't always
        give the tokens the model generated (`[0, ` as `[`, `0`, `, `, in a long file write). The request then leaves
        the finished turn there, inside the reply, and a hybrid model reads the rest of the reply again from the state
        kept where that turn's prompt ended: 9 times in a day of agent tasks on Aris 3.8-27B, up to 15k tokens each.
        With the entry's tokens there, the request continues the turn whole, the text exactly the same, the model
        reading the tokens it generated. Each later request of the conversation tokenizes that reply the other way
        again, and lines up with the conversation's cached turn the same way, wherever it is cached."""
        prompt, segments, types, initial_state, vision = tokenized
        if self.token_bytes is None or vision is not None or len(prompt) < self.light_min_tokens:
            return tokenized
        prompt = list(prompt)
        found = self._trie.search(model, prompt)
        if found.exact is not None:
            return tokenized
        candidates = [(found.common_prefix, list(found.longer), "cached")] if found.longer is not None else []
        if self.disk is not None:
            for t, _ in list(self.disk.index.values()):
                if len(t) > found.common_prefix:
                    key = list(t)
                    candidates.append((_first_difference(key, prompt, 0, 0), key, "saved"))
        if not candidates:
            return tokenized
        start, key, where = max(candidates, key=lambda c: (c[0], len(c[1])))
        if not self.light_min_tokens <= start < min(len(key), len(prompt)):
            return tokenized  # it covers all it shares already, or too little to be worth lining up
        try:
            aligned, shared, places = align_tokens(key, prompt, start, self.token_bytes)
        except Exception as e:  # noqa: BLE001  a token the table can't spell: the request's own tokens, as before
            logging.warning("prompt cache: could not line a request up with a cached entry: %s", e)
            return tokenized
        if not places or shared <= start:
            return tokenized
        resume = len(prompt) - (len(aligned) - shared)  # where the request's own tokens take over again
        bounds, at = [], 0
        for seg in segments[:-1]:
            at += len(seg)
            if start < at < resume:
                return tokenized  # a segment would end inside what the entry's tokens now spell
            bounds.append(at if at <= start else at - resume + shared)
        cut = [0, *bounds, len(aligned)]
        self._note_own_tokens(key, prompt, shared, places, where)
        return aligned, [aligned[a:b] for a, b in zip(cut, cut[1:])], types, initial_state, vision

    message_starts: frozenset = frozenset()  # the tokens that open a message (Qwen's <|im_start|>), for note_session

    def note_session(self, model, tokenized, session):
        """Which client session a request comes from, and the cached conversation it starts over, if any. A Claude
        Code session that compacts, or is cleared, sends its system prompt again and then a first message other than
        its conversation's: nothing continues that conversation's cached turn after that, yet on its way out of memory
        it was saved to disk, 3.4 GB for an 87k-token turn of Aris 3.8-27B, written for nothing at every compaction.
        Such a turn is dropped instead, once this request has taken what it shares. Only within one named session
        (Claude Code's X-Claude-Code-Session-Id): two sessions can share a system prompt, and one's new conversation
        must leave the other's cache alone. A subagent's system prompt is its own, and a fork of the conversation (a
        compaction's own request, say) shares it past its first message, so neither ends it."""
        prompt, segments, types = tokenized[0], tokenized[1], tokenized[2]
        system = len(segments[0]) if len(segments) > 1 and types and types[0] == "system" else None
        self._asking = self._asking[-15:] + [(list(prompt), session, system)]
        if session is None or system is None or not self.message_starts:
            return
        for kind, lru in self._lru._lrus.items():
            if kind == "system":
                continue
            for m, key in list(lru):
                if m != model or self._sessions.get(tuple(key)) != (session, system) or len(key) <= system:
                    continue
                if key[:system] != list(prompt[:system]):
                    continue
                first = next((i for i in range(system, len(key)) if key[i] in self.message_starts), len(key))
                if system + _first_difference(key, list(prompt), system, system) < first:
                    self._ended.add(tuple(key))

    def _tag_session(self, tokens):
        """The session of the request whose turn (or a snapshot of whose prompt) these tokens are."""
        key = list(tokens)
        for prompt, session, system in reversed(self._asking):
            if session is not None and system is not None and key[:len(prompt)] == prompt:
                self._sessions[tuple(tokens)] = (session, system)
                break
        if len(self._sessions) > 64:  # forget entries no longer cached
            live = {tuple(k) for lru in self._lru._lrus.values() for _, k in lru}
            self._sessions = {k: v for k, v in self._sessions.items() if k in live or k == tuple(tokens)}

    def _drop_ended(self, model):
        """Drops what note_session found ended, without saving it. A request that started over from one may have
        resumed from its attention caches as views (see _from_checkpoint); it holds what it took."""
        if not self._ended:
            return
        gb = 1 << 30
        for kind, lru in self._lru._lrus.items():
            for m, entry_key in list(lru):
                if m != model or tuple(entry_key) not in self._ended:
                    continue
                self._trie.taking = True  # dropped, not evicted: never written
                try:
                    entry = self._trie.pop(m, entry_key)
                finally:
                    self._trie.taking = False
                self._lru.remove(m, entry_key)
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._turn_state.pop(tuple(entry_key), None)
                logging.info("memory cache: dropped a %d-token %s (%.1f GB) that its session started over from (a "
                             "compaction or a clear): nothing continues it, so it isn't saved", len(entry_key),
                             entry.cache_type, entry.nbytes / gb)
        self._ended.clear()

    def _note_own_tokens(self, key, prompt, shared, places, where):
        """Logged once for each new place a conversation's history spells its reply in other tokens: each of its later
        requests lines up at the same places again."""
        si, sj = places[-1]
        seen_key = (places[0][0], tuple(key[max(0, places[0][0] - 8):places[0][0] + 8]))
        seen = self._own_seen.pop(seen_key, 0)
        self._own_seen[seen_key] = max(seen, len(places))
        while len(self._own_seen) > 64:
            self._own_seen.pop(next(iter(self._own_seen)))
        log = logging.info if len(places) > seen else logging.debug
        shown = ""
        if self.decode is not None:
            try:
                shown = (f": generated {self.decode(key[max(0, si - 12):si + 8])!r}, history "
                         f"{self.decode(prompt[max(0, sj - 12):sj + 8])!r}")
            except Exception:  # noqa: BLE001  logging only
                pass
        whole = "all" if shared == len(key) else f"{shared}"
        log("prompt cache: the request's history spells a %d-token %s entry's text in other tokens (%d place%s, the "
            "latest at token %d), so it takes the entry's there and shares %s of them%s", len(key), where,
            len(places), "" if len(places) == 1 else "s", si, whole, shown)

    def _show_divergence(self, generated, rendered, at: int, names=("generated", "history")) -> str:
        if self.decode is None:
            return ""
        try:
            show = lambda ids: repr(self.decode(list(ids[max(0, at - 12):at + 8])))  # noqa: E731
            return f": {names[0]} {show(generated)}, {names[1]} {show(rendered)}"
        except Exception:  # noqa: BLE001  logging only
            return ""

    def _rebuild_diverged(self, model, tokens, have: int):
        """The finished turn of this conversation, rebuilt to where its state was kept (its history's end, or its
        prompt's), when the request parts from it after that point and the rebuild covers more than `have` tokens."""
        key = list(tokens)
        for entry_key, (n, states) in list(self._turn_state.items()):
            if not (have < n < len(key)) or list(entry_key[:n]) != key[:n] or list(entry_key) == key[:len(entry_key)]:
                continue
            try:
                entry = self._trie.get(model, list(entry_key))
            except Exception:  # noqa: BLE001  no longer cached: taken, evicted or dropped
                self._turn_state.pop(entry_key, None)
                continue
            if len(states) != len(entry.prompt_cache) or not all(
                    _can_put_back(layer, state, n) for layer, state in zip(entry.prompt_cache, states)):
                continue  # a cache type that can't be cut back: the disk path covers it
            self._trie.taking = True
            try:
                self._trie.pop(model, list(entry_key))
            finally:
                self._trie.taking = False
            self._lru.remove(model, list(entry_key))
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            self._turn_state.pop(entry_key, None)
            for layer, state in zip(entry.prompt_cache, states):
                _put_back(layer, state, n)
            at = next((i for i, (a, b) in enumerate(zip(entry_key, key)) if a != b), n)
            logging.info("rebuilt a %d-token cache in memory from a %d-token turn the request parts from at token %d%s",
                         n, len(entry_key), at, self._show_divergence(entry_key, key, at))
            return entry.prompt_cache, key[n:]
        return None

    def fetch_nearest_cache(self, model, tokens):
        self._resuming = None
        self._fetches += 1
        self._touch(tokens)
        cache, rest = self._fetch_nearest(model, tokens)
        self._drop_ended(model)
        resuming, self._resuming = self._resuming, None
        if resuming is not None and resuming[0] is cache:  # a checkpoint was the best there was
            logging.info("memory cache: resumed at token %d from a checkpoint and a %d-token %s, which the request "
                         "leaves at token %d%s", *resuming[2:])
        if self.kv_bytes_per_token:
            # Reading the prompt's unread tokens, plus a growth block (the cache grows in blocks, and the reply grows it
            # too), peaks above the cache they build, by the same rate the context limit allows for.
            self._pending_bytes = int(PEAK_PER_KV_BYTE * (len(rest) + KV_GROWTH_TOKENS) * self.kv_bytes_per_token)
            self._read_peak = self._pending_bytes
        self._plan_checkpoints(tokens, len(tokens) - len(rest))
        return cache, rest

    def trim_to(self, *, n_sequences=None, n_bytes=None):
        """mlx-lm trims saved entries to the budget minus what active sequences hold right after a request takes its
        cache, before the prompt is read. What reading that prompt is about to need counts too: without it, a cold
        53k-token prompt built 3.2 GB of cache beside 3.5 GB of saved ones and peaked at 36.9 of 37.4 GB on a 48 GB
        Mac. Room for only the cache it builds isn't enough either: a cold 100k-token prompt beside 9.7 GB of saved
        conversations ran a 64 GB Mac's GPU out of memory before it finished reading. Checkpoints count too."""
        pending, self._pending_bytes = self._pending_bytes, 0
        self._fit_checkpoints()
        before = set(self._entry_keys())
        super().trim_to(n_sequences=n_sequences,
                        n_bytes=None if n_bytes is None else n_bytes - pending - self._checkpoint_bytes())
        gone = before - set(self._entry_keys())
        if gone and pending:
            gb = 1 << 30
            logging.info("memory cache: making room for the %.1f GB reading a prompt needs evicted %s; %d left, %.1f of "
                         "%.1f GB", pending / gb, ", ".join(f"{n}-token {kind}" for kind, n in sorted(gone)),
                         len(self), self.nbytes / gb, self.max_bytes / gb)

    def _checkpoint_bytes(self) -> int:
        return sum(c.nbytes for c in self._checkpoints.values())

    def _touch(self, tokens):
        """The checkpoints this request's prompt runs along are recently used, in the order they come in it."""
        key = tuple(tokens)
        for k in sorted((k for k in self._checkpoints if len(k) < len(key) and key[:len(k)] == k), key=len):
            c = self._checkpoints.pop(k)
            c.touched = self._fetches
            self._checkpoints[k] = c

    def _victim(self, newest) -> tuple:
        """The checkpoint to drop when they hold more than their share. Those along the prompt the newest was taken
        from stay while there are others; of the others, those the least recent request ran along go first (an older
        version of a system prompt, a conversation that ended). Within either, see _thinnest."""
        if newest not in self._checkpoints:
            newest = next(reversed(self._checkpoints))
        along, others = [], []
        for k in self._checkpoints:
            (along if newest[:len(k)] == k else others).append(k)
        if not others:
            return self._thinnest(along)
        oldest = min(self._checkpoints[k].touched for k in others)
        return self._thinnest([k for k in others if self._checkpoints[k].touched == oldest])

    def _thinnest(self, keys: list) -> tuple:
        """Of checkpoints along one prompt, the one to drop: one taken on the grid before one taken where a request
        left the cache (that line is where its client rewrites its prompt), and of those, the one whose loss leaves
        the smallest gap, so the rest stay evenly spread. The last of each kind, which covers everything after it,
        goes last."""
        keys = sorted(keys, key=len)
        at = [0] + [len(k) for k in keys] + [math.inf]
        return min((self._checkpoints[k].learned, at[i + 2] - at[i], len(k), k) for i, k in enumerate(keys))[3]

    def _fit_checkpoints(self, newest=()):
        """Checkpoints go, one _victim at a time, while they hold more than their share of the budget."""
        while self._checkpoints and (len(self._checkpoints) > MAX_CHECKPOINTS
                                     or self._checkpoint_bytes() > self.max_bytes // CHECKPOINT_SHARE):
            self._checkpoints.pop(self._victim(newest))

    def _keep_checkpoint(self, tokens, prompt_cache):
        states = [_end_state(c) for c in prompt_cache]
        nbytes = sum(a.nbytes for a in _state_arrays(states))
        if not nbytes or not _all_kept(states):
            return  # attention only, which cuts back to any token; or state a checkpoint can't keep
        # Kept on its own: in a shared batch, a lazy slice of the batch's rows would keep every row alive.
        mx.eval(list(_state_arrays(states)))
        key = tuple(tokens)
        old = self._checkpoints.pop(key, None)
        learned = key in self._learn or (old is not None and old.learned)
        self._learn.pop(key, None)
        self._checkpoints[key] = _Checkpoint(states, nbytes, self._fetches, learned)
        self._fit_checkpoints(key)
        logging.debug("memory cache: a checkpoint at token %d (%d kept, %.0f MB)", len(key), len(self._checkpoints),
                      self._checkpoint_bytes() / 1e6)

    def _memory_reach(self, result, tokens) -> int:
        """The tokens _fetch_from_memory reuses, from the trie's search result for them."""
        if result.exact is not None:
            return len(tokens)
        short = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short and can_trim_prompt_cache(
                self._trie.get(result.model, result.longer).prompt_cache):
            return min(len(tokens) - 1, result.common_prefix)
        return short

    def _from_checkpoint(self, model, tokens, result):
        """The request's cache from the latest checkpoint at or before where it leaves the cached entries, when that
        reuses more than memory otherwise would: the attention layers of an entry that shares that much, cut back to
        the checkpoint (a view: the entry stays, for whatever else continues from it), and the recurrent state saved
        there."""
        if not self._checkpoints or result.exact is not None or result.longer is None:
            return None
        have = self._memory_reach(result, tokens)
        limit = min(result.common_prefix, len(tokens) - 1)
        key = max((k for k in self._checkpoints if have < len(k) <= limit and tuple(tokens[:len(k)]) == k),
                  key=len, default=None)
        if key is None:
            return None
        entry = self._trie.get(result.model, result.longer)
        n = len(key)
        resumed = _resume(entry.prompt_cache, self._checkpoints[key].states, n)
        if resumed is None:
            return None
        # Counted as used (and logged) only if nothing else covers more: see fetch_nearest_cache.
        self._resuming = (resumed, key, n, len(result.longer), entry.cache_type, result.common_prefix,
                          self._show_divergence(result.longer, tokens, result.common_prefix, ("cached", "request")))
        return resumed, list(tokens[n:])

    def _piece(self, token: int) -> str:
        if token not in self._pieces:
            self._pieces[token] = self.decode([token])
        return self._pieces[token]

    def _line_start(self, tokens, at: int, floor: int) -> int:
        """Where the line holding token `at` starts, so that a later change anywhere in that line still finds a
        checkpoint taken there; `at` itself when no line break is within LINE_BACKOFF tokens after `floor`."""
        if self.decode is None:
            return at
        try:
            for i in range(at, max(floor, at - LINE_BACKOFF), -1):
                if "\n" in self._piece(tokens[i - 1]):
                    return i
        except Exception:  # noqa: BLE001  a token the tokenizer can't decode alone: the exact point will do
            pass
        return at

    def _plan_checkpoints(self, tokens, have: int):
        """Cut the request's segments where checkpoints are to be taken as it is read: every checkpoint_every tokens
        of its system region (its system prompt and tools, where a request's first change usually is; not further,
        where they would hold a conversation's state as of one version of its system prompt, made again each time
        that changes), and at the start of the line where it left the cached entries, if that was well past what it
        reuses: a client that rewrites a line once tends to rewrite it again. Never at its last token: that one is read
        as generation starts. Also where its history ends, for the state a rebuild resumes from (see _history_end), and
        where messages the server added to it start (`hold`): the client never sees them, so its next request parts
        there, and without a state to resume from it went back to the system prompt (33,765 tokens on Inkling 2 Mini)."""
        tokenized, self.tokenized = self.tokenized, None
        hold, self.hold = self.hold, None
        if self.checkpoint_every is None or tokenized is None or tokenized[0] is not tokens:
            return
        every, end = self.checkpoint_every, len(tokens) - 1
        first = min(self.checkpoint_span, end - 1)
        if self.system_ends:
            first = min(first, next((i for i, t in enumerate(tokens[:first]) if t in self.system_ends), first))
        at = set(range((have // every + 1) * every, first + 1, every))
        if hold is not None:
            at.add(hold)
        leaving, leaves = self._leaving
        if leaves - have >= self.checkpoint_min_gain:
            line = self._line_start(tokens, leaves, have)
            at.add(line)
            if have < line < end:
                self._learn = {k: None for k in list(self._learn)[-7:]}  # a few requests' worth, if never taken
                self._learn[tuple(tokens[:line])] = None
            logging.info("memory cache: the request leaves a cached %d-token entry at token %d and reuses %d; a "
                         "checkpoint at token %d covers that line next time%s", len(leaving), leaves, have, line,
                         self._show_divergence(leaving, tokens, leaves, ("cached", "request")))
        history = self._history_end(tokens, have)
        if history is not None:
            self._history_ends = dict(list(self._history_ends.items())[-7:])
            self._history_ends[tuple(tokens[:history])] = len(tokens)
        _split_segments(tokenized[1], tokenized[2], [p for p in at if have < p < end], history)

    def _history_end(self, tokens, have: int) -> int | None:
        """Where the request's history ends: the start of the turn it asks for, the last turn header near its end. A
        compaction asks for a summary in place of that turn, and a retry or an interrupted turn changes it too, so they
        share the request's history up to here and no further. A hybrid model's state can't be cut back to where they
        leave it, and kept only at the prompt's end (a few tokens past) it made Claude Code's compaction read its
        conversation again from the system prompt: 73,537 tokens, 20 minutes, on Aris 3.8-27B. None without such a
        header near the end, or with a history too short to be worth a state."""
        if not self.turn_starts:
            return None
        for i in range(len(tokens) - 2, max(have, len(tokens) - 1 - HISTORY_LOOKBACK), -1):
            if tokens[i] in self.turn_starts:
                return i if i >= self.light_min_tokens else None
        return None

    def _fetch_nearest(self, model, tokens):
        result = self._trie.search(model, tokens)
        # How far the request runs along a cached entry, before any is taken: where checkpoints are planned for.
        self._leaving = ((), len(tokens)) if result.exact is not None else (result.longer or (), result.common_prefix)
        cache, rest = self._from_checkpoint(model, tokens, result) or self._fetch_from_memory(model, tokens)
        rebuilt = self._rebuild_diverged(model, tokens, len(tokens) - len(rest))
        if rebuilt is not None:
            return rebuilt
        if self.disk is None:
            return cache, rest
        # Memory holds what fits beside the weights; on a small Mac that can be less than one long conversation plus
        # the system prompt, so a turn finds only the system prompt. Reading the turn's cache back from disk takes a
        # second or two; re-reading the conversation takes minutes.
        found = self.disk.nearest(tokens, longer_than=len(tokens) - len(rest) + DISK_MIN_GAIN)
        if found is None:
            return cache, rest
        data, n = found
        t0 = time.time()
        del cache
        self._drop_diverged(model, tokens, n)
        # An evicted turn's arrays can outlive their last reference until the next collection; with a long context
        # that is a second conversation's worth of memory under the one being loaded.
        gc.collect()
        mx.clear_cache()
        before = mx.get_active_memory()
        loaded = self.disk.load(data)
        if loaded is None:
            # Dropped from disk, so it can't come up again: look again. Memory still has the system prompt, say;
            # giving up here sent the turn back to its first token, minutes for a long conversation.
            return self._fetch_nearest(model, tokens)
        gb = 1 << 30
        logging.info("loaded a %d-token cache from disk in place of %d cached tokens (%.1fs, active %.1f -> %.1f GB)",
                     n, len(tokens) - len(rest), time.time() - t0, before / gb, mx.get_active_memory() / gb)
        return loaded, list(tokens[n:])

    def _drop_diverged(self, model, tokens, shared: int):
        """Entries of this conversation that run past `shared` tokens and then differ from the request: a finished
        turn whose re-rendered history tokenized differently from what was generated, say. Nothing will continue
        them, and kept beside the copy loaded from disk they double the conversation in memory."""
        key = list(tokens)
        for kind, lru in self._lru._lrus.items():
            if kind == "system":
                continue
            for m, entry_key in list(lru):
                if m != model or len(entry_key) <= shared or entry_key[:shared] != key[:shared]:
                    continue
                if entry_key == key[:len(entry_key)]:
                    continue  # still a prefix: not diverged
                self._trie.taking = True  # dropped, not evicted: never written
                try:
                    entry = self._trie.pop(m, entry_key)
                finally:
                    self._trie.taking = False
                self._lru.remove(m, entry_key)
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                logging.info("memory cache: dropped a %d-token %s that diverged from its conversation at or after "
                             "token %d", len(entry_key), entry.cache_type, shared)

    def _fetch_from_memory(self, model, tokens):
        result = self._trie.search(model, tokens)
        short = len(result.shorter) if result.shorter is not None else 0
        if result.exact is None:
            if result.longer is not None and result.common_prefix > short:
                longer = self._trie.get(result.model, result.longer)
                if can_trim_prompt_cache(longer.prompt_cache):
                    return super().fetch_nearest_cache(model, tokens)
            if not short:
                return None, tokens
        key = result.exact if result.exact is not None else result.shorter
        entry = self._trie.get(result.model, key)
        if entry.cache_type == "system":
            self._lru.touch(result.model, key)  # copied, and stays for the next conversation that starts from it
            return super().fetch_nearest_cache(model, tokens)
        self._trie.taking = True
        try:
            self._trie.pop(result.model, key)
        finally:
            self._trie.taking = False
        self._turn_state.pop(tuple(key), None)  # continued as-is: it won't need rebuilding
        self._lru.remove(result.model, key)
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        return entry.prompt_cache, tokens[len(key):]


def shrink_cap(cap: int, peak: int, limit: int, headroom: int, floor: int = 1 << 30) -> int:
    over = peak - (limit - headroom)
    return max(floor, cap - over) if over > 0 else cap


def turn_start_tokens(tokenizer) -> frozenset:
    """The token that opens the turn a request asks the model for: the first one its chat template adds as the
    generation prompt (Qwen's <|im_start|>). Empty when the template adds none, as Inkling's: its prompt ends where its
    history does, so the state kept at the prompt's end already serves a compaction."""
    messages = [{"role": "user", "content": "hi"}]

    def ids(asking: bool) -> list:
        # mlx-lm's tokenizer returns the tokens; a bare transformers 5 tokenizer returns a dict of them.
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=asking, tokenize=True)
        return list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)

    try:
        bare, asking = ids(False), ids(True)
    except Exception:  # noqa: BLE001  a template that needs more than this: no history cuts, as before
        return frozenset()
    if len(asking) > len(bare) and asking[:len(bare)] == bare and isinstance(asking[len(bare)], int):
        return frozenset({asking[len(bare)]})
    return frozenset()


def _byte_alphabet() -> dict[str, int]:
    """GPT-2's byte-level BPE alphabet: the character a vocabulary writes each byte as -> that byte."""
    shown = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, extra = list(shown), 0
    for b in range(256):
        if b not in shown:
            shown.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(shown, chars)}


def token_bytes(tokenizer) -> Callable[[int], bytes] | None:
    """A token's bytes, for telling whether two tokenizations spell one text byte for byte: decoding can't say, for a
    token that holds part of a character. For byte-level BPE (GPT-2's alphabet, which Qwen's and Inkling's tokenizers
    use); None for any other kind, or one whose decoding doesn't give back exactly these bytes."""
    alphabet = _byte_alphabet()
    try:
        added = {int(i): t.content for i, t in tokenizer.added_tokens_decoder.items()}
    except Exception:  # noqa: BLE001  no table of added tokens: not a tokenizer this knows
        return None
    known: dict[int, bytes] = {}

    def of(token: int) -> bytes:
        b = known.get(token)
        if b is None:
            content = added.get(token)
            b = content.encode() if content is not None else bytes(
                alphabet[c] for c in tokenizer.convert_ids_to_tokens(token))
            known[token] = b
        return b

    try:
        check = BYTES_CHECK + (tokenizer.eos_token or "")
        ids = list(tokenizer.encode(check, add_special_tokens=False))
        if not ids or b"".join(of(t) for t in ids) != tokenizer.decode(ids).encode():
            return None
    except Exception:  # noqa: BLE001  a character outside the alphabet, say: not byte-level BPE
        return None
    return of


# Code, accents, CJK, an emoji split over tokens, and the spaces a decoder's clean-up would remove.
BYTES_CHECK = "def f(x):\n\treturn {'a': x[0], \"b\": 1.5e-3}  # ünïcödé ✓ 中文 🙂 , . 's n't\n\n"


def _first_difference(a: list, b: list, i: int, j: int, chunk: int = 4096) -> int:
    """How many tokens a[i:] and b[j:] share before they first differ (or either ends), compared a chunk at a time."""
    n = min(len(a) - i, len(b) - j)
    k = 0
    while k < n:
        m = min(chunk, n - k)
        if a[i + k:i + k + m] == b[j + k:j + k + m]:
            k += m
            continue
        while a[i + k] == b[j + k]:
            k += 1
        return k
    return n


def align_tokens(key, prompt, start: int, of: Callable[[int], bytes]):
    """The request's tokens with a cached entry's in their place wherever the two spell the same text in other tokens,
    from `start`, where they first differ. Returns the tokens, which spell exactly what `prompt` does, how many of the
    entry's they start with, and the places the entry's tokens stand in for the request's: (entry index, request
    index) where each starts. Where the two spell different text, the entry's tokens stop."""
    key, prompt = list(key), list(prompt)
    i = j = start
    places = []
    while True:
        d = _first_difference(key, prompt, i, j)
        i, j = i + d, j + d
        if i == len(key):
            return key + prompt[j:], i, places
        if j == len(prompt):
            return key[:i], i, places
        si, sj = i, j
        a, b, checked = bytearray(), bytearray(), 0
        while True:  # each side takes tokens until both have spelled the same bytes, ending a token on each side
            if len(a) <= len(b):
                if i == len(key):
                    break
                a += of(key[i])
                i += 1
            else:
                if j == len(prompt):
                    break
                b += of(prompt[j])
                j += 1
            n = min(len(a), len(b))
            if a[checked:n] != b[checked:n] or max(len(a), len(b)) > RESYNC_BYTES:
                return key[:si] + prompt[sj:], si, places
            checked = n
            if len(a) == len(b):
                break
        if len(a) != len(b):  # one ran out inside the other's last token
            return key[:si] + prompt[sj:], si, places
        places.append((si, sj))


def appended_start(tokenize, a, k, raw_prompt, prompt) -> int | None:
    """Where the messages the server added to the end of a request (Plan.appended: a stuck-call reminder, a
    tool_choice note) start in its prompt as read, or None when it added none. The client never sees them, so its
    next request parts from this turn there. Found by tokenizing the request without them: where the two prompts
    part (a chat template's generation prompt can share a header token with the first added message), counted from
    the end, so it moves with a history read in a cached entry's own tokens (MovingPromptCache.own_tokens)."""
    request = a[1] if len(a) > 1 else k.get("request")
    n = getattr(request, "aris_appended", 0)
    if not n or len(request.messages) <= n:
        return None
    shorter = copy.copy(request)
    shorter.messages = copy.deepcopy(request.messages[:-n])  # mlx-lm's tokenize may rewrite a message's content
    if len(a) > 1:
        prefix = tokenize(a[0], shorter, *a[2:], **k)[0]
    else:
        prefix = tokenize(*a, **{**k, "request": shorter})[0]
    shared = next((i for i, (x, y) in enumerate(zip(raw_prompt, prefix)) if x != y), min(len(raw_prompt), len(prefix)))
    added = len(raw_prompt) - shared
    return len(prompt) - added if 0 < added < len(prompt) else None


def merge_prompt_tail(tokenized):
    """mlx-lm splits a chat prompt into system / conversation / thinking-tail segments and saves a copy of the cache
    at the end of each but the last. The conversation snapshot is a second full copy of a long context, held through
    the turn and redundant: the finished turn's cache, which the next turn continues, is saved anyway. Folding the
    conversation into the last segment keeps only the shared system-prompt snapshot."""
    prompt, segments, types, initial_state, vision = tokenized
    if len(segments) >= 2 and types[-2] == "user":
        segments = segments[:-2] + [segments[-2] + segments[-1]]
        types = types[:-2] + ["assistant"]
    return prompt, segments, types, initial_state, vision


def model_bytes(model) -> int:
    """Bytes of a loaded model's weights, with its MTP head if one is attached (kept beside the model's own
    parameters, so they don't count it)."""
    total = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    head = getattr(model, "_mtp_head", None)
    if head is not None:
        # A DSpark drafter also keeps tables derived from the target's output layer (Drafter.nbytes).
        total += getattr(head, "nbytes", None) or sum(v.nbytes for _, v in tree_flatten(head.parameters()))
        total += getattr(getattr(model, "_mtp_served", None), "extra_bytes", 0)  # an MTP head's draft output layer
    return total


@dataclass
class Piece:
    state: str          # "reasoning" | "normal" | "tool", or inkling.MESSAGE (no text) where an Inkling message starts
    text: str


@dataclass
class Run:
    prompt_tokens: int
    cached_tokens: int
    tool_parser: Callable
    pieces: object       # iterator of (list[Piece], finish_reason | None, n_tokens)
    stop: Callable[[], None]
    budget: list         # [ThinkBudget] once generation has started, if a budget applies
    guard: list = field(default_factory=list)  # [LoopGuard] once generation has started, unless guards are off
    mtp_start: tuple | None = None  # MTP (rounds, drafted, accepted) when the request began, with an MTP head
    exact_text: bool = False  # thinking and text go to the client untrimmed (stream.MessageBuilder's `exact`)


class Engine:
    def __init__(self, spec: ModelSpec, decode_concurrency: int = 4, prompt_concurrency: int = 2,
                 prompt_cache_size: int = 8, prompt_cache_bytes: int | None = None, reserve_gb: float = 6.0,
                 disk_cache_gb: float = 20.0, guards: bool = True):
        self.spec = spec
        self.guards = guards
        ns = argparse.Namespace(
            model=spec.path, adapter_path=spec.adapter, draft_model=None, num_draft_tokens=3, trust_remote_code=False,
            chat_template="", use_default_chat_template=False, temp=spec.temperature, top_p=spec.top_p,
            top_k=spec.top_k, min_p=0.0, max_tokens=32000, chat_template_args={},
            decode_concurrency=decode_concurrency, prompt_concurrency=prompt_concurrency, prefill_step_size=512,
            prompt_cache_size=prompt_cache_size, prompt_cache_bytes=prompt_cache_bytes, kv_bits=None,
            kv_group_size=64, quantized_kv_start=DEFAULT_QUANTIZED_KV_START, pipeline=False, allowed_origins=["*"],
        )
        self.provider = mls.ModelProvider(ns)
        self.provider._mtp_spec = spec if spec.mtp else None  # mtp_serve attaches the head on the loading thread
        cache = MovingPromptCache(prompt_cache_size, prompt_cache_bytes or (1 << 63))
        # The generator thread loads the model, so all MLX work stays on one thread and stream.
        self.generator = mls.ResponseGenerator(self.provider, cache)
        tokenize = self.generator._tokenize

        def tokenize_and_track(*a, **k):
            # Runs on the generation thread before the prompt is read, so a mid-request snapshot can't beat it.
            raw = tokenize(*a, **k)
            tokenized = cache.own_tokens(self.provider.model_key, merge_prompt_tail(raw))
            request = a[1] if len(a) > 1 else k.get("request")
            cache.note_session(self.provider.model_key, tokenized, getattr(request, "aris_session", None))
            cache.reading = cache.reading[-15:] + [tokenized[0]]
            cache.tokenized = tokenized  # fetching its cache next cuts its segments where checkpoints go
            cache.hold = appended_start(tokenize, a, k, raw[0], tokenized[0])
            return tokenized

        self.generator._tokenize = tokenize_and_track
        while self.provider.tokenizer is None:
            if not self.generator.generation_available():
                raise RuntimeError(f"could not load {spec.path}")
            time.sleep(0.2)
        self.tokenizer = self.provider.tokenizer
        # mlx-lm sets the tokenizer as it finishes loading; the MTP head is attached right after, on that thread.
        while spec.mtp and getattr(self.provider.model, "_mtp_head", None) is None:
            if not self.generator.generation_available():
                raise RuntimeError(f"could not load the MTP head {spec.mtp}")
            time.sleep(0.1)
        # Before any request makes a cache, and after the MTP head adds its cache slot.
        if spec.kv_bits and not quantize_kv(self.provider.model, spec.kv_bits):
            # Its attention keeps its caches inside its own layer caches (Inkling) and reads them itself.
            logging.warning("%s keeps its KV cache at full precision: kv_bits = %d doesn't apply to it", spec.name,
                            spec.kv_bits)
            self.spec = spec = replace(spec, kv_bits=None)
        self.inkling = inkling.is_inkling(self.tokenizer)
        if _checkpointable(self.provider.model.make_cache()):
            # Recurrent layers, or sliding windows: a cache can't be cut back without checkpoints (see _resume).
            cache.checkpoint_every = CHECKPOINT_GRID
            cache.system_ends = inkling.conversation_starts(self.tokenizer) if self.inkling \
                else frozenset(self.tokenizer.eos_token_ids or ())
            cache.turn_starts = turn_start_tokens(self.tokenizer)
        _OWN_TOOL_CALLS.active = self.inkling
        cache.message_starts = inkling.conversation_starts(self.tokenizer) if self.inkling \
            else turn_start_tokens(self.tokenizer)
        cache.decode = self.tokenizer.decode
        cache.token_bytes = token_bytes(self.tokenizer)
        if cache.token_bytes is None:
            logging.info("prompt cache: this tokenizer's tokens can't be compared byte for byte, so a reply its next "
                         "request tokenizes differently is read again from where they part")
        self.budget_tokens = BudgetTokens(self.tokenizer)
        self.loop_exempt = loop_exempt_ids(self.tokenizer)
        self.cache = cache
        self.generator.on_stop = cache.flush
        try:
            self.max_context = self._context_that_fits()
        except NotEnoughMemory:
            self.generator.stop_and_join()  # its thread holds the loaded model
            raise
        self._fit_memory(cache, ns, prompt_cache_bytes, reserve_gb)
        if disk_cache_gb > 0:
            disk = DiskCache(home() / "cache", spec.path, int(disk_cache_gb * (1 << 30)), spec.adapter, spec.kv_bits,
                             spec.mtp, spec.mtp_bits)
            key = self.provider.model_key
            restored = disk.restore(lambda t, c, kind: LRUPromptCache.insert_cache(cache, key, t, c, cache_type=kind),
                                    cache.max_bytes)
            cache.disk = disk
            if restored:
                logging.info("restored %.1f GB of prompt caches from disk", restored / (1 << 30))
        threading.Thread(target=self._watch, daemon=True).start()
        logging.info("loaded %s%s (thinking=%s, tools=%s, think-end id=%s, KV cache %s%s)", spec.path,
                     f" + adapter {spec.adapter}" if spec.adapter else "",
                     "Inkling messages" if self.inkling else self.tokenizer.has_thinking,
                     "Inkling messages" if self.inkling else self.tokenizer.has_tool_calling, self.budget_tokens.end_id,
                     f"{spec.kv_bits}-bit" if spec.kv_bits else "full precision",
                     f", MTP head {spec.mtp} ({spec.mtp_bits or 16}-bit, {spec.mtp_depth} drafts)" if spec.mtp else "")

    def _kv_bytes_per_token(self) -> int | None:
        try:
            layers = sum(isinstance(c, (KVCache, QuantizedKVCache))
                         for c in _flat_caches(self.provider.model.make_cache()))
            path = Path(self.spec.path)
            if not (path / "config.json").is_file():
                return None
            _, heads, head_dim = _kv_shape(json.loads((path / "config.json").read_text()))
            return kv_bytes_per_token(layers, heads, head_dim, self.spec.kv_bits) if layers else None
        except Exception:  # noqa: BLE001  an unusual model: no memory-derived context limit
            return None

    def _context_that_fits(self) -> int | None:
        """A context limit this machine can hold. Past it the server answers "prompt is too long" (Claude Code then
        compacts the conversation) instead of running out of memory mid-turn."""
        weights = model_bytes(self.provider.model)
        limit = mx.device_info()["max_recommended_working_set_size"]
        per_token = self._kv_bytes_per_token()
        if per_token is None:
            check_memory(self.spec.name, limit, weights)
            return None
        self.cache.kv_bytes_per_token = per_token
        fits = context_that_fits(limit, weights, per_token, overhead=CONTEXT_OVERHEAD)
        logging.info("memory: about %d tokens of context fit (%d KB of KV cache per token; model allows %s)", fits,
                     per_token // 1024, self.spec.context or "unknown")
        check_memory(self.spec.name, limit, weights, fits)
        return fits

    def _fit_memory(self, cache, ns, prompt_cache_bytes, reserve_gb):
        """Size the prompt cache and MLX's buffer pool to what the GPU can hold beside the weights: unbounded,
        saved prompt caches and freed buffers grow until Metal runs out of memory mid-generation."""
        gb = 1 << 30
        weights = model_bytes(self.provider.model)
        limit = mx.device_info()["max_recommended_working_set_size"]
        spare = limit - weights - int(reserve_gb * gb)
        # The cap covers saved caches plus the caches active sequences hold (mlx-lm trims the LRU to the cap minus the
        # active bytes). Too small, and a long conversation is evicted after every turn and re-read from scratch; the
        # reserve covers prefill's attention scratch, which grows with context length.
        if prompt_cache_bytes is None:
            prompt_cache_bytes = max(spare, gb // 2)
        cache.max_bytes = ns.prompt_cache_bytes = prompt_cache_bytes
        mx.set_cache_limit(max(min(spare // 8, gb), gb // 4))
        logging.info("memory: GPU limit %.1f GB, weights %.1f GB, prompt cache up to %.1f GB, %.1f GB reserved for "
                     "active sequences", limit / gb, weights / gb, prompt_cache_bytes / gb, reserve_gb)
        if spare < gb:
            logging.warning("this model leaves under 1 GB beside its weights; long contexts may run out of memory")

    def _watch(self):
        """mlx-lm's generation thread can't recover from a Metal failure (out of memory, most often). A server that
        stays up after it dies only fails every request, so exit and let whatever started it start it again."""
        while self.generator.generation_available():
            time.sleep(1)
        logging.critical("the generation engine stopped (see the error above); exiting")
        os._exit(70)

    def adapt_memory(self, headroom_gb: float = 2.5):
        """After a request: if memory came within `headroom_gb` of the GPU limit, shrink the prompt cache by the
        overshoot. A fixed reserve can't be right for every Mac, model and context length; this settles on what the
        machine actually needs after its first long session."""
        gb = 1 << 30
        peak, limit = mx.get_peak_memory(), mx.device_info()["max_recommended_working_set_size"]
        mx.reset_peak_memory()
        cap = self.cache.lower_budget(lambda cap: shrink_cap(cap, peak, limit, int(headroom_gb * gb)))
        if cap is not None:
            self.provider.cli_args.prompt_cache_bytes = cap
            logging.warning("memory peaked at %.1f of %.1f GB; prompt cache lowered to %.1f GB", peak / gb, limit / gb,
                            cap / gb)

    @staticmethod
    def memory() -> str:
        gb = 1 << 30
        return (f"mem active={mx.get_active_memory() / gb:.1f}G peak={mx.get_peak_memory() / gb:.1f}G "
                f"pool={mx.get_cache_memory() / gb:.1f}G")

    def count_tokens(self, plan: Plan) -> int:
        return len(self.tokenizer.apply_chat_template(plan.messages, tools=plan.tools, add_generation_prompt=True,
                                                      tokenize=True, **plan.template_kwargs))

    def run(self, plan: Plan, on_progress: Callable[[int, int], None] | None = None) -> Run:
        budget: list[ThinkBudget] = []
        guard: list[LoopGuard] = []

        def extra():
            procs = []
            if self.guards:
                # Ahead of the budget: a token the budget forces replaces the logits whole, so it stays possible.
                guard.append(LoopGuard(self.loop_exempt))
                procs.append(guard[-1])
            b = self.budget_tokens.make(plan.think_budget)
            if b is not None:
                budget.append(b)
                procs.append(b)
            return procs

        args = _Args(
            model=mls.ModelDescription(model="default_model", draft="default_model", adapter=None),
            sampling=mls.SamplingArguments(temperature=plan.temperature, top_p=plan.top_p, top_k=plan.top_k,
                                           min_p=0.0, xtc_probability=0.0, xtc_threshold=0.0),
            logits=mls.LogitsProcessorArguments(logit_bias=None, repetition_penalty=0.0, repetition_context_size=20,
                                                presence_penalty=0.0, presence_context_size=20,
                                                frequency_penalty=0.0, frequency_context_size=20),
            stop_words=plan.stop, max_tokens=plan.max_tokens, num_draft_tokens=3, logprobs=False, top_logprobs=0,
            seed=None, chat_template_kwargs=plan.template_kwargs,
            extra_processors=extra,
        )
        request = mls.CompletionRequest(request_type="chat", prompt="", messages=plan.messages, tools=plan.tools,
                                        role_mapping=None)
        request.aris_session = getattr(plan, "session", None)  # read back by the tokenize hook (note_session)
        request.aris_appended = getattr(plan, "appended", 0)  # and this (appended_start)
        ctx, responses = self.generator.generate(request, args, progress_callback=on_progress)

        classify = _classify_inkling if self.inkling else _classify

        def pieces():
            try:
                yield from classify(ctx, responses, lambda: bool(guard) and guard[0].gave_up)
            finally:
                self.cache.reading = [p for p in self.cache.reading if p != ctx.prompt]

        return Run(prompt_tokens=len(ctx.prompt), cached_tokens=max(ctx.prompt_cache_count, 0),
                   tool_parser=inkling.parse_tool_calls if self.inkling else ctx.tool_parser, pieces=pieces(),
                   stop=ctx.stop, budget=budget, guard=guard, mtp_start=mtp_snapshot(self.provider.model),
                   exact_text=self.inkling)


def _classify(ctx, responses, gave_up: Callable[[], bool] = lambda: False):
    """Split generated text into reasoning / normal / tool pieces with mlx-lm's state machine, which also strips
    the control markers (<think>, <tool_call> ...). A response the loop guard gave up on ends here as if it had hit
    the token cap: generation stops, and the client sees a "length" finish."""
    sm = ctx.text_sm.make_state(ctx.initial_state)
    n = 0
    for gen in responses:
        n += 1
        if gen.finish_reason == "stop":
            sm, state = TextStateMachine.discard(sm)
            segments = [(state, "")]
        elif gen.finish_reason == "length" or gave_up():
            sm, segments = TextStateMachine.split(sm, gen.text)
            sm, flushed, state = TextStateMachine.flush(sm)
            segments.append((state, flushed))
            if gen.finish_reason != "length":
                ctx.stop()
                yield [Piece(s, t) for s, t in segments], "length", n
                return
        else:
            sm, segments = TextStateMachine.split(sm, gen.text)
        yield [Piece(s, t) for s, t in segments], gen.finish_reason, n


def _classify_inkling(ctx, responses, gave_up: Callable[[], bool] = lambda: False):
    """_classify for Inkling's reply messages (inkling.py): thinking, text and JSON tool calls, markers removed."""
    stream = inkling.ReplyStream()
    n = 0
    for gen in responses:
        n += 1
        if gen.finish_reason == "stop":
            stream.discard()
            segments = [(stream.state, "")]
        elif gen.finish_reason == "length" or gave_up():
            segments = stream.feed(gen.text) + stream.flush()
            if not segments:
                segments = [(stream.state, "")]
            if gen.finish_reason != "length":
                ctx.stop()
                yield [Piece(s, t) for s, t in segments], "length", n
                return
        else:
            segments = stream.feed(gen.text)
        yield [Piece(s, t) for s, t in segments], gen.finish_reason, n
