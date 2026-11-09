"""mlx-lm's server engine (batching, prompt cache, thinking and tool parsing), driven by Aris Server."""

import argparse
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import server as mls
from mlx_lm.generate import DEFAULT_QUANTIZED_KV_START, PromptProcessingBatch, TextStateMachine
from mlx_lm.models.cache import LRUPromptCache, can_trim_prompt_cache

from .budget import BudgetTokens, ThinkBudget
from .config import ModelSpec, home
from .disk_cache import DiskCache
from .translate import Plan


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


DISK_MIN_GAIN = 1024  # fewer extra tokens than this are as quick to read as the cache file


class MovingPromptCache(LRUPromptCache):
    """mlx-lm copies a saved cache for the request that continues it, so a long conversation sits in memory twice
    for the whole turn (2.4 GB extra at 37k tokens on a 27B model, enough to run a 48 GB Mac out of memory). A
    conversation's cache is only ever continued by its next turn, so it moves to that request instead; the shared
    system-prompt segment is still copied for every conversation that starts from it."""

    disk = None  # DiskCache: entries are also written to disk as they come in

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reading: list[list[int]] = []  # prompts of the requests in flight

    def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        # A snapshot of part of a prompt still being answered: the request holds its own copy, so keeping this one
        # in memory too doubles the conversation (3.5 GB more at 51k tokens) and ran a 48 GB Mac out of memory.
        # Disk is enough for the retry it exists for. The shared system prompt stays in memory.
        mid_request = cache_type != "system" and any(
            len(tokens) < len(p) and p[:len(tokens)] == list(tokens) for p in self.reading)
        if not mid_request:
            before = {k for k in self._entry_keys()}
            # The turn that just finished is the cache its conversation's next turn continues, often within seconds:
            # when it's larger than the budget, other entries go and it stays. Evicting it sent that next turn to
            # disk while mlx-lm still briefly held the evicted copy, putting two copies of the conversation in memory.
            cap, size = self.max_bytes, sum(c.nbytes for c in prompt_cache)
            keep = size + self._n_bytes_by_type.get("system", 0)
            if cache_type != "system" and keep > cap:
                self.max_bytes = keep
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
        if self.disk is not None:
            try:
                self.disk.save(tokens, prompt_cache, cache_type)
            except OSError as e:  # a full disk must not stop generation
                logging.warning("could not save a prompt cache to disk: %s", e)

    def _entry_keys(self):
        """(type, length) of each entry, for logging what an insert evicted."""
        return [(kind, len(key)) for kind, lru in self._lru._lrus.items() for _, key in lru] \
            if hasattr(self._lru, "_lrus") else []

    def fetch_nearest_cache(self, model, tokens):
        cache, rest = self._fetch_from_memory(model, tokens)
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
        # An evicted turn's arrays can outlive their last reference until the next collection; with a long context
        # that is a second conversation's worth of memory under the one being loaded.
        gc.collect()
        mx.clear_cache()
        before = mx.get_active_memory()
        try:
            loaded = self.disk.load(data)
        except Exception as e:
            logging.warning("could not load a prompt cache from disk: %s", e)
            return None, tokens
        gb = 1 << 30
        logging.info("loaded a %d-token cache from disk in place of %d cached tokens (%.1fs, active %.1f -> %.1f GB)",
                     n, len(tokens) - len(rest), time.time() - t0, before / gb, mx.get_active_memory() / gb)
        return loaded, list(tokens[n:])

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
            return super().fetch_nearest_cache(model, tokens)
        self._trie.pop(result.model, key)
        self._lru.remove(result.model, key)
        self._n_bytes -= entry.nbytes
        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        return entry.prompt_cache, tokens[len(key):]


def shrink_cap(cap: int, peak: int, limit: int, headroom: int, floor: int = 1 << 30) -> int:
    over = peak - (limit - headroom)
    return max(floor, cap - over) if over > 0 else cap


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


@dataclass
class Piece:
    state: str          # "reasoning" | "normal" | "tool"
    text: str


@dataclass
class Run:
    prompt_tokens: int
    cached_tokens: int
    tool_parser: Callable
    pieces: object       # iterator of (list[Piece], finish_reason | None, n_tokens)
    stop: Callable[[], None]
    budget: list         # [ThinkBudget] once generation has started, if a budget applies


class Engine:
    def __init__(self, spec: ModelSpec, decode_concurrency: int = 4, prompt_concurrency: int = 2,
                 prompt_cache_size: int = 8, prompt_cache_bytes: int | None = None, reserve_gb: float = 6.0,
                 disk_cache_gb: float = 20.0):
        self.spec = spec
        ns = argparse.Namespace(
            model=spec.path, adapter_path=spec.adapter, draft_model=None, num_draft_tokens=3, trust_remote_code=False,
            chat_template="", use_default_chat_template=False, temp=spec.temperature, top_p=spec.top_p,
            top_k=spec.top_k, min_p=0.0, max_tokens=32000, chat_template_args={},
            decode_concurrency=decode_concurrency, prompt_concurrency=prompt_concurrency, prefill_step_size=512,
            prompt_cache_size=prompt_cache_size, prompt_cache_bytes=prompt_cache_bytes, kv_bits=None,
            kv_group_size=64, quantized_kv_start=DEFAULT_QUANTIZED_KV_START, pipeline=False, allowed_origins=["*"],
        )
        self.provider = mls.ModelProvider(ns)
        cache = MovingPromptCache(prompt_cache_size, prompt_cache_bytes or (1 << 63))
        # The generator thread loads the model, so all MLX work stays on one thread and stream.
        self.generator = mls.ResponseGenerator(self.provider, cache)
        tokenize = self.generator._tokenize

        def tokenize_and_track(*a, **k):
            # Runs on the generation thread before the prompt is read, so a mid-request snapshot can't beat it.
            tokenized = merge_prompt_tail(tokenize(*a, **k))
            cache.reading = cache.reading[-15:] + [tokenized[0]]
            return tokenized

        self.generator._tokenize = tokenize_and_track
        while self.provider.tokenizer is None:
            if not self.generator.generation_available():
                raise RuntimeError(f"could not load {spec.path}")
            time.sleep(0.2)
        self.tokenizer = self.provider.tokenizer
        self.budget_tokens = BudgetTokens(self.tokenizer)
        self.cache = cache
        self._fit_memory(cache, ns, prompt_cache_bytes, reserve_gb)
        if disk_cache_gb > 0:
            disk = DiskCache(home() / "cache", spec.path, int(disk_cache_gb * (1 << 30)), spec.adapter)
            key = self.provider.model_key
            restored = disk.restore(lambda t, c, kind: LRUPromptCache.insert_cache(cache, key, t, c, cache_type=kind),
                                    cache.max_bytes)
            cache.disk = disk
            if restored:
                logging.info("restored %.1f GB of prompt caches from disk", restored / (1 << 30))
        threading.Thread(target=self._watch, daemon=True).start()
        logging.info("loaded %s%s (thinking=%s, tools=%s, think-end id=%s)", spec.path,
                     f" + adapter {spec.adapter}" if spec.adapter else "", self.tokenizer.has_thinking,
                     self.tokenizer.has_tool_calling, self.budget_tokens.end_id)

    def _fit_memory(self, cache, ns, prompt_cache_bytes, reserve_gb):
        """Size the prompt cache and MLX's buffer pool to what the GPU can hold beside the weights: unbounded,
        saved prompt caches and freed buffers grow until Metal runs out of memory mid-generation."""
        gb = 1 << 30
        weights = sum(v.nbytes for _, v in tree_flatten(self.provider.model.parameters()))
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
        cap = shrink_cap(self.cache.max_bytes, peak, limit, int(headroom_gb * gb))
        if cap < self.cache.max_bytes:
            self.cache.max_bytes = self.provider.cli_args.prompt_cache_bytes = cap
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

        def extra():
            b = self.budget_tokens.make(plan.think_budget)
            if b is None:
                return []
            budget.append(b)
            return [b]

        args = _Args(
            model=mls.ModelDescription(model="default_model", draft="default_model", adapter=None),
            sampling=mls.SamplingArguments(temperature=plan.temperature, top_p=plan.top_p, top_k=plan.top_k,
                                           min_p=0.0, xtc_probability=0.0, xtc_threshold=0.0),
            logits=mls.LogitsProcessorArguments(logit_bias=None, repetition_penalty=0.0, repetition_context_size=20,
                                                presence_penalty=0.0, presence_context_size=20,
                                                frequency_penalty=0.0, frequency_context_size=20),
            stop_words=plan.stop, max_tokens=plan.max_tokens, num_draft_tokens=3, logprobs=False, top_logprobs=0,
            seed=None, chat_template_kwargs=plan.template_kwargs,
            extra_processors=extra if plan.thinking else None,
        )
        request = mls.CompletionRequest(request_type="chat", prompt="", messages=plan.messages, tools=plan.tools,
                                        role_mapping=None)
        ctx, responses = self.generator.generate(request, args, progress_callback=on_progress)

        def pieces():
            try:
                yield from _classify(ctx, responses)
            finally:
                self.cache.reading = [p for p in self.cache.reading if p != ctx.prompt]

        return Run(prompt_tokens=len(ctx.prompt), cached_tokens=max(ctx.prompt_cache_count, 0),
                   tool_parser=ctx.tool_parser, pieces=pieces(), stop=ctx.stop, budget=budget)


def _classify(ctx, responses):
    """Split generated text into reasoning / normal / tool pieces with mlx-lm's state machine, which also strips
    the control markers (<think>, <tool_call> ...)."""
    sm = ctx.text_sm.make_state(ctx.initial_state)
    n = 0
    for gen in responses:
        n += 1
        if gen.finish_reason == "stop":
            sm, state = TextStateMachine.discard(sm)
            segments = [(state, "")]
        elif gen.finish_reason == "length":
            sm, segments = TextStateMachine.split(sm, gen.text)
            sm, flushed, state = TextStateMachine.flush(sm)
            segments.append((state, flushed))
        else:
            sm, segments = TextStateMachine.split(sm, gen.text)
        yield [Piece(s, t) for s, t in segments], gen.finish_reason, n
