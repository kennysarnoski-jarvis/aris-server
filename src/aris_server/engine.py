"""mlx-lm's server engine (batching, prompt cache, thinking and tool parsing), driven by Aris Server."""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from mlx_lm import server as mls
from mlx_lm.generate import DEFAULT_QUANTIZED_KV_START, TextStateMachine
from mlx_lm.models.cache import LRUPromptCache

from .budget import BudgetTokens, ThinkBudget
from .config import ModelSpec
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
                 prompt_cache_size: int = 8, prompt_cache_bytes: int | None = None):
        self.spec = spec
        ns = argparse.Namespace(
            model=spec.path, adapter_path=None, draft_model=None, num_draft_tokens=3, trust_remote_code=False,
            chat_template="", use_default_chat_template=False, temp=spec.temperature, top_p=spec.top_p,
            top_k=spec.top_k, min_p=0.0, max_tokens=32000, chat_template_args={},
            decode_concurrency=decode_concurrency, prompt_concurrency=prompt_concurrency, prefill_step_size=2048,
            prompt_cache_size=prompt_cache_size, prompt_cache_bytes=prompt_cache_bytes, kv_bits=None,
            kv_group_size=64, quantized_kv_start=DEFAULT_QUANTIZED_KV_START, pipeline=False, allowed_origins=["*"],
        )
        self.provider = mls.ModelProvider(ns)
        cache = LRUPromptCache(prompt_cache_size, prompt_cache_bytes or (1 << 63))
        # The generator thread loads the model, so all MLX work stays on one thread and stream.
        self.generator = mls.ResponseGenerator(self.provider, cache)
        while self.provider.tokenizer is None:
            if not self.generator.generation_available():
                raise RuntimeError(f"could not load {spec.path}")
            time.sleep(0.2)
        self.tokenizer = self.provider.tokenizer
        self.budget_tokens = BudgetTokens(self.tokenizer)
        logging.info("loaded %s (thinking=%s, tools=%s, think-end id=%s)", spec.path, self.tokenizer.has_thinking,
                     self.tokenizer.has_tool_calling, self.budget_tokens.end_id)

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
        return Run(prompt_tokens=len(ctx.prompt), cached_tokens=max(ctx.prompt_cache_count, 0),
                   tool_parser=ctx.tool_parser, pieces=_classify(ctx, responses), stop=ctx.stop, budget=budget)


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
