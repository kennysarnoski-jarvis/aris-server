"""A model's own draft head inside mlx-lm's batch generator, the path Aris Server serves through.

Two kinds of head: Qwen3.8's MTP head (mtp.py) and Inkling 2's DSpark drafter (dspark.py). Attaching one gives a
loaded model extra cache slots after its own layers, holding what the head reads: the MTP head's KV cache and the
last trunk hidden state, or the drafter's attention contexts and the input of its recurrent head. Batch generation,
prompt caches and the disk cache treat them like any layer's cache, so they are padded, merged, filtered, extracted,
saved and restored with the rest. Every model call keeps them in step with the target's own caches, so the head is
ready to draft whenever a sequence is alone in the batch, whatever came before.

While a sequence is alone, GenerationBatch steps speculatively: the head drafts, one target pass checks the newest
token and the drafts, each accepted token is returned as its own Response (the server streams them in order), and
the caches are cut back to the last returned token. With two or more sequences the batch steps as plain mlx-lm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from mlx_lm import server as mls
from mlx_lm import utils as mlx_utils
from mlx_lm.generate import GenerationBatch
from mlx_lm.models.cache import ArraysCache

from . import dspark, mtp
from .speculative import Sampling, SpeculativeStats, choose

log = logging.getLogger(__name__)


@dataclass
class MTPConfig:
    depth: int = 2
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    drafter: object = None  # tests only: tokens so far -> drafts, instead of the head
    draft_override: object = None  # tests only: the head drafts as usual (rows, trims), then these replace its drafts


def _slots(model, cache):
    """(the target's caches, the head's), or None when this cache list has no head slots."""
    n = getattr(model, "_mtp_trunk_layers", None)
    served = getattr(model, "_mtp_served", None)
    if n is None or served is None or cache is None or len(cache) != n + served.n_slots:
        return None
    return cache[:n], cache[n:]


def _attach(model, served, head, depth: int) -> None:
    """Give `model` a head: its cache slots after the model's own, and calls that keep them in step."""
    model._mtp_head = head
    model._mtp_served = served
    model._mtp_config = MTPConfig(depth=depth)
    model._mtp_trunk_layers = len(model.make_cache())
    make_cache = model.make_cache

    def make_cache_with_head():
        return [*make_cache(), *served.make_slots()]

    model.make_cache = make_cache_with_head
    base = type(model)

    class WithHead(base):
        def __call__(self, inputs, cache=None, **kwargs):
            slots = _slots(self, cache)
            if slots is None:
                return base.__call__(self, inputs, cache=cache, **kwargs)
            return served.call(self, inputs, *slots, **kwargs)

    WithHead.__name__ = f"{base.__name__}With{served.name}"
    model.__class__ = WithHead


# --- Qwen3.8's MTP head --------------------------------------------------------------------------------------------

class _ServedMTP:
    """The MTP head reads (trunk hidden p-1, token p) for each position p: its KV cache, and the last hidden state."""

    name = "MTP"
    n_slots = 2
    kv_quantizable = True  # its attention is mlx-lm's, which reads a quantized cache

    def __init__(self, head: mtp.MTPHead):
        self.head = head

    def make_slots(self) -> list:
        return [self.head.make_cache(), ArraysCache(size=1)]

    def call(self, model, inputs, target, own, **kwargs):
        head_cache, pending = own
        hidden = mtp.trunk_hidden(model, inputs, target)
        add_head_rows(model, inputs, hidden, head_cache, pending)
        return mtp.lm_logits(model, hidden)

    def draft(self, model, own, token: int, depth: int, history: list, config: MTPConfig):
        """Drafts for after `token`. The newest token's row (with the trunk's real hidden state before it) stays in
        the head's cache; the rows chained on the head's own output are drafts and come out again."""
        head_cache, pending = own
        if depth and config.drafter is not None:
            return list(config.drafter(history))[:depth], False
        if not depth:
            return [], False
        h = self.head(mtp.embed(model, mx.array([[token]])), pending[0], head_cache)
        drafts = []
        for i in range(depth):
            drafts.append(int(mx.argmax(mtp.lm_logits(model, h)[0, -1]).item()))
            if i + 1 < depth:
                h = self.head(mtp.embed(model, mx.array([[drafts[-1]]])), h, head_cache)
        if depth > 1:
            head_cache.trim(depth - 1)
        if config.draft_override is not None:
            drafts = list(config.draft_override(history))[:depth]
        return drafts, True  # the newest token's row is in

    def verify(self, model, target, tokens: list):
        hidden, _ = mtp.verify_forward(model, tokens, target)
        return mtp.lm_logits(model, hidden)[0], hidden

    def commit(self, model, target, own, tokens: list, kept: int, newest_row_added: bool, hidden) -> None:
        """Keep `kept` of the verified tokens, and give the head their rows from the trunk's real hidden states."""
        head_cache, pending = own
        mtp.commit(target, len(tokens), kept)
        if not newest_row_added:
            add_head_rows(model, mx.array([tokens[:kept]]), hidden[:, :kept], head_cache, pending)
        elif kept > 1:
            self.head(mtp.embed(model, mx.array([tokens[1:kept]])), hidden[:, :kept - 1], head_cache)
        pending[0] = hidden[:, kept - 1:kept]


def attach_mtp(model, head: mtp.MTPHead, *, depth: int = 2) -> None:
    """Give `model` an MTP head: two more cache slots, and calls that keep the head's cache in step."""
    _attach(model, _ServedMTP(head), head, depth)


def add_head_rows(model, inputs, hidden, head_cache, pending, *, previous=None) -> None:
    """Add the head's rows for `inputs`, pairing each token with the trunk's hidden state one position earlier
    (`previous`, or the pending one, for the first), and remember the last hidden state."""
    B, S, H = hidden.shape
    if previous is None:
        previous = pending[0] if pending[0] is not None else mx.zeros((B, 1, H), dtype=hidden.dtype)
    rows_hidden = mx.concatenate([previous, hidden[:, :-1]], axis=1) if S > 1 else previous
    model._mtp_head(mtp.embed(model, inputs), rows_hidden, head_cache)
    pending[0] = hidden[:, -1:]


# --- Inkling 2's DSpark drafter --------------------------------------------------------------------------------------

class _ServedDSpark:
    """The drafter reads the target's taps for every position it reads: its layers' contexts, and the recurrent
    head's input from the newest one."""

    name = "DSpark"
    kv_quantizable = False  # its attention reads its contexts as arrays

    def __init__(self, drafter: dspark.Drafter):
        self.drafter = drafter
        self.taps = drafter.config.target_layer_ids
        self.n_slots = len(drafter.make_cache())

    def make_slots(self) -> list:
        return self.drafter.make_cache()

    def call(self, model, inputs, target, own, input_embeddings=None, pixel_values=None):
        if pixel_values is not None:
            input_embeddings = model.embed_inputs(inputs, pixel_values=pixel_values)
        hidden, taps = dspark.read(model, inputs, target, self.taps, input_embeddings=input_embeddings)
        self.drafter.observe(taps, own)
        return model.logits(hidden)

    def draft(self, model, own, token: int, depth: int, history: list, config: MTPConfig):
        if depth and config.drafter is not None:
            return list(config.drafter(history))[:depth], None
        if not depth:
            return [], None
        contexts = dspark.one_sequence(own)  # a batch of one in the generator: its caches are batch caches
        drafts = self.drafter.propose(model, token, min(depth, self.drafter.width - 1), contexts).tolist()
        if config.draft_override is not None:
            drafts = list(config.draft_override(history))[:depth]
        return drafts, None

    def verify(self, model, target, tokens: list):
        hidden, taps = dspark.read(model, mx.array([tokens]), target, self.taps, record=True)
        return model.logits(hidden)[0], taps

    def commit(self, model, target, own, tokens: list, kept: int, _, taps) -> None:
        """Keep `kept` of the verified tokens, and give the drafter their taps."""
        dspark.rewind(target, len(tokens), kept)
        self.drafter.observe([t[:, :kept] for t in taps], own)


def attach_dspark(model, drafter: dspark.Drafter, *, depth: int = 3) -> None:
    """Give an Inkling 2 `model` its draft head: its contexts as more cache slots, and calls that keep them in step."""
    _attach(model, _ServedDSpark(drafter), drafter, depth)


# --- Sampling parameters ride on the sampler ---------------------------------------------------------------------

_original_make_sampler = mls._make_sampler


def _make_sampler(args, tokenizer):
    """mlx-lm's sampler, labelled with the distribution it draws from, which exact speculative sampling needs."""
    sampler = _original_make_sampler(args, tokenizer)
    s = args.sampling
    if s.xtc_probability > 0:  # XTC isn't a fixed distribution; such requests decode without MTP
        return sampler
    # A new function per request, except at temperature 0, where every request shares mlx-lm's greedy sampler (and
    # batches its sampling as one call): the label is the same for all of them, so it goes on the shared function.
    sampler.mtp_sampling = Sampling(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k, min_p=s.min_p)
    return sampler


mls._make_sampler = _make_sampler


# --- The speculative step ------------------------------------------------------------------------------------------

_original_next = GenerationBatch.next


def _sampling_of(batch) -> Sampling | None:
    sampler = (batch.samplers or [None])[0] or batch.fallback_sampler
    return getattr(sampler, "mtp_sampling", None)


def _can_speculate(batch) -> bool:
    model = batch.model
    return (len(batch.uids) == 1 and getattr(model, "_mtp_served", None) is not None
            and _slots(model, batch.prompt_cache) is not None and _sampling_of(batch) is not None
            and batch._next_tokens is not None and model._mtp_config.depth > 0)


def _speculative_next(batch):
    """One speculative round for a batch of one: a Response per returned token, the last perhaps finished."""
    model = batch.model
    served = model._mtp_served
    config = model._mtp_config
    stats = config.stats
    sampling = _sampling_of(batch)
    target, own = _slots(model, batch.prompt_cache)
    processors = (batch.logits_processors or [None])[0] or ()

    token = int(batch._next_tokens[0].item())
    token_logprobs = batch._next_logprobs[0]
    remaining = batch.max_tokens[0] - batch._num_tokens[0]
    depth = max(0, min(config.depth, remaining - 1))

    drafts, draft_state = served.draft(model, own, token, depth, [*batch.tokens[0], token], config)

    # Verify: one target pass over the newest token and the drafts.
    verify_tokens = [token, *drafts]
    rows_logits, verified = served.verify(model, target, verify_tokens)

    key = mx.random.key(int(mx.random.randint(0, 2**31 - 1).item()))
    fed = [(token, token_logprobs)]  # tokens the target has read, with the logprobs each was drawn from
    context = batch._token_context[0]
    for i in range(len(verify_tokens)):
        logits = rows_logits[i:i + 1]
        if processors:
            ctx = context.update_and_fetch(mx.array([verify_tokens[i]]))
            for p in processors:
                logits = p(ctx, logits)
        logprobs = (logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0]
        key, sub = mx.random.split(key)
        draft = drafts[i] if i < len(drafts) else None
        chosen, accepted = choose(logits, draft, sampling, sub)
        if not accepted:
            break
        fed.append((chosen, logprobs))

    # Return the read tokens in order, stopping at the first that finishes the sequence.
    responses = []
    for t, lp in fed:
        batch._num_tokens[0] += 1
        reason = "length" if batch._num_tokens[0] >= batch.max_tokens[0] else None
        if batch._matchers[0].advance(t):
            reason = "stop"
        batch.tokens[0].append(t)
        responses.append((t, lp, reason))
        if reason is not None:
            break
    kept = len(responses)
    served.commit(model, target, own, verify_tokens, kept, draft_state, verified)

    accepted_now = len(fed) - 1
    stats.rounds += 1
    stats.drafted += len(drafts)
    stats.accepted += accepted_now
    while len(stats.accepted_at) < len(drafts):
        stats.accepted_at.append(0)
    for i in range(accepted_now):
        stats.accepted_at[i] += 1

    finished = responses[-1][2] is not None
    out = [GenerationBatch.Response(
        uid=batch.uids[0], token=t, logprobs=lp, finish_reason=reason,
        prompt_cache=batch.extract_cache(0) if reason is not None else None,
        all_tokens=batch.tokens[0] if reason is not None else None) for t, lp, reason in responses]
    batch._current_tokens = mx.array([responses[-1][0]])
    batch._current_logprobs = [responses[-1][1]]
    if finished:
        batch.filter([])
    else:
        batch._next_tokens = mx.array([chosen])
        batch._next_logprobs = [logprobs]
    return out


def _next(self):
    if not _can_speculate(self):
        return _original_next(self)
    return _speculative_next(self)


GenerationBatch.next = _next


def mtp_snapshot(model) -> tuple | None:
    """(rounds, drafted, accepted) so far, for a model with a draft head; a request's log shows its difference."""
    config = getattr(model, "_mtp_config", None)
    if config is None:
        return None
    s = config.stats
    return s.rounds, s.drafted, s.accepted


# --- Loading: the head is attached on mlx-lm's loading thread, where all MLX work happens ----------------------

_original_load = mls.ModelProvider._load


def _is_inkling(model) -> bool:
    return getattr(model, "model_type", None) == "inkling2_loop"


def _load_with_mtp(self, *args, **kwargs):
    result = _original_load(self, *args, **kwargs)
    spec = getattr(self, "_mtp_spec", None)
    if spec is not None and self.model is not None and getattr(self.model, "_mtp_head", None) is None:
        if _is_inkling(self.model):
            # Its mtp_config is in the model's own config.json (mlx-lm's _load takes the model path first).
            model_path = str(args[0] if args else kwargs["model_path"])
            folder = Path(model_path) if Path(model_path).exists() else mlx_utils._download(model_path)
            drafter = dspark.load_drafter(folder, self.model, weights=spec.mtp, bits=spec.mtp_bits or None)
            attach_dspark(self.model, drafter, depth=spec.mtp_depth)
            kind = "DSpark draft head"
        else:
            head = mtp.load_mtp_head(spec.mtp, mtp.text_model(self.model).args, bits=spec.mtp_bits or None)
            attach_mtp(self.model, head, depth=spec.mtp_depth)
            kind = "MTP head"
        log.info("%s %s attached (%s, %d drafts per round)", kind, spec.mtp,
                 f"{spec.mtp_bits}-bit" if spec.mtp_bits else "unquantized", spec.mtp_depth)
    return result


mls.ModelProvider._load = _load_with_mtp
