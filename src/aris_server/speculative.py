"""Speculative decoding's model-independent half: the draft, verify and commit loop and its exact acceptance rule.

A drafter guesses the next few tokens; the target model reads the newest token and the guesses in one forward pass,
which at batch size 1 costs about as much as reading one token, and every guess it agrees with is a token for free.
What drafts and what the target keeps between rounds is model-specific (a `Speculator`); the loop and the rule
that keeps the output exact are here.

Drafts are greedy, so the draft distribution is a point mass and speculative sampling stays exact: a draft is
accepted with the target's probability of it, and a rejection samples the target with the draft removed. At
temperature 0 that means: accept while a draft equals the target's argmax. Batched verify math rounds differently
from one-token decode math, so greedy output can drift from plain decoding at near-ties; the distribution doesn't.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

# A logits processor, as mlx-lm's: (tokens so far, logits (1, vocab)) -> logits.
Processor = Callable[[mx.array, mx.array], mx.array]


@dataclass
class Sampling:
    """The target distribution, built exactly as mlx-lm's `make_sampler` builds it: the filters (top-p, min-p,
    top-k, in that order) act on log-probabilities at temperature 1, then the temperature applies. 0 means greedy."""

    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0

    def probs(self, logits: mx.array) -> mx.array:
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if 0 < self.top_p < 1.0:
            logprobs = apply_top_p(logprobs, self.top_p)
        if self.min_p != 0.0:
            logprobs = apply_min_p(logprobs, self.min_p, 1)
        if self.top_k > 0:
            logprobs = apply_top_k(logprobs, self.top_k)
        return mx.softmax(logprobs / self.temperature, axis=-1)


def choose(logits: mx.array, draft: int | None, sampling: Sampling, key) -> tuple[int, bool]:
    """Pick the token for one verify row (`logits` is (1, vocab)). Returns (token, whether it is the draft).

    A greedy draft is a point mass, so exact speculative sampling accepts it with the target's probability of it,
    and a rejection samples the target with the draft taken out."""
    if sampling.temperature == 0:
        token = int(mx.argmax(logits, axis=-1).item())
        return token, token == draft
    probs = sampling.probs(logits)[0]
    k_accept, k_sample = mx.random.split(key)
    if draft is not None:
        if mx.random.uniform(key=k_accept).item() < probs[draft].item():
            return draft, True
        probs = probs.at[draft].multiply(0.0)
        probs = probs / probs.sum()
    token = int(mx.random.categorical(mx.log(probs), key=k_sample).item())
    return token, False


@dataclass
class SpeculativeStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    accepted_at: list = field(default_factory=list)  # accepted_at[i] counts rounds where draft i+1 was accepted
    log_drafts: bool = False  # keep (tokens generated so far, drafts) for every round, for tests and diagnosis
    drafts_log: list = field(default_factory=list)
    prefill_seconds: float = 0.0  # reading the prompt, through the first token
    decode_seconds: float = 0.0  # everything after the first token

    @property
    def decode_tok_s(self) -> float:
        generated = self.accepted + self.rounds  # tokens produced after the first one
        return generated / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


class DepthControl:
    """Each round's draft depth: the one expected to give the most tokens a second, so drafting runs where it pays.

    How often each draft position is accepted changes with the text (an edit that copies, prose that doesn't), so it
    is learned as generation runs, and what a round of each depth costs is measured as rounds of it run. Depth 0
    decodes plainly. Every `explore` rounds it tries a depth next to its pick, alternately deeper and shallower, so it
    notices when the text gets easier or harder. The first two rounds measure the starting depth and a plain step."""

    def __init__(self, max_depth: int, *, start: int | None = None, explore: int = 16, decay: float = 0.95,
                 prior: float = 0.7):
        self.max_depth = max_depth
        self.start = max_depth if start is None else min(start, max_depth)
        self.explore, self.decay = explore, decay
        self.accept = [prior] * max_depth  # P(draft i+1 is accepted | every draft before it was)
        self.seen = [0] * max_depth  # rounds that tested each position
        self.seconds: dict[int, float] = {}  # depth -> recent seconds a round
        self.rounds = 0
        self.picks: Counter = Counter()  # how often each depth was chosen

    def rates(self) -> list[float]:
        """Each position's acceptance. One not yet tested takes the deepest tested position's: acceptance usually
        falls with depth, so that's optimistic, and the next rounds that reach it find out."""
        out = []
        for i, p in enumerate(self.accept):
            out.append(p if self.seen[i] or not out else out[-1])
        return out

    def tokens(self, depth: int) -> float:
        """Tokens a round of this depth is expected to add: the target's own, then each draft while all before held."""
        expected = run = 1.0
        for p in self.rates()[:depth]:
            run *= p
            expected += run
        return expected

    def cost(self, depth: int) -> float:
        """Seconds a round of this depth takes: measured, or along the line through the nearest measured depths."""
        if depth in self.seconds:
            return self.seconds[depth]
        known = sorted(self.seconds)
        lower = [d for d in known if d < depth]
        upper = [d for d in known if d > depth]
        if lower and upper:
            a, b = lower[-1], upper[0]
        elif len(known) >= 2:
            a, b = (known[-2], known[-1]) if lower else (known[0], known[1])
        else:
            return self.seconds[known[0]] * (1 + depth) / (1 + known[0])  # one measurement: cost grows with width
        slope = (self.seconds[b] - self.seconds[a]) / (b - a)
        return max(self.seconds[a] + slope * (depth - a), 1e-9)

    def best(self, room: int) -> int:
        limit = max(0, min(self.max_depth, room))
        if not self.seconds:
            return limit
        return max(range(limit + 1), key=lambda d: (self.tokens(d) / self.cost(d), -d))

    def choose(self, room: int) -> int:
        """The depth for the next round, given `room`: how many drafts the token budget can still use."""
        depth = self._choose(room)
        self.picks[depth] += 1
        return depth

    def _choose(self, room: int) -> int:
        limit = max(0, min(self.max_depth, room))
        if self.rounds == 0:
            return min(self.start, limit)
        if self.rounds == 1 and 0 not in self.seconds:
            return 0
        pick = self.best(room)
        if self.rounds % self.explore == 0:
            step = 1 if (self.rounds // self.explore) % 2 else -1
            if pick == 0:
                step = 1
            return min(max(pick + step, 0), limit)
        return pick

    def observe(self, depth: int, accepted: int, seconds: float) -> None:
        """A round of `depth` drafts kept `accepted` of them and took `seconds`."""
        self.rounds += 1
        rates = self.rates()
        for i in range(min(depth, accepted + 1)):  # after the first rejection, later positions weren't tested
            start = self.accept[i] if self.seen[i] else rates[i]
            self.accept[i] = self.decay * start + (1 - self.decay) * (1.0 if i < accepted else 0.0)
            self.seen[i] += 1
        old = self.seconds.get(depth)
        self.seconds[depth] = seconds if old is None else 0.8 * old + 0.2 * seconds


class Speculator(Protocol):
    """What one model brings to the loop. The target reads every token the loop sends it, in order; `commit` then
    keeps only what the loop chose."""

    def prefill(self, prompt: Sequence[int], step: int) -> mx.array:
        """Read the prompt, `step` tokens at a time. Returns the logits (1, vocab) for the first new token."""

    def draft(self, token: int, depth: int) -> mx.array | list[int]:
        """Up to `depth` guesses to follow `token`, the newest chosen token, which the target has not read yet.
        May return them unevaluated: they are read once the target has checked them."""

    def verify(self, tokens: mx.array) -> mx.array:
        """Read `tokens` (the newest chosen token, then the guesses). Returns their logits (len(tokens), vocab)."""

    def commit(self, total: int, kept: int) -> None:
        """Keep the first `kept` of the `total` tokens the last verify read, as if the rest were never read."""


# A test's stand-in for the drafter: given how many tokens have been generated, the guesses to try next.
DraftHook = Callable[[int], Sequence[int]]


def speculate(spec: Speculator, prompt: Sequence[int], *, max_tokens: int, depth: int,
              sampling: Sampling | None = None, processors: Sequence[Processor] = (), seed: int = 0,
              stop_tokens: Sequence[int] = (), prefill_step: int = 2048, drafter: DraftHook | None = None,
              stats: SpeculativeStats | None = None, control: DepthControl | None = None) -> list[int]:
    """Generate up to `max_tokens` tokens after `prompt` (batch size 1), drafting up to `depth` tokens a round, or as
    many as `control` picks each round."""
    sampling = sampling or Sampling()
    stats = stats if stats is not None else SpeculativeStats()
    stop = set(stop_tokens)
    greedy = sampling.temperature == 0 and not processors
    started = time.perf_counter()
    key = mx.random.key(seed)
    history = mx.array(list(prompt))

    logits = spec.prefill(prompt, prefill_step)
    for p in processors:
        logits = p(history, logits)
    key, sub = mx.random.split(key)
    token, _ = choose(logits, None, sampling, sub)
    out = [token]
    history = mx.concatenate([history, mx.array([token])])
    first_token_at = time.perf_counter()
    stats.prefill_seconds = first_token_at - started

    while len(out) < max_tokens and token not in stop:
        # A round adds the accepted drafts and one token of the target's own, so drafting past the budget is waste.
        room = max_tokens - len(out) - 1
        want = control.choose(room) if control is not None else min(depth, room)
        round_started = time.perf_counter()
        guesses = drafter(len(out)) if drafter is not None else spec.draft(token, want)
        if isinstance(guesses, mx.array):
            guesses = guesses[:want].astype(mx.int32)
            verify_tokens = mx.concatenate([mx.array([token], mx.int32), guesses])
        else:
            guesses = list(guesses)[:want]
            verify_tokens = mx.array([token, *guesses], mx.int32)
        rows = spec.verify(verify_tokens)
        if greedy:
            targets = mx.argmax(rows, axis=-1)
            mx.eval(targets, guesses)
            targets = targets.tolist()
        else:
            mx.eval(rows, guesses)
        drafts = guesses.tolist() if isinstance(guesses, mx.array) else guesses
        if stats.log_drafts:
            stats.drafts_log.append((len(out), list(drafts)))

        new, accepted_now = [], 0
        for i in range(len(drafts) + 1):
            draft = drafts[i] if i < len(drafts) else None
            if greedy:
                chosen = targets[i]
                accepted = chosen == draft
            else:
                logits = rows[i:i + 1]
                for p in processors:
                    logits = p(history, logits)
                key, sub = mx.random.split(key)
                chosen, accepted = choose(logits, draft, sampling, sub)
                history = mx.concatenate([history, mx.array([chosen])])
            new.append(chosen)
            accepted_now += accepted
            if not accepted or chosen in stop:
                break
        # The target keeps the newest token and each accepted draft it read; the last chosen token is read next round.
        spec.commit(len(drafts) + 1, len(new))
        if control is not None:
            control.observe(len(drafts), accepted_now, time.perf_counter() - round_started)

        stats.rounds += 1
        stats.drafted += len(drafts)
        stats.accepted += accepted_now
        while len(stats.accepted_at) < len(drafts):
            stats.accepted_at.append(0)
        for i in range(accepted_now):
            stats.accepted_at[i] += 1

        out.extend(new)
        token = new[-1]
    stats.decode_seconds = time.perf_counter() - first_token_at
    return out[:max_tokens]
