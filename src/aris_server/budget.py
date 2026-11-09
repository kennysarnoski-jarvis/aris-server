"""Think budget: close a model's reasoning on a complete thought once it has used its allowance."""

import mlx.core as mx


class ThinkBudget:
    """Logits processor. Counts generated tokens while thinking; past `budget` and outside a code fence, it blocks
    paragraph breaks and forces the end-of-thinking token after the next newline. Inert once thinking has ended.

    Ending on a single newline matters: chat templates re-render past reasoning as `text\\n</think>`, so a turn that
    closed after a paragraph break (`text\\n\\n</think>`) no longer matches its own prompt cache, and the next turn
    re-reads everything from that point."""

    def __init__(self, budget: int, end_id: int, newline_id: int, paragraph_ids: frozenset[int],
                 fence_counts: dict[int, int]):
        self.budget, self.end_id, self.newline_id = budget, end_id, newline_id
        self.paragraph_ids, self.fence_counts = paragraph_ids, fence_counts
        self.n = self.fences = 0
        self.done = self.forced = False
        self._no_paragraphs = None

    def __call__(self, tokens, logits):
        if self.done or tokens is None or len(tokens) == 0:
            return logits
        last = int(tokens[-1].item())
        if last == self.end_id:
            self.done = True
            return logits
        self.n += 1
        self.fences += self.fence_counts.get(last, 0)
        if self.n < self.budget or self.fences % 2:
            return logits
        if last == self.newline_id:
            self.done = self.forced = True
            mask = mx.arange(logits.shape[-1]) == self.end_id
            forced = mx.where(mask, mx.array(0.0, logits.dtype), mx.array(-mx.inf, logits.dtype))
            return mx.broadcast_to(forced, logits.shape)
        if self._no_paragraphs is None:
            blocked = mx.zeros((logits.shape[-1],), mx.bool_)
            if self.paragraph_ids:
                blocked[mx.array(sorted(self.paragraph_ids))] = True
            self._no_paragraphs = blocked
        return mx.where(self._no_paragraphs, mx.array(-mx.inf, logits.dtype), logits)


class BudgetTokens:
    """Per-tokenizer lookups the processor needs, computed once per model."""

    def __init__(self, tokenizer):
        end = tokenizer.think_end_tokens if hasattr(tokenizer, "think_end_tokens") else None
        self.end_id = end[0] if end and len(end) == 1 else None
        newline = tokenizer.encode("\n", add_special_tokens=False) if hasattr(tokenizer, "encode") else []
        self.newline_id = newline[0] if len(newline) == 1 else None
        self.paragraph_ids, self.fence_counts = set(), {}
        decode = tokenizer.decode
        for i in range(len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size):
            text = decode([i])
            if text.endswith("\n\n"):
                self.paragraph_ids.add(i)
            if "```" in text:
                self.fence_counts[i] = text.count("```")
        self.paragraph_ids = frozenset(self.paragraph_ids)

    def make(self, budget: int) -> ThinkBudget | None:
        if budget <= 0 or self.end_id is None or self.newline_id is None:
            return None
        return ThinkBudget(budget, self.end_id, self.newline_id, self.paragraph_ids, self.fence_counts)
