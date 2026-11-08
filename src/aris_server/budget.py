"""Think budget: close a model's reasoning on a complete thought once it has used its allowance."""

import mlx.core as mx


class ThinkBudget:
    """Logits processor. Counts generated tokens while thinking; past `budget`, at the next paragraph
    break outside a code fence, forces the end-of-thinking token. Inert once thinking has ended."""

    def __init__(self, budget: int, end_id: int, paragraph_ids: frozenset[int], fence_counts: dict[int, int]):
        self.budget, self.end_id = budget, end_id
        self.paragraph_ids, self.fence_counts = paragraph_ids, fence_counts
        self.n = self.fences = 0
        self.done = self.forced = False

    def __call__(self, tokens, logits):
        if self.done or tokens is None or len(tokens) == 0:
            return logits
        last = int(tokens[-1].item())
        if last == self.end_id:
            self.done = True
            return logits
        self.n += 1
        self.fences += self.fence_counts.get(last, 0)
        if self.n >= self.budget and last in self.paragraph_ids and self.fences % 2 == 0:
            self.done = self.forced = True
            mask = mx.arange(logits.shape[-1]) == self.end_id
            forced = mx.where(mask, mx.array(0.0, logits.dtype), mx.array(-mx.inf, logits.dtype))
            return mx.broadcast_to(forced, logits.shape)
        return logits


class BudgetTokens:
    """Per-tokenizer lookups the processor needs, computed once per model."""

    def __init__(self, tokenizer):
        end = tokenizer.think_end_tokens if hasattr(tokenizer, "think_end_tokens") else None
        self.end_id = end[0] if end and len(end) == 1 else None
        self.paragraph_ids, self.fence_counts = set(), {}
        decode = tokenizer.decode
        for i in range(len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size):
            text = decode([i])
            if "\n\n" in text and text.strip() == "":
                self.paragraph_ids.add(i)
            if "```" in text:
                self.fence_counts[i] = text.count("```")
        self.paragraph_ids = frozenset(self.paragraph_ids)

    def make(self, budget: int) -> ThinkBudget | None:
        if budget <= 0 or self.end_id is None:
            return None
        return ThinkBudget(budget, self.end_id, self.paragraph_ids, self.fence_counts)
