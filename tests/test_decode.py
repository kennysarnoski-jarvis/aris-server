"""Batched decoding must not grow a graph in what the batch carries from step to step: an unevaluated graph keeps
every array it was built from alive, and on the GPU each of those holds one of Metal's 499,000 buffers."""

import io

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import ArraysCache

import aris_server.engine as engine  # installs the patches
from tiny_models import tiny_hybrid_model


def graph_edges(a):
    f = io.StringIO()
    mx.export_to_dot(f, a)
    return f.getvalue().count("->")


def test_decoding_does_not_grow_a_graph_in_the_linear_attention_padding_counts():
    # Each decode step, every linear-attention layer counts its cache's padding down (`left_padding -= 1`), lazily,
    # and only the first such layer's mask reads it. With requests always in flight the batch never empties, so the
    # unread counts grew a graph per layer, a buffer a step: Arete-27B (48 such layers) reached Metal's limit after
    # about 10,600 steps and the engine died.
    n = engine.STATE_EVAL_STEPS
    gen = BatchGenerator(tiny_hybrid_model(), max_tokens=10 * n)
    gen.insert([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]], max_tokens=[10 * n] * 2)  # read cold, unpadded: counts kept

    def deepest():
        caches = [c for c in gen._generation_batch.prompt_cache if isinstance(c, ArraysCache)]
        assert caches and all(c.left_padding is not None for c in caches)  # the path under test
        return max(graph_edges(c.left_padding) for c in caches)

    for _ in range(n // 2 + 2 * n):
        gen.next_generated()
    earlier = deepest()
    for _ in range(2 * n):
        gen.next_generated()
    later = deepest()
    gen.close()
    assert later <= earlier + 10, f"the padding graph grew from {earlier} to {later} edges over {2 * n} steps"
