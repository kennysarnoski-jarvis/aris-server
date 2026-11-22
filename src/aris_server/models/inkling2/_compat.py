"""Where mlx-lm 0.32.2 keeps the helpers the Inkling 2 package imports from mlx-lm's main branch.

Each import below names its main-branch home. The bodies are identical in 0.32.2: only the modules moved.
"""

import mlx.core as mx

from mlx_lm.kernels._common import row_threads, scalar  # main: mlx_lm.kernels.common
from mlx_lm.kernels.expert_routing import expert_routing  # main: same module
from mlx_lm.models.base import gather_last_axis  # main: mlx_lm.layers.distributed
from mlx_lm.models.pipeline import PipelineMixin  # main: mlx_lm.layers.distributed
from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort  # main: mlx_lm.layers.moe

__all__ = ["PipelineMixin", "SwitchLinear", "_gather_sort", "_scatter_unsort", "expert_routing", "gather_last_axis",
           "row_threads", "scalar", "shard_experts_inplace", "splice"]


def splice(embeddings: mx.array, inputs: mx.array, *, rows: mx.array, token_id: int) -> mx.array:
    """The embeddings with the rows, in order, at the token_id positions of inputs (main: mlx_lm.layers.common)."""
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)


def shard_experts_inplace(*args, **kwargs):
    """Tensor parallelism across Macs. Aris Server serves a model on one Mac."""
    raise NotImplementedError("Aris Server serves a model on one Mac: sharding Inkling 2 is not supported")
