"""Tiny random models with a real model's architecture, for tests that need mlx-lm's own code paths."""

import mlx.core as mx
from mlx_lm.models import qwen3_5


def tiny_hybrid_model():
    """Arete's architecture (Qwen3.5: three linear-attention layers to each full-attention one), tiny and random."""
    args = qwen3_5.TextModelArgs(model_type="qwen3_5", hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                                 num_attention_heads=12, num_key_value_heads=2, head_dim=64, vocab_size=64,
                                 linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
                                 linear_value_head_dim=32, full_attention_interval=2)
    mx.random.seed(0)
    model = qwen3_5.TextModel(args)
    mx.eval(model.parameters())
    return model
