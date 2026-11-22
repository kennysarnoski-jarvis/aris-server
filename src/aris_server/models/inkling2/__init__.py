# Vendored from mlx-lm pull request #2086 (jundot/mlx-lm @ e2b5bbd3), MIT License: see LICENSE.mlx-lm.
# Imports of mlx-lm main-branch helpers go through `_compat`, which maps them onto mlx-lm 0.32.2.
# Copyright © 2026 Apple Inc.

"""Inkling 2.

https://huggingface.co/thinkingmachines/Inkling-2-Small
"""

from .config import ModelArgs
from .model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
