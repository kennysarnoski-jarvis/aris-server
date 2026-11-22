"""Model architectures Aris Server serves before mlx-lm releases them.

`register()` puts each one where mlx-lm's loader looks (`mlx_lm.models.<name>`, through MODEL_REMAPPING), unless
the installed mlx-lm already ships it: then mlx-lm's own wins.
"""

import importlib
import importlib.util
import pkgutil
import sys

from mlx_lm import utils as mlx_utils

# model_type in config.json -> package here.
ARCHITECTURES = {"inkling2_loop": "inkling2"}


def register() -> None:
    for model_type, name in ARCHITECTURES.items():
        target = f"mlx_lm.models.{name}"
        if target not in sys.modules and importlib.util.find_spec(target) is not None:
            continue  # mlx-lm ships it now
        package = importlib.import_module(f"{__name__}.{name}")
        sys.modules[target] = package
        # Submodules too, so a lookup by mlx-lm's name never loads a second copy of a module.
        for info in pkgutil.iter_modules(package.__path__):
            sys.modules[f"{target}.{info.name}"] = importlib.import_module(f"{package.__name__}.{info.name}")
        mlx_utils.MODEL_REMAPPING.setdefault(model_type, name)
