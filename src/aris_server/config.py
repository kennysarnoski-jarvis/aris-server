"""Model settings: which weights a model name means and how effort and thinking map onto it."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh", "max")
KV_BITS_CHOICES = (4, 8)
DEFAULT_THINK_BUDGET = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384, "max": 0}


def home() -> Path:
    return Path(os.environ.get("ARIS_SERVER_HOME", Path.home() / ".aris-server"))


# A model given several weights (`path` as a list, best first) loads the first that leaves this much context on the
# Mac it runs on: a coding agent's conversation outgrows less, and compacting it costs a long re-read each time.
DEFAULT_MIN_CONTEXT = 65536


@dataclass
class ModelSpec:
    name: str
    path: str
    context: int | None = None
    # Weights to choose from by memory, best first (`path` is the first): for example the same model at 8 and 4 bits.
    choices: tuple[str, ...] = ()
    min_context: int = DEFAULT_MIN_CONTEXT
    adapter: str | None = None  # a LoRA adapter folder (adapter_config.json + adapters.safetensors) on top of path
    # Bits per value of the KV cache (8 or 4), None for full precision: 8 bits holds about twice the context in the
    # same memory.
    kv_bits: int | None = None
    # Template variable that carries effort (e.g. "reasoning_effort"), and how request efforts map to its values.
    effort_kwarg: str | None = None
    effort_map: dict[str, str] = field(default_factory=dict)
    # Thinking tokens allowed per effort before thinking is closed at the next paragraph break; 0 = unlimited.
    think_budget: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_THINK_BUDGET))
    template_kwargs: dict = field(default_factory=dict)
    # Share of the think budget for turns that continue the model's own tool loop: the plan exists by then, and
    # overthinking models re-plan from scratch on every tool result.
    followup_budget: float = 0.5
    # Sampling when the request doesn't say (Claude Code sends none with thinking on). Qwen-family thinking defaults.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    def budget_for(self, effort: str) -> int:
        return int(self.think_budget.get(effort, DEFAULT_THINK_BUDGET.get(effort, 0)))


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    path = path or home() / "models.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    out = {}
    for name, raw in (data.get("models") or {}).items():
        if "path" not in raw:
            raise ValueError(f"model {name!r} in {path} has no path")
        paths = raw["path"] if isinstance(raw["path"], list) else [raw["path"]]
        if not paths or not all(isinstance(p, str) and p for p in paths):
            raise ValueError(f"model {name!r} in {path}: path must be a folder or repo id, or a list of them")
        paths = [_weights_path(p) for p in paths]
        budget = dict(DEFAULT_THINK_BUDGET)
        budget.update(raw.get("think_budget") or {})
        if raw.get("kv_bits") not in (None, *KV_BITS_CHOICES):
            raise ValueError(f"model {name!r} in {path}: kv_bits must be one of {KV_BITS_CHOICES}")
        out[name] = ModelSpec(
            name=name,
            path=paths[0],
            choices=tuple(paths) if len(paths) > 1 else (),
            min_context=int(raw.get("min_context", DEFAULT_MIN_CONTEXT)),
            context=raw.get("context"),
            adapter=str(Path(raw["adapter"]).expanduser()) if raw.get("adapter") else None,
            kv_bits=raw.get("kv_bits"),
            effort_kwarg=raw.get("effort_kwarg"),
            effort_map=dict(raw.get("effort_map") or {}),
            think_budget=budget,
            template_kwargs=dict(raw.get("template_kwargs") or {}),
            **{k: raw[k] for k in ("temperature", "top_p", "top_k", "followup_budget") if k in raw},
        )
    return out


def _weights_path(p: str) -> str:
    """A local folder with its home expanded, or a Hugging Face repo id as given."""
    return str(Path(p).expanduser()) if p.startswith(("~", "/", ".")) else p


def resolve(name: str, models: dict[str, ModelSpec]) -> ModelSpec:
    """A configured model by name, or an unconfigured one by local path or Hugging Face repo id."""
    if name in models:
        return models[name]
    return ModelSpec(name=name, path=_weights_path(name))
