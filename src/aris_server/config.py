"""Model settings: which weights a model name means and how effort and thinking map onto it."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_THINK_BUDGET = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384, "max": 0}


def home() -> Path:
    return Path(os.environ.get("ARIS_SERVER_HOME", Path.home() / ".aris-server"))


@dataclass
class ModelSpec:
    name: str
    path: str
    context: int | None = None
    # Template variable that carries effort (e.g. "reasoning_effort"), and how request efforts map to its values.
    effort_kwarg: str | None = None
    effort_map: dict[str, str] = field(default_factory=dict)
    # Thinking tokens allowed per effort before thinking is closed at the next paragraph break; 0 = unlimited.
    think_budget: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_THINK_BUDGET))
    template_kwargs: dict = field(default_factory=dict)
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
        budget = dict(DEFAULT_THINK_BUDGET)
        budget.update(raw.get("think_budget") or {})
        out[name] = ModelSpec(
            name=name,
            path=str(Path(raw["path"]).expanduser()) if raw["path"].startswith(("~", "/", ".")) else raw["path"],
            context=raw.get("context"),
            effort_kwarg=raw.get("effort_kwarg"),
            effort_map=dict(raw.get("effort_map") or {}),
            think_budget=budget,
            template_kwargs=dict(raw.get("template_kwargs") or {}),
            **{k: raw[k] for k in ("temperature", "top_p", "top_k") if k in raw},
        )
    return out


def resolve(name: str, models: dict[str, ModelSpec]) -> ModelSpec:
    """A configured model by name, or an unconfigured one by local path or Hugging Face repo id."""
    if name in models:
        return models[name]
    return ModelSpec(name=name, path=str(Path(name).expanduser()) if name.startswith(("~", "/", ".")) else name)
