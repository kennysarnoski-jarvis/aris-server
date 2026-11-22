"""Model settings: which weights a model name means and how effort and thinking map onto it."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh", "max")
KV_BITS_CHOICES = (4, 8)
MAX_MTP_DEPTH = 7  # drafts per round: an Inkling 2 draft block holds 7; past 3, Qwen's chained MTP drafts rarely survive
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
    # Its value for a request that doesn't think, for a template with no enable_thinking switch (Inkling: "none").
    effort_off: str | None = None
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
    # A draft head for speculative decoding: Qwen3.8-27B's MTP head (a safetensors file of its `mtp.*` tensors) or
    # Inkling 2's (its mtp.safetensors). It drafts `mtp_depth` tokens ahead and one pass of the model checks them.
    # `mtp_bits` quantizes its linear layers (0 keeps them).
    mtp: str | None = None
    mtp_depth: int = 2
    mtp_bits: int = 8

    def budget_for(self, effort: str) -> int:
        return int(self.think_budget.get(effort, DEFAULT_THINK_BUDGET.get(effort, 0)))


# Models the server knows by name with no models.toml entry; an entry there with the same name replaces one. Their
# weights are on the Hugging Face Hub, best first: the server picks what each Mac holds before downloading anything.
BUILTIN_MODELS = {
    # Aris 3.8-27B: Arete-27B (Nous Research, on Qwen3.8-27B) trained on its own passing agent runs. The settings are
    # the ones its evals ran with: Claude Code's effort onto Arete's trained effort levels, and a think budget each.
    "aris-3.8-27b": {
        "path": ["YourAris/Aris-3.8-27B-MLX-8bit", "YourAris/Aris-3.8-27B-MLX-4bit"],
        "kv_bits": 8,
        "effort_kwarg": "reasoning_effort",
        "effort_map": {"low": "low", "medium": "medium", "high": "medium", "xhigh": "xhigh", "max": "xhigh"},
        "think_budget": {"low": 1024, "medium": 3072, "high": 4096, "xhigh": 12288, "max": 0},
    },
    # Inkling 2 Mini (Thinking Machines, Apache 2.0): 26B with 3.4B active and its middle layers looped three times,
    # so it fits 32 GB Macs. Effort goes to its own effort levels (none to max, which its template turns into a
    # number); sampling is its generation_config's. Its KV cache stays at full precision (models/inkling2).
    "inkling-2-mini": {
        "path": "mlx-community/Inkling-2-Mini-4bit",
        "effort_kwarg": "reasoning_effort",
        "effort_map": {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"},
        "effort_off": "none",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    },
}


def load_models(path: Path | None = None) -> dict[str, ModelSpec]:
    """The models configured in models.toml (not the built-in ones: see builtin_models)."""
    path = path or home() / "models.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    return {name: _spec(name, raw, f"model {name!r} in {path}") for name, raw in (data.get("models") or {}).items()}


def builtin_models() -> dict[str, ModelSpec]:
    return {name: _spec(name, raw, f"built-in model {name!r}") for name, raw in BUILTIN_MODELS.items()}


def _spec(name: str, raw: dict, where: str) -> ModelSpec:
    if "path" not in raw:
        raise ValueError(f"{where} has no path")
    paths = raw["path"] if isinstance(raw["path"], list) else [raw["path"]]
    if not paths or not all(isinstance(p, str) and p for p in paths):
        raise ValueError(f"{where}: path must be a folder or repo id, or a list of them")
    paths = [_weights_path(p) for p in paths]
    budget = dict(DEFAULT_THINK_BUDGET)
    budget.update(raw.get("think_budget") or {})
    if raw.get("kv_bits") not in (None, *KV_BITS_CHOICES):
        raise ValueError(f"{where}: kv_bits must be one of {KV_BITS_CHOICES}")
    if not isinstance(raw.get("mtp_depth", 2), int) or not 0 <= raw.get("mtp_depth", 2) <= MAX_MTP_DEPTH:
        raise ValueError(f"{where}: mtp_depth must be a whole number from 0 to {MAX_MTP_DEPTH}")
    if raw.get("mtp_bits", 8) not in (0, 4, 8):
        raise ValueError(f"{where}: mtp_bits must be 0 (unquantized), 4 or 8")
    return ModelSpec(
        name=name,
        path=paths[0],
        choices=tuple(paths) if len(paths) > 1 else (),
        min_context=int(raw.get("min_context", DEFAULT_MIN_CONTEXT)),
        context=raw.get("context"),
        adapter=str(Path(raw["adapter"]).expanduser()) if raw.get("adapter") else None,
        kv_bits=raw.get("kv_bits"),
        effort_kwarg=raw.get("effort_kwarg"),
        effort_map=dict(raw.get("effort_map") or {}),
        effort_off=raw.get("effort_off"),
        think_budget=budget,
        template_kwargs=dict(raw.get("template_kwargs") or {}),
        mtp=str(Path(raw["mtp"]).expanduser()) if raw.get("mtp") else None,
        **{k: raw[k] for k in ("temperature", "top_p", "top_k", "followup_budget", "mtp_depth", "mtp_bits")
           if k in raw},
    )


def _weights_path(p: str) -> str:
    """A local folder with its home expanded, or a Hugging Face repo id as given."""
    return str(Path(p).expanduser()) if p.startswith(("~", "/", ".")) else p


def resolve(name: str, models: dict[str, ModelSpec]) -> ModelSpec:
    """A configured model by name, then a built-in one, or else an unconfigured one by local path or Hugging Face repo
    id."""
    if name in models:
        return models[name]
    if name in BUILTIN_MODELS:
        return builtin_models()[name]
    return ModelSpec(name=name, path=_weights_path(name))
