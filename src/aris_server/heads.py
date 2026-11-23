"""Draft heads come from a local file, or from a file in a Hugging Face repo, written "owner/repo:file" (Inkling 2's
own is "thinkingmachines/Inkling-2-Mini:mtp.safetensors"). A repo's file downloads into the Hub's cache, as model
weights do, and is found there after, offline too."""

from __future__ import annotations

import re
from pathlib import Path

_HUB_FILE = re.compile(r"^([\w.-]+/[\w.-]+):([^:]+)$")


def hub_file(ref: str | None) -> tuple[str, str] | None:
    """(repo, file) for a head written "owner/repo:file"; None for a local path (a file that exists is one)."""
    if not ref or Path(ref).expanduser().exists():
        return None
    m = _HUB_FILE.match(ref)
    return (m[1], m[2]) if m else None


def _cached(repo: str, name: str) -> str | None:
    import huggingface_hub

    found = huggingface_hub.try_to_load_from_cache(repo, name)
    return found if isinstance(found, str) else None


def head_bytes(ref: str | None) -> int:
    """The head file's size, known before it loads or downloads: a local file's, or the Hub's listing of its repo (or
    the cached copy's, when the Hub can't be reached). 0 for no head, or when nothing can say."""
    if not ref:
        return 0
    where = hub_file(ref)
    if where is None:
        path = Path(ref).expanduser()
        return path.stat().st_size if path.is_file() else 0
    from .engine import hub_info

    repo, name = where
    if (info := hub_info(repo)) is not None:
        return next((s.size or 0 for s in info.siblings or [] if s.rfilename == name), 0)
    cached = _cached(repo, name)
    return Path(cached).stat().st_size if cached else 0


def extra_bytes(spec, config: dict) -> int:
    """What a model's draft head adds to its weights, before anything loads: the head's file, and for an MTP head
    with mtp_draft_bits, the copy of the output layer its drafts go through (a weight per vocabulary row and hidden
    unit at those bits, and a 16-bit scale and bias per 64 of them). Inkling's head keeps no such copy."""
    if not spec.mtp:
        return 0
    copy = 0
    if spec.mtp_draft_bits and config.get("model_type") != "inkling2_loop":
        text = config.get("text_config", config)
        copy = int(text.get("vocab_size", 0) * text.get("hidden_size", 0) * (spec.mtp_draft_bits / 8 + 4 / 64))
    return head_bytes(spec.mtp) + copy


def local_head(ref: str, tqdm_class=None) -> str:
    """The head's local file: the path itself, or the repo's file from the Hub's cache, downloaded into it first when
    it isn't there (with `tqdm_class` counting the bytes, as `aris-server pull` does)."""
    where = hub_file(ref)
    if where is None:
        return str(Path(ref).expanduser())
    repo, name = where
    if (cached := _cached(repo, name)) is not None:
        return cached
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(repo, allow_patterns=[name], tqdm_class=tqdm_class)
    return str(Path(folder) / name)
