"""Prompt caches on disk, so a restart, a crash or a reopened thread resumes in seconds instead of re-reading a long
conversation (minutes for a 27B model at 40k tokens). A conversation evicted from memory is also loaded back from
here. Each finished turn's cache and the shared system-prompt snapshot
are written as they enter the in-memory cache; a conversation's older snapshot is deleted when its next turn is saved.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

MIN_TOKENS = 1024  # shorter prompts are quick to re-read


def model_id(model_path: str, adapter: str | None = None, kv_bits: int | None = None) -> str:
    """Caches are only valid for the exact weights that made them (the base model and any adapter on it) and for the
    precision of the KV cache they hold."""
    parts = []
    for folder in (model_path, adapter):
        if folder is None:
            continue
        path = Path(folder)
        parts.append(str(path))
        if path.is_dir():
            parts += [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in sorted(path.glob("*.safetensors"))]
    if kv_bits:
        parts.append(f"kv_bits={kv_bits}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


class DiskCache:
    def __init__(self, root: Path, model_path: str, max_bytes: int, adapter: str | None = None,
                 kv_bits: int | None = None):
        self.dir = root / model_id(model_path, adapter, kv_bits)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.index: dict[Path, tuple[tuple[int, ...], str]] = {}
        for data in self.dir.glob("*.safetensors"):
            if not data.with_suffix(".json").exists():  # a write cut off by a crash or a kill
                data.unlink(missing_ok=True)
        for f in self.dir.glob("*.json"):
            data = f.with_suffix(".safetensors")
            if not data.exists():
                f.unlink(missing_ok=True)
                continue
            meta = json.loads(f.read_text())
            self.index[data] = (tuple(meta["tokens"]), meta["type"])

    def has(self, tokens) -> bool:
        key = tuple(tokens)
        return any(t == key for t, _ in self.index.values())

    def save(self, tokens, prompt_cache, cache_type: str):
        if len(tokens) < MIN_TOKENS or cache_type not in ("system", "assistant"):
            return
        t0 = time.time()
        key = tuple(tokens)
        data = self.dir / f"{hashlib.sha1(json.dumps(key).encode()).hexdigest()[:20]}.safetensors"
        save_prompt_cache(str(data), prompt_cache)
        # The metadata file goes last: a file without it is an interrupted write and is ignored.
        data.with_suffix(".json").write_text(json.dumps({"tokens": list(key), "type": cache_type}))
        # Each turn saves the read prompt (what a retry of the same request needs) and then the finished answer (what
        # the next turn continues). Of the same conversation's earlier snapshots, keep only the longest.
        earlier = sorted((p for p, (t, kind) in self.index.items()
                          if kind != "system" and p != data and key[:len(t)] == t), key=lambda p: len(self.index[p][0]))
        for other in earlier[:-1]:
            self._delete(other)
        self.index[data] = (key, cache_type)
        size = data.stat().st_size
        self._trim()
        logging.info("saved %d-token %s cache to disk (%.0f MB, %.1fs)", len(key), cache_type, size / 1e6,
                     time.time() - t0)

    def _delete(self, data: Path):
        data.unlink(missing_ok=True)
        data.with_suffix(".json").unlink(missing_ok=True)
        self.index.pop(data, None)

    def _trim(self):
        files = sorted(self.index, key=lambda p: p.stat().st_mtime if p.exists() else 0)
        total = sum(p.stat().st_size for p in files if p.exists())
        while files and total > self.max_bytes:
            oldest = files.pop(0)
            total -= oldest.stat().st_size if oldest.exists() else 0
            self._delete(oldest)

    def nearest(self, tokens, longer_than: int = 0) -> tuple[Path, int] | None:
        """The saved cache covering the longest proper prefix of `tokens`, if it covers more than `longer_than`."""
        key = tuple(tokens)
        best = None
        for data, (t, _) in self.index.items():
            if longer_than < len(t) < len(key) and key[:len(t)] == t and (best is None or len(t) > best[1]):
                best = (data, len(t))
        return best

    @staticmethod
    def load(data: Path):
        cache = load_prompt_cache(str(data))
        # Loading is lazy and bound to the loading thread's stream; generation may run on another thread.
        mx.eval([c.state for c in cache])
        return cache

    def restore(self, insert, budget: int) -> int:
        """Load the most recent caches that fit in `budget` bytes through insert(tokens, cache, type)."""
        loaded = 0
        newest = sorted(self.index, key=lambda p: p.stat().st_mtime, reverse=True)
        systems = [p for p in newest if self.index[p][1] == "system"][:1]
        for data in systems + [p for p in newest if self.index[p][1] != "system"]:
            size = data.stat().st_size
            if loaded + size > budget:
                continue
            try:
                cache = self.load(data)
            except Exception:  # a damaged file: drop it and carry on
                logging.warning("dropping unreadable prompt cache %s", data.name)
                self._delete(data)
                continue
            tokens, cache_type = self.index[data]
            insert(list(tokens), cache, cache_type)
            loaded += size
        return loaded
