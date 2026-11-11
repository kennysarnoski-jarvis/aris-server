"""Prompt caches on disk, so a restart, a crash or a reopened thread resumes in seconds instead of re-reading a long
conversation (minutes for a 27B model at 40k tokens). A conversation evicted from memory is also loaded back from
here. Each finished turn's cache and the shared system-prompt snapshot
are written as they enter the in-memory cache; a conversation's older snapshot is deleted when its next turn is saved.

Each setup (weights, adapter, KV precision) has a folder of its own, and one cap covers them all. A server reads only
its own folder, so the caches of setups no running server holds go first, then its own oldest. A running server holds
a shared lock on its folder; another server sharing the directory clears only folders nobody holds.
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

MIN_TOKENS = 1024  # shorter prompts are quick to re-read
LOCK = "lock"      # shared by each server using the folder, exclusive for one clearing it


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


def _stat(data: Path) -> tuple[float, int]:
    """A cache file's (mtime, size), or (0, 0) once it is gone."""
    try:
        s = data.stat()
    except FileNotFoundError:
        return 0.0, 0
    return s.st_mtime, s.st_size


def _claim(folder: Path):
    """The folder's lock, held exclusively, or None while a server uses the folder or another is clearing it."""
    try:
        lock = open(folder / LOCK, "a")
    except OSError:
        return None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def _remove(data: Path):
    """Delete a cache and its metadata. One that can't be deleted (in a folder made read-only, say) stays, with a
    warning: the disk cache must never stop the server."""
    try:
        data.unlink(missing_ok=True)
        data.with_suffix(".json").unlink(missing_ok=True)
    except OSError as e:
        logging.warning("could not delete prompt cache %s: %s", data.name, e)


class DiskCache:
    def __init__(self, root: Path, model_path: str, max_bytes: int, adapter: str | None = None,
                 kv_bits: int | None = None):
        self.root = root
        self.dir = root / model_id(model_path, adapter, kv_bits)
        self.max_bytes = max_bytes
        self._lock = self._hold()
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
        self._trim()  # started on another setup: what the last one left over the cap goes now

    def _hold(self):
        """A shared lock on this folder for as long as the server runs."""
        while True:
            self.dir.mkdir(parents=True, exist_ok=True)
            try:
                lock = open(self.dir / LOCK, "a")
            except FileNotFoundError:  # another server removed the folder just now
                continue
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                if os.path.samestat(os.stat(self.dir / LOCK), os.fstat(lock.fileno())):
                    return lock
            except FileNotFoundError:
                pass
            lock.close()  # the folder was cleared while this waited for it: make it again

    def __del__(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            lock.close()

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
        self.index.pop(data, None)
        _remove(data)

    def _trim(self):
        """Keep this setup's caches, with those of setups no running server holds, under the cap: the others' go first,
        oldest first, since this server can't read them; then this setup's oldest."""
        claimed = [(folder, lock) for folder in self.root.iterdir()
                   if folder != self.dir and folder.is_dir() and re.fullmatch(r"[0-9a-f]{16}", folder.name)
                   and (lock := _claim(folder)) is not None]
        try:
            spare = sorted((p for folder, _ in claimed for p in folder.glob("*.safetensors")), key=_stat)
            own = sorted(self.index, key=_stat)
            total = sum(_stat(p)[1] for p in spare + own)
            freed = removed = 0
            for data in spare:
                if total <= self.max_bytes:
                    break
                size = _stat(data)[1]
                _remove(data)
                total -= size
                freed += size
                removed += 1
            for data in own:
                if total <= self.max_bytes:
                    break
                total -= _stat(data)[1]
                self._delete(data)
            if removed:
                logging.info("disk cache: removed %d caches (%.1f GB) that other models or settings left", removed,
                             freed / (1 << 30))
            for folder, _ in claimed:
                if any(folder.glob("*.safetensors")):
                    continue
                for leftover in [*folder.glob("*.json"), folder / LOCK]:
                    leftover.unlink(missing_ok=True)
                try:
                    folder.rmdir()
                except OSError:  # something else is in it: leave the folder
                    pass
        finally:
            for _, lock in claimed:
                lock.close()

    def nearest(self, tokens, longer_than: int = 0) -> tuple[Path, int] | None:
        """The saved cache covering the longest proper prefix of `tokens`, if it covers more than `longer_than`. One
        whose file is gone from disk (deleted by hand, say) is forgotten here, before the caller lets go of what memory
        holds for the turn."""
        key = tuple(tokens)
        while True:
            best = None
            for data, (t, _) in self.index.items():
                if longer_than < len(t) < len(key) and key[:len(t)] == t and (best is None or len(t) > best[1]):
                    best = (data, len(t))
            if best is None or best[0].exists():
                return best
            logging.warning("prompt cache %s is gone from disk; forgetting it", best[0].name)
            self._delete(best[0])

    def load(self, data: Path):
        """The cache saved in `data`, or None if it can't be read (damaged, or gone): then it is dropped, so no later
        turn tries it again."""
        try:
            cache = load_prompt_cache(str(data))
            # Loading is lazy and bound to the loading thread's stream; generation may run on another thread.
            mx.eval([c.state for c in cache])
        except Exception as e:  # noqa: BLE001  MLX raises RuntimeError for a damaged or missing file
            logging.warning("dropping unreadable prompt cache %s: %s", data.name, e)
            self._delete(data)
            return None
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
            cache = self.load(data)
            if cache is None:  # damaged: dropped
                continue
            tokens, cache_type = self.index[data]
            insert(list(tokens), cache, cache_type)
            loaded += size
        return loaded
