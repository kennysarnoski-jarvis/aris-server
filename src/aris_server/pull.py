"""`aris-server pull`: download the weights a model would load on this Mac, ahead of serving it.

The same choice `serve` makes (the best weights this Mac holds, sized before anything big downloads), then the
download into Hugging Face's cache, where `serve` finds it. With --json, every step is one JSON object per line on
stdout, for apps that show the download:

  {"event": "chosen", "model": ..., "path": ..., "bytes": ..., "context": ...}
  {"event": "progress", "done": ..., "total": ...}      (about once a second)
  {"event": "done", "path": ..., "folder": ...}
  {"event": "error", "message": ...}
"""

import fnmatch
import json
import sys
import threading
import time
from pathlib import Path

from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS  # mlx-lm's own download patterns: pull fetches what loading would

from .config import ModelSpec

PROGRESS_EVERY = {True: 1.0, False: 10.0}  # seconds between progress reports: for apps (JSON), for people


class _Report:
    """Where a pull says what it's doing: JSON lines for apps, or plain lines for people."""

    def __init__(self, as_json: bool, out=None):
        self.as_json, self.out = as_json, out or sys.stdout

    def __call__(self, event: str, text: str = "", **fields):
        if self.as_json:
            print(json.dumps({"event": event, **fields}), file=self.out, flush=True)
        elif text:
            print(text, file=self.out, flush=True)


class _Bytes:
    """Bytes a snapshot download has moved, from huggingface_hub's two summed progress bars: bytes written to disk
    ("Reconstructing", the only one HTTP downloads move) and bytes received ("Downloading bytes"). Xet downloads move
    the second continuously but write to disk in big flushes, so the larger of the two is the better reading."""

    def __init__(self):
        self.lock, self.disk, self.net = threading.Lock(), 0, 0

    def add(self, kind: str, n: int):
        with self.lock:
            setattr(self, kind, getattr(self, kind) + n)

    def moved(self) -> int:
        with self.lock:
            return max(self.disk, self.net)

    def bar_class(self):
        """A progress-bar class for huggingface_hub that counts into this and draws nothing."""
        from tqdm.auto import tqdm

        counter = self

        class Counter(tqdm):
            def __init__(self, *args, **kwargs):
                desc = str(kwargs.get("desc", ""))
                bytes_bar = kwargs.get("unit") == "B"
                self.kind = ("disk" if desc.startswith("Reconstruct") else "net" if desc.startswith("Downloading")
                             else None) if bytes_bar else None
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                if self.kind and n:
                    counter.add(self.kind, int(n))
                return super().update(n)

        return Counter


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def pull(spec: ModelSpec, limit: int, as_json: bool = False, out=None) -> int:
    """Choose `spec`'s weights for a Mac whose GPU can use `limit` bytes, and download them. 0 when they are on disk,
    1 when they can't be found or downloaded, 2 when they don't fit this Mac."""
    import huggingface_hub

    from .engine import NotEnoughMemory, check_memory, choose_weights, context_before_loading, hub_info, \
        weights_and_config

    report = _Report(as_json, out)
    chosen = choose_weights(spec, limit)
    sized = weights_and_config(chosen.path, chosen.adapter)
    if sized is None:
        report("error", f"can't find {chosen.path}: it isn't a local folder, and the Hugging Face Hub doesn't know it "
                        "(check the name, and that the repo is public)",
               message=f"can't find {chosen.path} locally or on the Hugging Face Hub")
        return 1
    context = context_before_loading(chosen.path, chosen.adapter, chosen.kv_bits, limit)
    try:
        check_memory(spec.name, limit, sized[0], fits=context)
    except NotEnoughMemory as e:
        report("error", str(e), message=str(e))
        return 2
    report("chosen", f"{spec.name}: {chosen.path} ({_gb(sized[0])}), about {context:,} tokens of context on this Mac",
           model=spec.name, path=chosen.path, bytes=sized[0], context=context)
    if Path(chosen.path).is_dir():
        report("done", f"{chosen.path} is a local folder: nothing to download", path=chosen.path, folder=chosen.path)
        return 0

    info = hub_info(chosen.path)
    files = [s for s in (info.siblings if info else []) if any(fnmatch.fnmatch(s.rfilename, p)
                                                              for p in DEFAULT_ALLOW_PATTERNS)]
    total = sum(s.size or 0 for s in files)
    # Files already in the cache aren't downloaded again, so they count as done from the start.
    cached = sum(s.size or 0 for s in files if info and isinstance(huggingface_hub.try_to_load_from_cache(
        chosen.path, s.rfilename, revision=info.sha), str))
    moved, finished, failure, folder = _Bytes(), threading.Event(), [], []

    def download():
        try:
            folder.append(huggingface_hub.snapshot_download(
                chosen.path, revision=info.sha if info else None, allow_patterns=DEFAULT_ALLOW_PATTERNS,
                tqdm_class=moved.bar_class()))
        except Exception as e:  # noqa: BLE001  network, disk, a repo that went private: say what, and stop
            failure.append(e)
        finally:
            finished.set()

    threading.Thread(target=download, name="pull", daemon=True).start()
    last = 0.0
    while not finished.wait(0.2):
        if time.monotonic() - last >= PROGRESS_EVERY[as_json]:
            last = time.monotonic()
            now = cached + moved.moved()
            now = min(now, total) if total else now
            report("progress", f"  {_gb(now)} of {_gb(total)}" + (f" ({100 * now // total}%)" if total else ""),
                   done=now, total=total)
    if failure:
        report("error", f"download of {chosen.path} failed: {failure[0]}",
               message=f"download of {chosen.path} failed: {failure[0]}")
        return 1
    report("progress", "", done=total, total=total)
    report("done", f"downloaded {chosen.path} to {folder[0]}", path=chosen.path, folder=folder[0])
    return 0
