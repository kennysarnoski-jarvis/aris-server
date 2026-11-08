"""aris-server command line."""

import argparse
import logging
import sys

from . import __version__
from .config import home, load_models, resolve


def _wired_limit():
    from mlx_lm.utils import maybe_set_recommended_wired_limit

    maybe_set_recommended_wired_limit()


def _quiet_dependencies(level):
    """mlx-lm logs through the root logger and httpx logs every Hugging Face request; keep only their warnings."""
    for name in ("httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(level)

    def keep(record):
        noisy = "mlx_lm" in record.pathname or record.name.startswith(("httpx", "huggingface_hub"))
        return record.levelno >= level or not noisy

    for handler in logging.getLogger().handlers:
        handler.addFilter(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aris-server", description="Open-weight models behind the Anthropic Messages API.")
    ap.add_argument("--version", action="version", version=f"aris-server {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="load a model and serve it")
    s.add_argument("--model", help="configured model name, local MLX folder, or Hugging Face repo id")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8484)
    s.add_argument("--api-key", help="require this key in x-api-key")
    s.add_argument("--concurrency", type=int, default=4, help="sequences decoded at once")
    s.add_argument("--prompt-cache-gb", type=float, help="memory for saved prompt caches (default: fit to the GPU)")
    s.add_argument("--reserve-gb", type=float, default=6.0, help="GPU memory kept free for active sequences")
    s.add_argument("--log-level", default="INFO")
    sub.add_parser("models", help="list configured models")
    a = ap.parse_args(argv)

    models = load_models()
    if a.cmd == "models":
        if not models:
            print(f"no models configured in {home() / 'models.toml'}")
        for name, spec in models.items():
            print(f"{name:24} {spec.path}")
        return 0

    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    _quiet_dependencies(logging.DEBUG if a.log_level.upper() == "DEBUG" else logging.WARNING)
    name = a.model or next(iter(models), None)
    if not name:
        print("no model: pass --model or configure one in models.toml", file=sys.stderr)
        return 2
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        logging.warning("serving on %s without --api-key: anyone who can reach this port can use the model", a.host)

    from .engine import Engine
    from .http import Server

    _wired_limit()
    engine = Engine(resolve(name, models), decode_concurrency=a.concurrency, reserve_gb=a.reserve_gb,
                    prompt_cache_bytes=int(a.prompt_cache_gb * (1 << 30)) if a.prompt_cache_gb else None)
    server = Server((a.host, a.port), engine, models, a.api_key)
    logging.info("Aris Server %s serving %s on http://%s:%d", __version__, name, a.host, a.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.generator.stop_and_join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
