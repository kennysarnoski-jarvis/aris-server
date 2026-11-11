"""aris-server command line."""

import argparse
import dataclasses
import logging
import signal
import sys
from pathlib import Path

from . import __version__
from .config import builtin_models, home, load_models, resolve


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
    s.add_argument("--adapter", help="LoRA adapter folder to apply on top of the model")
    s.add_argument("--kv-bits", type=int, choices=[4, 8, 16],
                   help="bits per KV cache value (16 = full precision); overrides the model's kv_bits")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8484)
    s.add_argument("--api-key", help="require this key in x-api-key")
    s.add_argument("--concurrency", type=int, default=4, help="sequences decoded at once")
    s.add_argument("--prompt-cache-gb", type=float, help="memory for saved prompt caches (default: fit to the GPU)")
    s.add_argument("--disk-cache-gb", type=float, default=20.0,
                   help="disk for saved prompt caches, all models together (0 turns it off)")
    s.add_argument("--reserve-gb", type=float, default=6.0, help="GPU memory kept free for active sequences")
    s.add_argument("--log-level", default="INFO")
    sub.add_parser("models", help="list configured and built-in models")
    p = sub.add_parser("pull", help="download the weights a model would load on this Mac, ahead of serving it")
    p.add_argument("model", help="configured or built-in model name, or Hugging Face repo id")
    p.add_argument("--json", action="store_true", help="report each step as a JSON line on stdout, for apps")
    svc = sub.add_parser("service", help="run as a login service that restarts itself (macOS launchd)")
    svc.add_argument("action", choices=["install", "uninstall", "status"])
    svc.add_argument("serve_args", nargs=argparse.REMAINDER, help="arguments for serve, after --")
    a = ap.parse_args(argv)

    if a.cmd == "service":
        from . import service

        if a.action == "install":
            return service.install([x for x in a.serve_args if x != "--"])
        return service.uninstall() if a.action == "uninstall" else service.status()

    models = load_models()
    if a.cmd == "models":
        if not models:
            print(f"no models configured in {home() / 'models.toml'}")
        for name, spec in models.items():
            print(f"{name:24} {' or '.join(spec.choices) or spec.path}")
        for name, spec in builtin_models().items():
            if name not in models:
                print(f"{name:24} {' or '.join(spec.choices) or spec.path}  (built in)")
        return 0
    if a.cmd == "pull":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")  # stdout is the report
        _quiet_dependencies(logging.WARNING)
        import mlx.core as mx

        from .pull import pull

        return pull(resolve(a.model, models), mx.device_info()["max_recommended_working_set_size"], as_json=a.json)

    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    _quiet_dependencies(logging.DEBUG if a.log_level.upper() == "DEBUG" else logging.WARNING)
    name = a.model or next(iter(models), None)
    if not name:
        print("no model: pass --model or configure one in models.toml", file=sys.stderr)
        return 2
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        logging.warning("serving on %s without --api-key: anyone who can reach this port can use the model", a.host)

    import mlx.core as mx

    from .engine import Engine, NotEnoughMemory, check_memory, choose_weights, weights_and_config
    from .http import Server

    _wired_limit()
    spec = resolve(name, models)
    if a.adapter:
        spec = dataclasses.replace(spec, adapter=str(Path(a.adapter).expanduser().resolve()))
    if a.kv_bits:
        spec = dataclasses.replace(spec, kv_bits=None if a.kv_bits == 16 else a.kv_bits)
    if spec.adapter and not (Path(spec.adapter) / "adapter_config.json").is_file():
        print(f"no adapter_config.json in {spec.adapter}", file=sys.stderr)
        return 2
    try:
        # Before loading, and for a Hugging Face repo before downloading: a model this Mac can't hold would swap it to
        # a standstill, and under the service, again after every restart.
        limit = mx.device_info()["max_recommended_working_set_size"]
        spec = choose_weights(spec, limit)
        if (sized := weights_and_config(spec.path, spec.adapter)) is not None:
            check_memory(name, limit, sized[0])
        engine = Engine(spec, decode_concurrency=a.concurrency, reserve_gb=a.reserve_gb,
                        disk_cache_gb=a.disk_cache_gb,
                        prompt_cache_bytes=int(a.prompt_cache_gb * (1 << 30)) if a.prompt_cache_gb else None)
    except NotEnoughMemory as e:
        logging.error("%s", e)
        return 2
    server = Server((a.host, a.port), engine, models, a.api_key)
    logging.info("Aris Server %s serving %s on http://%s:%d", __version__, name, a.host, a.port)

    def stop(*_):  # a service manager stops with SIGTERM: finish the current step (and any cache write) first
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.generator.stop_and_join()  # the generation thread writes the prompt caches as it exits
    return 0


if __name__ == "__main__":
    sys.exit(main())
