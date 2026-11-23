"""Decode speed through the path Aris Server serves by: mlx-lm's BatchGenerator, with the model's draft head attached
as the server attaches it (mtp_serve), against the same generator without a head. Batch size 1.

  python bench/speed_served.py --model ~/models/inkling-2-mini-mlx-4bit \\
      --head ~/models/inkling-2-mini-mtp/mtp.safetensors --prompts bench/prompts.json --depths 3,4 --adaptive 7 \\
      --out runs/served-inkling.json

Per prompt: tok/s without a head; with the head attached but drafting off (the cost of keeping its caches in step);
at each fixed depth; and adaptive (mtp_adaptive: each round drafts as many tokens as pay, up to the given depth).
Run it on the M4 only, with Aris Server stopped, so one model is in memory at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

from aris_server import dspark, mtp, mtp_serve
from aris_server import models as _models
from aris_server.speculative import DepthControl, Sampling, SpeculativeStats

_models.register()


def greedy(logprobs):
    return mx.argmax(logprobs, axis=-1)


greedy.mtp_sampling = Sampling(0.0)


def template_args(model, args: dict) -> dict:
    """bench/prompts.json is written for Qwen's chat template; Inkling's turns thinking off with reasoning_effort."""
    args = dict(args)
    if getattr(model, "model_type", None) == "inkling2_loop" and args.pop("enable_thinking", True) is False:
        args["reasoning_effort"] = "none"
    return args


def decode(model, prompt, n, stop):
    """Tokens and decode tok/s (after the first token) through BatchGenerator, and the seconds to the first token."""
    gen = BatchGenerator(model, max_tokens=n, sampler=greedy, stop_tokens=[[t] for t in stop])
    started = time.perf_counter()
    gen.insert([prompt], max_tokens=[n])
    out, first = [], None
    while True:
        responses = gen.next_generated()
        for r in responses:
            out.append(r.token)
            if first is None:
                first = time.perf_counter()
        if responses and responses[-1].finish_reason is not None:
            break
    end = time.perf_counter()
    gen.close()
    return out, ((len(out) - 1) / (end - first) if len(out) > 1 else 0.0), first - started


def attach(model, head_path: Path, model_dir: Path, bits: int, depth: int, draft_bits: int = 0) -> None:
    if getattr(model, "model_type", None) == "inkling2_loop":
        mtp_serve.attach_dspark(model, dspark.load_drafter(model_dir, model, weights=head_path, bits=bits or None),
                                depth=depth)
    else:
        mtp_serve.attach_mtp(model, mtp.load_mtp_head(head_path, mtp.text_model(model).args, bits=bits or None),
                             depth=depth, draft_bits=draft_bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="the draft head: Qwen3.8's mtp.* safetensors or Inkling 2's")
    ap.add_argument("--head-bits", type=int, default=8, help="quantize the head (0 keeps it as shipped)")
    ap.add_argument("--draft-bits", type=int, default=0,
                    help="MTP heads: drafts go through a copy of the output layer at these bits (0: the model's own)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--depths", default="3", help="fixed depths to measure")
    ap.add_argument("--adaptive", type=int, default=0, help="also measure adaptive depth, up to this many drafts")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = Path(a.model).expanduser()
    model, tokenizer = load(str(path))
    stop = sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    only = {n for n in a.only.split(",") if n}
    items = [i for i in json.loads(Path(a.prompts).read_text()) if not only or i["name"] in only]
    prompts = [(i["name"], list(tokenizer.apply_chat_template(i["messages"], add_generation_prompt=True,
                                                              **template_args(model, i.get("template_args", {})))))
               for i in items]
    results = {"model": a.model, "head": a.head, "head_bits": a.head_bits, "draft_bits": a.draft_bits,
               "max_tokens": a.max_tokens, "prompts": []}
    rows = {name: {"name": name, "prompt_tokens": len(p)} for name, p in prompts}

    for name, prompt in prompts:  # without a head, before one is attached
        out, speed, ttft = decode(model, prompt, a.max_tokens, stop)
        rows[name]["no_head"] = {"tokens": len(out), "tok_s": speed, "ttft_s": ttft}
        rows[name]["_reference"] = out
        print(f"== {name}: {len(prompt)} prompt tokens; no head {speed:.2f} tok/s (first token {ttft:.1f} s)",
              flush=True)
        mx.clear_cache()

    depths = [int(d) for d in a.depths.split(",") if d]
    attach(model, Path(a.head).expanduser(), path, a.head_bits, max(depths + [a.adaptive, 1]), a.draft_bits)
    config = model._mtp_config
    runs = [("head_off", 0, None)] + [(f"depth{d}", d, None) for d in depths]
    if a.adaptive:
        runs.append((f"adaptive{a.adaptive}", a.adaptive, "adaptive"))
    for name, prompt in prompts:
        row = rows[name]
        base = row["no_head"]["tok_s"]
        for label, depth, mode in runs:
            config.depth, config.stats = depth, SpeculativeStats()
            config.control = DepthControl(depth) if mode == "adaptive" else None
            out, speed, ttft = decode(model, prompt, a.max_tokens, stop)
            picks = config.control.picks if config.control is not None else {}
            s = config.stats
            row[label] = {"tokens": len(out), "tok_s": speed, "speedup": speed / base if base else None,
                          "ttft_s": ttft, "acceptance": s.acceptance, "tokens_per_round": s.tokens_per_round,
                          "rounds": s.rounds, "same_as_no_head": out == row["_reference"],
                          **({"depths": dict(sorted(picks.items()))} if picks else {})}
            print(f"   {label}: {speed:.2f} tok/s ({row[label]['speedup'] or 0:.2f}x), acceptance {s.acceptance:.2f}, "
                  f"{s.tokens_per_round:.2f} tokens/round, same tokens as no head: {out == row['_reference']}"
                  + (f", depths {dict(sorted(picks.items()))}" if picks else ""), flush=True)
            mx.clear_cache()
        results["prompts"].append({k: v for k, v in row.items() if k != "_reference"})
        Path(a.out).write_text(json.dumps(results, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
