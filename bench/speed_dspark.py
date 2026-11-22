"""Speed of plain decoding vs speculative decoding with Inkling 2's own draft head (DSpark), batch size 1.

  python bench/speed_dspark.py --model ~/models/inkling-2-mini-mlx-4bit --prompts bench/prompts.json \\
      --out runs/dspark-speed.json

The draft head is `mtp.safetensors` in the model directory (or --head). Reports, per prompt:
  - decode tok/s for plain decoding (one token per target pass) and for DSpark at each depth;
  - draft acceptance by position, tokens per round, and where greedy output first differs from plain decoding;
  - the cost of one target pass over 1-8 tokens at the prompt's context, and of one draft (what decides the
    speedup: a MoE target reads more experts for more tokens, so a wide verify is not free).
Run it on the M4 only, with Aris Server stopped, so one model is in memory at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten

from aris_server import dspark
from aris_server import models as _models
from aris_server.speculative import Sampling, SpeculativeStats, choose

_models.register()


def read_prompt(model, drafter, prompt, step=2048):
    spec = dspark.DSparkSpeculator(model, drafter)
    logits = spec.prefill(prompt, step)
    return spec, logits


def inkling_template_args(args: dict) -> dict:
    """bench/prompts.json is written for Qwen's chat template; Inkling's turns thinking off with reasoning_effort."""
    args = dict(args)
    if args.pop("enable_thinking", True) is False:
        args["reasoning_effort"] = "none"
    return args


def plain_decode(model, prompt, n, sampling, seed, stop):
    """One token per target pass, sampled with the same Sampling as the speculative loop."""
    spec, logits = read_prompt(model, None, prompt)
    key = mx.random.key(seed)
    out, t0 = [], None
    while len(out) < n:
        key, sub = mx.random.split(key)
        token, _ = choose(logits, None, sampling, sub)
        out.append(token)
        if t0 is None:
            t0 = time.perf_counter()  # decode timing starts after the first token, as for DSpark
        if token in stop:
            break
        hidden, _ = dspark.read(model, mx.array([[token]]), spec.cache, ())
        logits = model.logits(hidden)[:, -1]
    elapsed = time.perf_counter() - t0
    return out, (len(out) - 1) / elapsed if len(out) > 1 else 0.0


def mlx_lm_decode(model, prompt, n, sampling, stop):
    """mlx-lm's own decoding (generate_step, which overlaps building each step with running the last): the speed
    Aris Server decodes at today, so the honest baseline for a speedup."""
    sampler = make_sampler(sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k)
    out, t0 = [], None
    for token, _ in generate_step(mx.array(prompt), model, max_tokens=n, sampler=sampler):
        out.append(int(token))
        if t0 is None:
            t0 = time.perf_counter()
        if out[-1] in stop:
            break
    elapsed = time.perf_counter() - t0
    return out, (len(out) - 1) / elapsed if len(out) > 1 else 0.0


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def verify_costs(model, drafter, prompt, widths=range(1, 9), repeats=5):
    """Median seconds for one recorded target pass (taps and logits) over `width` tokens at this context, kept to
    one token each try so the context grows by one; and the median seconds of one full-block draft."""
    spec, _ = read_prompt(model, drafter, prompt)
    costs = {}
    for w in widths:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(spec.verify(mx.array([1] * w, mx.int32)), spec._pending)
            times.append(time.perf_counter() - t0)
            spec.commit(w, 1)
            mx.eval(dspark.cache_arrays(spec.cache), spec.draft_cache.arrays())
        costs[w] = sorted(times)[len(times) // 2]
    drafts = {}
    for count in (1, 2, 3, drafter.width - 1):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(drafter.propose(model, 1, count, spec.draft_cache))
            times.append(time.perf_counter() - t0)
        drafts[count] = sorted(times)[len(times) // 2]
    return costs, drafts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="", help="the draft head's safetensors (default: mtp.safetensors in --model)")
    ap.add_argument("--prompts", required=True, help="JSON list of {name, messages, template_args?} or {name, text}")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--depths", default="1,2,3,4,5,6,7")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--drafter-bits", default="0",
                    help="comma-separated: each quantizes a copy of the drafter (Drafter.quantize; 0 keeps bf16)")
    ap.add_argument("--only", default="", help="comma-separated prompt names")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = Path(a.model).expanduser()
    model, tokenizer = load(str(path))
    head = Path(a.head).expanduser() if a.head else path / "mtp.safetensors"
    drafters = {int(b): dspark.load_drafter(path, model, weights=head, bits=int(b) or None)
                for b in a.drafter_bits.split(",") if b != ""}

    def size_gb(drafter):
        fold = drafter._fold_q if drafter._fold_q is not None else [drafter._fold_weight]
        return (sum(v.nbytes for _, v in tree_flatten(drafter.parameters()))
                + sum(x.nbytes for x in fold) + drafter._fold_bias.nbytes) / 1e9

    sampling = Sampling(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    stop = sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    depths = [int(d) for d in a.depths.split(",") if d]
    only = {n for n in a.only.split(",") if n}
    results = {"model": a.model, "head": str(head), "drafter_gb": {b: size_gb(d) for b, d in drafters.items()},
               "sampling": vars(sampling), "max_tokens": a.max_tokens, "prompts": []}
    print("drafter GB by bits (0 = bf16):", {b: round(g, 2) for b, g in results["drafter_gb"].items()}, flush=True)

    for item in json.loads(Path(a.prompts).read_text()):
        if only and item["name"] not in only:
            continue
        if "messages" in item:
            prompt = list(tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True,
                                                        **inkling_template_args(item.get("template_args", {}))))
        else:
            prompt = tokenizer.encode(item["text"])
        row = {"name": item["name"], "prompt_tokens": len(prompt)}
        print(f"== {item['name']}: {len(prompt)} prompt tokens", flush=True)

        row["draft_seconds"] = {}
        for bits, drafter in drafters.items():
            row["verify_seconds"], row["draft_seconds"][bits] = verify_costs(model, drafter, prompt)
            print(f"   drafter bits {bits}: draft seconds by guesses",
                  {k: round(v, 4) for k, v in row["draft_seconds"][bits].items()}, flush=True)
        print("   verify pass seconds by width:", {k: round(v, 4) for k, v in row["verify_seconds"].items()},
              flush=True)
        mx.clear_cache()

        plain, plain_speed = plain_decode(model, prompt, a.max_tokens, sampling, 0, stop)
        base, base_speed = mlx_lm_decode(model, prompt, a.max_tokens, sampling, stop)
        row["plain"] = {"tokens": len(plain), "tok_s": plain_speed}
        row["mlx_lm"] = {"tokens": len(base), "tok_s": base_speed, "first_difference_from_plain":
                         first_difference(base, plain)}
        print(f"   plain: {len(plain)} tokens, {plain_speed:.2f} tok/s; mlx-lm's own decoding {base_speed:.2f} tok/s",
              flush=True)
        mx.clear_cache()

        for (bits, drafter), depth in ((d, depth) for d in drafters.items() for depth in depths):
            stats = SpeculativeStats()
            out = dspark.generate_dspark(model, drafter, prompt, max_tokens=a.max_tokens, depth=depth,
                                         sampling=sampling, seed=0, stop_tokens=stop, stats=stats)
            speed = (len(out) - 1) / stats.decode_seconds if stats.decode_seconds else 0.0
            r = row[f"dspark{depth}" + (f"_q{bits}" if bits else "")] = {
                "tokens": len(out), "tok_s": speed, "speedup": speed / base_speed if base_speed else None,
                "acceptance": stats.acceptance, "tokens_per_round": stats.tokens_per_round,
                "accepted_at": [x / stats.rounds for x in stats.accepted_at] if stats.rounds else [],
                "first_difference_from_plain": first_difference(out, plain),
                "prefill_seconds": stats.prefill_seconds,
            }
            print(f"   dspark{'' if not bits else f' q{bits}'} depth {depth}: {speed:.2f} tok/s ({r['speedup'] or 0:.2f}x mlx-lm's), acceptance "
                  f"{stats.acceptance:.2f}, {stats.tokens_per_round:.2f} tokens/round, by position "
                  f"{[round(x, 2) for x in r['accepted_at']]}, first difference {r['first_difference_from_plain']}",
                  flush=True)
            mx.clear_cache()
        results["prompts"].append(row)
        Path(a.out).write_text(json.dumps(results, indent=2))

    print("wrote", a.out)


if __name__ == "__main__":
    main()
