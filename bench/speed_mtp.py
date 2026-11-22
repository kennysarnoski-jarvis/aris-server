"""Speed of plain decoding vs MTP speculative decoding, and of prompt reading, on one model (batch size 1).

  python bench/speed_mtp.py --model ~/models/arete-27b-expert1-mlx-8bit --mtp ~/mtp-work/mtp-raw.safetensors \\
      --prompts bench/prompts.json --out runs/mtp-speed.json

Reports, per prompt:
  - decode tok/s for plain decoding (one token per trunk pass) and for MTP at each depth;
  - draft acceptance by position, tokens per round, and where greedy output first differs from plain decoding;
  - the cost of one trunk pass over 1-4 tokens at the prompt's context (what decides the MTP speedup);
  - prompt-reading tok/s at each prefill chunk size (prompts of 4k tokens or more).
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
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from aris_server import mtp
from aris_server.speculative import Sampling, SpeculativeStats, choose


def read_prompt(model, prompt, step=2048):
    cache = make_prompt_cache(model)
    hidden = None
    for start in range(0, len(prompt), step):
        hidden = mtp.trunk_hidden(model, mx.array([prompt[start:start + step]]), cache)
        mx.eval(hidden, *[c.state for c in cache])
    return cache, hidden


def plain_decode(model, prompt, n, sampling, seed, stop):
    """One token per trunk pass, sampled as generate_mtp samples (same Sampling, same key schedule)."""
    cache, hidden = read_prompt(model, prompt)
    key = mx.random.key(seed)
    logits = mtp.lm_logits(model, hidden[:, -1:])[:, -1]
    out = []
    t0 = None
    while len(out) < n:
        key, sub = mx.random.split(key)
        token, _ = choose(logits, None, sampling, sub)
        out.append(token)
        if t0 is None:
            t0 = time.perf_counter()  # decode timing starts after the first token, as for MTP
        if token in stop:
            break
        logits = mtp.lm_logits(model, mtp.trunk_hidden(model, mx.array([[token]]), cache))[:, -1]
    elapsed = time.perf_counter() - t0
    return out, (len(out) - 1) / elapsed if len(out) > 1 else 0.0


def mlx_lm_decode(model, prompt, n, sampling, stop):
    """mlx-lm's own decoding (generate_step, which overlaps building each step with running the last): the speed
    Aris Server decodes at without MTP, so the honest baseline for a speedup."""
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


def verify_costs(model, prompt, widths=(1, 2, 3, 4), repeats=5):
    """Median seconds for one verify-mode trunk pass (plus logits) over `width` tokens at this prompt's context."""
    cache, _ = read_prompt(model, prompt)
    costs = {}
    for w in widths:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            hidden, _ = mtp.verify_forward(model, [1] * w, cache)
            mx.eval(mtp.lm_logits(model, hidden))
            times.append(time.perf_counter() - t0)
            mtp.commit(cache, w, 1)  # keep one token each try, so the context grows by one
        costs[w] = sorted(times)[len(times) // 2]
    return costs


def prefill_speeds(model, prompt, steps):
    """Prompt-reading tok/s per chunk size, and the peak GPU memory above the weights while reading. With fused
    attention a chunk holds no score matrix, so the peak grows with the chunk only through activations."""
    speeds, peaks = {}, {}
    for step in steps:
        mx.synchronize()
        mx.clear_cache()
        steady = mx.get_active_memory()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        read_prompt(model, prompt, step)
        speeds[step] = len(prompt) / (time.perf_counter() - t0)
        peaks[step] = (mx.get_peak_memory() - steady) / 1e9
        mx.clear_cache()
    return speeds, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True, help="safetensors with mtp.* tensors")
    ap.add_argument("--mtp-bits", type=int, default=8, help="quantize the head's linear layers (0 keeps bf16)")
    ap.add_argument("--prompts", required=True, help="JSON list of {name, messages, template_args?} or {name, text}")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--prefill-steps", default="512,1024,2048")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated prompt names")
    ap.add_argument("--draft-bits", type=int, default=0,
                    help="quantize the output layer drafts use to this many bits (0: drafts use the model's own)")
    ap.add_argument("--head-hidden", choices=("post", "pre"), default="post",
                    help="the trunk hidden state the head reads: after its final norm, or before it")
    ap.add_argument("--variants", default="",
                    help="comma-separated DRAFT_BITS:HEAD_HIDDEN pairs (e.g. 0:post,4:post,0:pre), all measured in one "
                         "run against the same baselines; overrides --draft-bits and --head-hidden")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tokenizer = load(str(Path(a.model).expanduser()))
    head = mtp.load_mtp_head(Path(a.mtp).expanduser(), mtp.text_model(model).args, bits=a.mtp_bits or None)
    sampling = Sampling(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    pairs = [v.split(":") for v in a.variants.split(",") if v] or [[str(a.draft_bits), a.head_hidden]]
    variants = [(int(bits), hidden, mtp.draft_lm_head(model, int(bits) or None)) for bits, hidden in pairs]
    stop = sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    depths = [int(d) for d in a.depths.split(",") if d]
    only = {n for n in a.only.split(",") if n}
    results = {"model": a.model, "mtp": a.mtp, "mtp_bits": a.mtp_bits,
               "variants": [f"{bits}:{hidden}" for bits, hidden, _ in variants], "sampling": vars(sampling),
               "max_tokens": a.max_tokens, "prompts": []}

    for item in json.loads(Path(a.prompts).read_text()):
        if only and item["name"] not in only:
            continue
        if "messages" in item:
            text = tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True, tokenize=False,
                                                 **item.get("template_args", {}))
        else:
            text = item["text"]
        prompt = tokenizer.encode(text)
        row = {"name": item["name"], "prompt_tokens": len(prompt)}
        print(f"== {item['name']}: {len(prompt)} prompt tokens", flush=True)

        if not a.skip_prefill and len(prompt) >= 4096:
            speeds, peaks = prefill_speeds(model, prompt, [int(s) for s in a.prefill_steps.split(",")])
            row["prefill_tok_s"], row["prefill_peak_gb"] = speeds, peaks
            print("   prefill tok/s by chunk:", {k: round(v, 1) for k, v in speeds.items()},
                  "peak GB above steady:", {k: round(v, 2) for k, v in peaks.items()}, flush=True)

        row["verify_seconds"] = verify_costs(model, prompt)
        print("   verify pass seconds by width:", {k: round(v, 4) for k, v in row["verify_seconds"].items()},
              flush=True)

        plain, plain_speed = plain_decode(model, prompt, a.max_tokens, sampling, 0, stop)
        base, base_speed = mlx_lm_decode(model, prompt, a.max_tokens, sampling, stop)
        row["plain"] = {"tokens": len(plain), "tok_s": plain_speed}
        row["mlx_lm"] = {"tokens": len(base), "tok_s": base_speed, "first_difference_from_plain":
                         first_difference(base, plain)}
        print(f"   plain: {len(plain)} tokens, {plain_speed:.2f} tok/s; mlx-lm's own decoding {base_speed:.2f} tok/s",
              flush=True)
        mx.clear_cache()

        for (bits, hidden, draft_logits), depth in ((v, depth) for v in variants for depth in depths):
            stats = SpeculativeStats()
            out = mtp.generate_mtp(model, head, prompt, max_tokens=a.max_tokens, depth=depth, sampling=sampling,
                                   seed=0, stop_tokens=stop, stats=stats, draft_logits=draft_logits,
                                   head_hidden=hidden)
            speed = (len(out) - 1) / stats.decode_seconds if stats.decode_seconds else 0.0
            name = f"mtp{depth}" + ("" if (bits, hidden) == (0, "post") else f"_d{bits}_{hidden}")
            r = row[name] = {
                "tokens": len(out), "tok_s": speed, "speedup": speed / base_speed if base_speed else None,
                "speedup_vs_plain_loop": speed / plain_speed if plain_speed else None,
                "acceptance": stats.acceptance, "tokens_per_round": stats.tokens_per_round,
                "accepted_at": [x / stats.rounds for x in stats.accepted_at] if stats.rounds else [],
                "first_difference_from_plain": first_difference(out, plain),
                "prefill_seconds": stats.prefill_seconds,
            }
            print(f"   {name}: {speed:.2f} tok/s ({r['speedup'] or 0:.2f}x mlx-lm's), acceptance "
                  f"{stats.acceptance:.2f}, {stats.tokens_per_round:.2f} tokens/round, by position "
                  f"{[round(x, 2) for x in r['accepted_at']]}, first difference {r['first_difference_from_plain']}",
                  flush=True)
            mx.clear_cache()
        results["prompts"].append(row)
        Path(a.out).write_text(json.dumps(results, indent=2))

    print("wrote", a.out)


if __name__ == "__main__":
    main()
