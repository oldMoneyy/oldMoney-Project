#!/usr/bin/env python3
"""
Systematically compare sparse (31335) vs dense (31333) across multiple
input sizes to establish the sparse speedup baseline.

For each size, alternates sparse/dense so server warmth is comparable,
and runs each configuration multiple times for stability.

Usage:
    python sparse_vs_dense_sweep.py
    python sparse_vs_dense_sweep.py --sizes 40000 100000 200000 300000
    python sparse_vs_dense_sweep.py --runs 3
"""
import argparse
import json
import statistics
import time

import requests


def make_prompt(approx_tokens, prompt_type="text"):
    """Create a prompt of approximately N tokens."""
    if prompt_type == "text":
        # ~10 tokens per sentence, natural English
        sentence = "The weather in London is often rainy and cloudy during winter. "
        repeats = approx_tokens // 10
        needle = "The secret code is BLUE-TIGER-42. "
        half = repeats // 2
        text = (sentence * half) + needle + (sentence * (repeats - half))
        text += " Answer with just the secret code, nothing else."
        return text
    elif prompt_type == "repetition":
        # Pure "A" repetition - like test_512k.py
        return "A " * approx_tokens + " In one word, what was the repeated character?"
    elif prompt_type == "docs":
        # Multi-document style
        parts = [f"Document {i}: The annual rainfall in region {i} is {100+i}mm per year." 
                 for i in range(approx_tokens // 15)]
        parts[len(parts) // 2] = "Document X: The secret code is BLUE-TIGER-42."
        text = " ".join(parts)
        text += " What is the secret code mentioned in the text?"
        return text


def time_request(port, prompt, max_output=128):
    """Fire a single request and return (wall_time, prompt_tokens, answer_snippet)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "MiniCPM-SALA",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output,
                "temperature": 0.0,
            },
            timeout=600,
        )
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return dt, None, f"HTTP {r.status_code}"
        data = r.json()
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", -1)
        content = data["choices"][0]["message"]["content"] or ""
        if "</think>" in content:
            answer = content.split("</think>")[-1].strip()
        else:
            answer = content.strip()
        return dt, prompt_tokens, answer[:100]
    except Exception as e:
        return time.perf_counter() - t0, None, f"EXC: {e}"


def run_pair(size, prompt_type, runs, max_output, sparse_port, dense_port):
    """For one input size, alternately test sparse and dense."""
    prompt = make_prompt(size, prompt_type)

    sparse_times = []
    dense_times = []
    actual_toks = None

    for i in range(runs):
        # Sparse first this round
        dt_s, toks_s, ans_s = time_request(sparse_port, prompt, max_output)
        sparse_times.append(dt_s)
        if actual_toks is None:
            actual_toks = toks_s

        # Then dense
        dt_d, toks_d, ans_d = time_request(dense_port, prompt, max_output)
        dense_times.append(dt_d)

    def stats(xs):
        if not xs: return "n/a"
        if len(xs) == 1: return f"{xs[0]:.2f}s"
        return f"min={min(xs):.2f}s median={statistics.median(xs):.2f}s max={max(xs):.2f}s"

    sp_best = min(sparse_times)
    dn_best = min(dense_times)
    speedup = (dn_best - sp_best) / dn_best * 100 if dn_best > 0 else 0

    print(f"\n--- size={size} ({prompt_type}) actual_tokens={actual_toks} ---")
    print(f"  Sparse : {stats(sparse_times)}")
    print(f"  Dense  : {stats(dense_times)}")
    print(f"  Speedup (best-of-{runs}): {speedup:+.1f}%  "
          f"({dn_best:.2f}s → {sp_best:.2f}s)")

    return {
        "size": size,
        "prompt_type": prompt_type,
        "actual_tokens": actual_toks,
        "sparse_times": sparse_times,
        "dense_times": dense_times,
        "speedup_pct": speedup,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-port", type=int, default=31335)
    ap.add_argument("--dense-port", type=int, default=31333)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[20000, 40000, 80000, 150000, 300000])
    ap.add_argument("--prompt-types", nargs="+",
                    default=["text"],
                    choices=["text", "repetition", "docs"])
    ap.add_argument("--runs", type=int, default=2,
                    help="Number of runs per config for stability")
    ap.add_argument("--max-output", type=int, default=64,
                    help="Max output tokens (keep small to isolate prefill)")
    ap.add_argument("--save", type=str, default=None,
                    help="Save raw results to JSON")
    args = ap.parse_args()

    print(f"Sparse server: port {args.sparse_port}")
    print(f"Dense server : port {args.dense_port}")
    print(f"Sizes        : {args.sizes}")
    print(f"Prompt types : {args.prompt_types}")
    print(f"Runs per cfg : {args.runs}")
    print(f"Max output   : {args.max_output}")
    print()

    results = []
    for ptype in args.prompt_types:
        for size in args.sizes:
            res = run_pair(size, ptype, args.runs, args.max_output,
                          args.sparse_port, args.dense_port)
            results.append(res)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'size':>8}  {'type':<10}  {'actual':>8}  {'sparse':>8}  {'dense':>8}  {'speedup':>8}")
    for r in results:
        sp_best = min(r['sparse_times'])
        dn_best = min(r['dense_times'])
        print(f"{r['size']:>8}  {r['prompt_type']:<10}  "
              f"{r['actual_tokens']:>8}  "
              f"{sp_best:>7.2f}s  {dn_best:>7.2f}s  "
              f"{r['speedup_pct']:>+7.1f}%")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved raw results to {args.save}")


if __name__ == "__main__":
    main()
