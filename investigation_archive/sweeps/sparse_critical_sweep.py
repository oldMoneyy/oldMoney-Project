#!/usr/bin/env python3
"""
Find sparse-vs-dense crossover threshold.

Tests sparse (31335) and dense (31333) sequentially — never concurrently —
because the two servers share the same GPU and would interfere.

Sequence per size:
    1. Fire request to sparse server, wait for response, record time
    2. Fire request to dense server, wait for response, record time
    3. Repeat for stability (runs-per-size)

Usage:
    python sparse_critical_sweep.py
    python sparse_critical_sweep.py --sizes 50000 60000 70000 90000
    python sparse_critical_sweep.py --runs 3
"""
import argparse
import json
import statistics
import time
import requests


def make_prompt(approx_tokens):
    sentence = "The weather in London is often rainy and cloudy during winter. "
    repeats = approx_tokens // 10
    needle = "The secret code is BLUE-TIGER-42. "
    half = repeats // 2
    text = (sentence * half) + needle + (sentence * (repeats - half))
    text += " Answer with just the secret code, nothing else."
    return text


def time_request(port, prompt, max_output):
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
        return dt, prompt_tokens, "OK"
    except Exception as e:
        return time.perf_counter() - t0, None, f"EXC: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-port", type=int, default=31335)
    ap.add_argument("--dense-port", type=int, default=31333)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[50000, 60000, 70000, 90000])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-output", type=int, default=64)
    ap.add_argument("--warmup", action="store_true",
                    help="Do one warmup call per server before timing")
    ap.add_argument("--save", type=str, default="sparse_critical_sweep.json")
    args = ap.parse_args()

    print(f"Sparse port: {args.sparse_port}")
    print(f"Dense  port: {args.dense_port}")
    print(f"Sizes: {args.sizes}")
    print(f"Runs per size: {args.runs}")
    print(f"Max output tokens: {args.max_output}")
    print("Sequential firing (never parallel) to avoid GPU contention")
    print()

    if args.warmup:
        print("Warmup: tiny ping to each server...")
        for port in [args.sparse_port, args.dense_port]:
            time_request(port, "Hello. What is 1+1? One word answer.", 8)
        print()

    results = []
    for size in args.sizes:
        prompt = make_prompt(size)
        sparse_times = []
        dense_times = []
        actual_tokens = None

        for run_idx in range(args.runs):
            dt_s, toks_s, status_s = time_request(args.sparse_port, prompt, args.max_output)
            sparse_times.append(dt_s)
            if actual_tokens is None and toks_s is not None:
                actual_tokens = toks_s

            dt_d, toks_d, status_d = time_request(args.dense_port, prompt, args.max_output)
            dense_times.append(dt_d)

            print(f"  size={size:>7} run={run_idx+1}/{args.runs}: "
                  f"sparse={dt_s:6.2f}s  dense={dt_d:6.2f}s  "
                  f"Δ={(dt_d-dt_s)/dt_d*100:+5.1f}%")

        sp_best = min(sparse_times)
        dn_best = min(dense_times)
        speedup = (dn_best - sp_best) / dn_best * 100 if dn_best > 0 else 0

        print(f"  size={size:>7} SUMMARY: sparse_min={sp_best:.2f}s  "
              f"dense_min={dn_best:.2f}s  speedup={speedup:+.1f}%  "
              f"(actual_tokens={actual_tokens})\n")

        results.append({
            "size": size,
            "actual_tokens": actual_tokens,
            "sparse_times": sparse_times,
            "dense_times": dense_times,
            "sparse_best": sp_best,
            "dense_best": dn_best,
            "speedup_pct": speedup,
        })

    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"{'size':>8}  {'actual':>8}  {'sparse':>8}  {'dense':>8}  {'speedup':>9}")
    for r in results:
        print(f"{r['size']:>8}  {r['actual_tokens']:>8}  "
              f"{r['sparse_best']:>7.2f}s  {r['dense_best']:>7.2f}s  "
              f"{r['speedup_pct']:>+8.1f}%")

    # Find crossover: smallest size where speedup >= +5% consistently
    crossover = None
    for r in results:
        if r['speedup_pct'] >= 5.0:
            crossover = r
            break

    if crossover:
        print(f"\nCrossover point (first size with speedup >= +5%):")
        print(f"  target {crossover['size']} (actual {crossover['actual_tokens']} tokens)")
        print(f"  speedup {crossover['speedup_pct']:+.1f}%")
    else:
        print("\nNo crossover found. Try larger sizes.")

    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw data saved to {args.save}")


if __name__ == "__main__":
    main()
