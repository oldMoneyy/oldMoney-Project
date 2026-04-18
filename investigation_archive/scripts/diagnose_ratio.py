#!/usr/bin/env python3
"""Diagnose why sparse is 35% faster at bs=1 but not at bs=64.

Steps:
1. Clear /tmp/sparse_debug.log
2. Fire a single long-prompt request
3. Parse the log to see what ratio each [SPARSE-DETAIL] reports
4. Compare with server wall-clock and per-layer timings

Usage:
    python diagnose_sparse_ratio.py --port 31335 --input-tokens 200000
    python diagnose_sparse_ratio.py --port 31335 --input-tokens 80000
    python diagnose_sparse_ratio.py --port 31335 --input-tokens 40000
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

DEBUG_LOG = Path("/tmp/sparse_debug.log")


def clear_log():
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()
    DEBUG_LOG.touch()


def parse_sparse_log():
    """Return list of (layer, filtered_tokens, full_prefix, ratio, time_s)."""
    details = []
    pages = []
    if not DEBUG_LOG.exists():
        return details, pages

    detail_re = re.compile(
        r"\[SPARSE-DETAIL\] layer=(\d+) filtered_tokens=(\d+) vs full_prefix=(\d+) ratio=([\d.]+)"
    )
    page_re = re.compile(
        r"\[SPARSE-PAGED\] layer=(\d+) time=([\d.]+)s"
    )
    dense_re = re.compile(
        r"\[DENSE-PAGED\] layer=(\d+) time=([\d.]+)s"
    )
    meta_re = re.compile(
        r"\[SPARSE-META\] layer=(\d+) (returned None|prefix=)"
    )

    meta_results = []
    dense_pages = []
    with open(DEBUG_LOG) as f:
        for line in f:
            m = detail_re.search(line)
            if m:
                details.append({
                    "layer": int(m.group(1)),
                    "filtered": int(m.group(2)),
                    "full": int(m.group(3)),
                    "ratio": float(m.group(4)),
                })
                continue
            m = page_re.search(line)
            if m:
                pages.append({
                    "layer": int(m.group(1)),
                    "time": float(m.group(2)),
                    "kind": "sparse",
                })
                continue
            m = dense_re.search(line)
            if m:
                dense_pages.append({
                    "layer": int(m.group(1)),
                    "time": float(m.group(2)),
                    "kind": "dense",
                })
                continue
            m = meta_re.search(line)
            if m:
                meta_results.append({
                    "layer": int(m.group(1)),
                    "returned_none": "returned None" in line,
                })

    return details, pages, dense_pages, meta_results


def make_prompt(approx_tokens):
    """Dummy prompt of approximately N tokens. Using repeated short strings."""
    # With the MiniCPM tokenizer, "The weather in London is often rainy and cloudy. "
    # tokenizes to ~10 tokens per sentence. So for N tokens, repeat N/10 times.
    sentence = "The weather in London is often rainy and cloudy during winter. "
    repeats = approx_tokens // 10
    needle = "The secret code is BLUE-TIGER-42. "
    # Put needle roughly in the middle so sparse has to actually fetch it
    half = repeats // 2
    text = (sentence * half) + needle + (sentence * (repeats - half))
    text += " What is the secret code mentioned above? Answer in exactly the format XXX-YYY-NN."
    return text


def run_test(port, approx_tokens, max_output=64):
    clear_log()
    prompt = make_prompt(approx_tokens)

    t0 = time.perf_counter()
    r = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": "MiniCPM-SALA",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output,
            "temperature": 0.0,
        },
        timeout=None,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()

    prompt_tokens = data.get("usage", {}).get("prompt_tokens", "?")
    content = data["choices"][0]["message"]["content"] or ""
    # Strip think tags
    if "</think>" in content:
        answer = content.split("</think>")[-1].strip()
    else:
        answer = content.strip()

    # Give the file a moment to flush
    time.sleep(0.2)

    details, pages, dense_pages, meta_results = parse_sparse_log()

    print(f"\n=== Port {port} | target {approx_tokens} tokens ===")
    print(f"Wall time         : {dt:.2f}s")
    print(f"Actual prompt toks: {prompt_tokens}")
    print(f"Answer (first 120): {answer[:120]!r}")
    print(f"Correct           : {'BLUE-TIGER-42' in answer}")

    print(f"\n--- Debug log events ---")
    print(f"[SPARSE-META] events : {len(meta_results)}  "
          f"(returned_none: {sum(1 for m in meta_results if m['returned_none'])}, "
          f"real: {sum(1 for m in meta_results if not m['returned_none'])})")
    print(f"[SPARSE-DETAIL] events: {len(details)}")
    print(f"[SPARSE-PAGED] events : {len(pages)}  (sparse attention calls)")
    print(f"[DENSE-PAGED] events  : {len(dense_pages)}  (dense attention calls)")

    if details:
        ratios = [d["ratio"] for d in details]
        print(f"\n--- Sparse ratios across all SPARSE-DETAIL events ---")
        print(f"  min={min(ratios):.3f}  max={max(ratios):.3f}  "
              f"mean={sum(ratios)/len(ratios):.3f}")
        if max(ratios) >= 0.99:
            print(f"  ⚠  ratio ≈ 1.0 — sparse selection is returning FULL prefix")
        if min(ratios) < 0.5:
            print(f"  ✓  some ratios < 0.5 — sparse is actually filtering")

    if pages:
        total_sparse = sum(p["time"] for p in pages)
        print(f"\n--- Sparse-paged kernel time ---")
        print(f"  total: {total_sparse:.2f}s across {len(pages)} layer-calls")
        print(f"  avg per call: {total_sparse/len(pages):.3f}s")

    if dense_pages:
        total_dense = sum(p["time"] for p in dense_pages)
        print(f"\n--- Dense-paged kernel time ---")
        print(f"  total: {total_dense:.2f}s across {len(dense_pages)} layer-calls")
        print(f"  avg per call: {total_dense/len(dense_pages):.3f}s")

    if pages and dense_pages:
        # If sparse is working, sparse-kernel time should be lower than
        # dense-kernel time on the same-size prefix.
        print(f"\n--- Implied kernel speedup ---")
        avg_s = total_sparse/len(pages)
        avg_d = total_dense/len(dense_pages)
        if avg_d > 0:
            print(f"  sparse/dense avg = {avg_s/avg_d:.2f}x")
            if avg_s >= avg_d:
                print(f"  ⚠  Sparse kernel not faster than dense — either ratio≈1.0 "
                      f"or kernel overhead eats savings")

    return {
        "port": port,
        "target_tokens": approx_tokens,
        "actual_tokens": prompt_tokens,
        "wall_s": dt,
        "correct": "BLUE-TIGER-42" in answer,
        "n_sparse_detail": len(details),
        "n_sparse_paged": len(pages),
        "n_dense_paged": len(dense_pages),
        "ratios": [d["ratio"] for d in details],
        "sparse_kernel_s": sum(p["time"] for p in pages) if pages else 0,
        "dense_kernel_s": sum(p["time"] for p in dense_pages) if dense_pages else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31335)
    ap.add_argument("--input-tokens", type=int, default=80000,
                    help="Target prompt size in tokens (approximate)")
    ap.add_argument("--max-output", type=int, default=64)
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of runs (for stability)")
    args = ap.parse_args()

    all_results = []
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n{'='*60}")
            print(f"Run {i+1}/{args.runs}")
            print('='*60)
        res = run_test(args.port, args.input_tokens, args.max_output)
        all_results.append(res)
        if args.runs > 1 and i < args.runs - 1:
            time.sleep(2)

    if args.runs > 1:
        print(f"\n{'='*60}")
        print(f"Summary across {args.runs} runs")
        print('='*60)
        for r in all_results:
            ratio_str = (f"ratio min/max={min(r['ratios']):.2f}/{max(r['ratios']):.2f}"
                         if r['ratios'] else "no ratios")
            print(f"  wall={r['wall_s']:.2f}s  correct={r['correct']}  {ratio_str}")


if __name__ == "__main__":
    main()
