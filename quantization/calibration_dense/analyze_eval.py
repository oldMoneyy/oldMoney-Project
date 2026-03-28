#!/usr/bin/env python3
"""
Analyze SOAR eval predictions — per-task breakdown, failure modes, score distribution.

Usage:
    python analyze_eval.py --predictions /opt/oldMoney-Project/SOAR-Toolkit/outputs/YYYYMMDD_HHMMSS/predictions.jsonl
    python analyze_eval.py --predictions /opt/oldMoney-Project/SOAR-Toolkit/outputs/20260328_161149/predictions.jsonl
"""

import json
import argparse
from collections import defaultdict


def analyze(predictions_path):
    samples = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    total = len(samples)
    print(f"{'=' * 80}")
    print(f"  EVAL ANALYSIS: {predictions_path}")
    print(f"  Total samples: {total}")
    print(f"{'=' * 80}")

    # ---- Per-task breakdown ----
    tasks = defaultdict(lambda: {
        "scores": [], "none_count": 0, "total": 0,
        "in_tokens": [], "out_tokens": [],
        "collapse_count": 0,  # Out >= 60000 (likely token-0 collapse)
        "correct": 0,
    })

    for s in samples:
        task = s.get("task", "unknown")
        score = s.get("score", 0)
        extracted = s.get("extracted", None)
        in_tok = s.get("prompt_tokens", s.get("in_tokens", 0))
        out_tok = s.get("completion_tokens", s.get("out_tokens", 0))

        tasks[task]["scores"].append(score)
        tasks[task]["total"] += 1
        tasks[task]["in_tokens"].append(in_tok)
        tasks[task]["out_tokens"].append(out_tok)

        if extracted is None or extracted == "None":
            tasks[task]["none_count"] += 1
        if score >= 0.99:
            tasks[task]["correct"] += 1
        if out_tok >= 60000:
            tasks[task]["collapse_count"] += 1

    print(f"\n{'Task':<8} {'Avg%':>6} {'Correct':>8} {'None':>6} {'Collapse':>9} {'Total':>6}  {'AvgIn':>8} {'AvgOut':>8}")
    print("-" * 80)

    all_scores = []
    task_order = ["mcq", "niah", "qa", "fwe", "cwe"]
    for task in task_order:
        if task not in tasks:
            continue
        t = tasks[task]
        avg = sum(t["scores"]) / len(t["scores"]) * 100
        avg_in = sum(t["in_tokens"]) // max(len(t["in_tokens"]), 1)
        avg_out = sum(t["out_tokens"]) // max(len(t["out_tokens"]), 1)
        print(f"{task:<8} {avg:>5.1f}% {t['correct']:>6}/{t['total']:<2} {t['none_count']:>6} {t['collapse_count']:>9} {t['total']:>6}  {avg_in:>8,} {avg_out:>8,}")
        all_scores.extend(t["scores"])

    # Any unknown tasks
    for task in sorted(tasks.keys()):
        if task not in task_order:
            t = tasks[task]
            avg = sum(t["scores"]) / len(t["scores"]) * 100
            print(f"{task:<8} {avg:>5.1f}% {t['correct']:>6}/{t['total']:<2} {t['none_count']:>6} {t['collapse_count']:>9} {t['total']:>6}")
            all_scores.extend(t["scores"])

    overall = sum(all_scores) / len(all_scores) * 100
    print("-" * 80)
    print(f"{'TOTAL':<8} {overall:>5.1f}%")

    # ---- Score distribution ----
    print(f"\n  Score distribution:")
    buckets = {"1.0 (perfect)": 0, "0.7-0.99": 0, "0.3-0.69": 0, "0.01-0.29": 0, "0.0 (zero)": 0}
    for sc in all_scores:
        if sc >= 0.99:
            buckets["1.0 (perfect)"] += 1
        elif sc >= 0.7:
            buckets["0.7-0.99"] += 1
        elif sc >= 0.3:
            buckets["0.3-0.69"] += 1
        elif sc > 0.001:
            buckets["0.01-0.29"] += 1
        else:
            buckets["0.0 (zero)"] += 1
    for b, c in buckets.items():
        bar = "#" * (c * 2)
        print(f"    {b:<16} {c:>4}  {bar}")

    # ---- Failure analysis ----
    print(f"\n  Failure modes:")
    collapse_samples = [s for s in samples if s.get("completion_tokens", s.get("out_tokens", 0)) >= 60000]
    none_samples = [s for s in samples if s.get("extracted", None) is None or s.get("extracted") == "None"]
    zero_samples = [s for s in samples if s.get("score", 0) < 0.001]

    print(f"    Token collapse (Out>=60k):  {len(collapse_samples)}")
    print(f"    Extracted=None:             {len(none_samples)}")
    print(f"    Score=0:                    {len(zero_samples)}")

    # ---- Worst samples ----
    print(f"\n  Bottom 10 samples (lowest scores):")
    sorted_samples = sorted(enumerate(samples), key=lambda x: x[1].get("score", 0))
    for idx, s in sorted_samples[:10]:
        task = s.get("task", "?")
        score = s.get("score", 0)
        in_tok = s.get("prompt_tokens", s.get("in_tokens", 0))
        out_tok = s.get("completion_tokens", s.get("out_tokens", 0))
        extracted = str(s.get("extracted", "None"))[:50]
        print(f"    [{idx:>3}] {task:<5} score={score:.2f} in={in_tok:>7,} out={out_tok:>6,} extracted={extracted}")

    # ---- Best samples per task ----
    print(f"\n  Per-task score summary:")
    for task in task_order:
        if task not in tasks:
            continue
        t = tasks[task]
        scores = sorted(t["scores"])
        perfect = sum(1 for s in scores if s >= 0.99)
        zero = sum(1 for s in scores if s < 0.001)
        print(f"    {task}: {perfect} perfect, {zero} zero, median={scores[len(scores)//2]:.2f}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    args = parser.parse_args()
    analyze(args.predictions)
