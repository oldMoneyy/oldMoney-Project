#!/usr/bin/env python3
"""
Analyze SOAR eval predictions with tokenizer — per-task breakdown, failure modes,
response content analysis, score distribution.

Usage:
    python analyze_eval.py \
        --predictions /opt/oldMoney-Project/SOAR-Toolkit/outputs/20260328_161149/predictions.jsonl \
        --tokenizer-path /opt/model
"""

import json
import re
import argparse
from collections import defaultdict, Counter


def analyze(predictions_path, tokenizer_path):
    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"[INFO] Loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        print(f"[WARN] Tokenizer unavailable ({e}), skipping token-level analysis")

    samples = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    total = len(samples)
    print(f"\n{'=' * 90}")
    print(f"  EVAL ANALYSIS: {predictions_path}")
    print(f"  Total samples: {total}")
    print(f"{'=' * 90}")

    # ---- Tokenize predictions for deeper analysis ----
    for s in samples:
        prediction = s.get("prediction", "")
        question = s.get("question", "")

        # Token counts from eval tool
        s["_in_tok"] = s.get("prompt_tokens", 0)
        s["_out_tok"] = s.get("completion_tokens", 0)

        # Content analysis
        s["_has_think_open"] = "<think>" in prediction
        s["_has_think_close"] = "</think>" in prediction
        s["_pred_len"] = len(prediction)
        s["_is_empty"] = len(prediction.strip()) < 10
        s["_is_null"] = prediction is None or prediction == ""

        # Check for <unk> / token-0 collapse
        if tokenizer:
            pred_tokens = tokenizer.encode(prediction)
            s["_pred_tokens"] = len(pred_tokens)
            s["_unk_count"] = pred_tokens.count(0)
            s["_unk_ratio"] = s["_unk_count"] / max(len(pred_tokens), 1)
        else:
            s["_pred_tokens"] = s["_out_tok"]
            s["_unk_count"] = 0
            s["_unk_ratio"] = 0.0

        # Check for repetition loops
        if len(prediction) > 500:
            last_500 = prediction[-500:]
            chunks = [last_500[i:i+50] for i in range(0, len(last_500)-50, 50)]
            chunk_counts = Counter(chunks)
            s["_max_repeat"] = max(chunk_counts.values()) if chunk_counts else 0
        else:
            s["_max_repeat"] = 0

    # ---- Classify failure mode per sample ----
    for s in samples:
        score = s.get("score", 0)
        if score >= 0.99:
            s["_failure_mode"] = "correct"
        elif s["_is_null"] or s["_is_empty"]:
            s["_failure_mode"] = "empty_response"
        elif s["_unk_ratio"] > 0.5:
            s["_failure_mode"] = "token0_collapse"
        elif s["_out_tok"] >= 60000 and score < 0.5:
            s["_failure_mode"] = "max_tokens_exhaust"
        elif s["_max_repeat"] >= 3:
            s["_failure_mode"] = "repetition_loop"
        elif s.get("extracted") is None:
            s["_failure_mode"] = "extraction_failed"
        elif score > 0:
            s["_failure_mode"] = "partial_correct"
        else:
            s["_failure_mode"] = "wrong_answer"

    # ---- Per-task breakdown ----
    tasks = defaultdict(lambda: {
        "scores": [], "total": 0,
        "in_tokens": [], "out_tokens": [], "pred_tokens": [],
        "failure_modes": Counter(),
    })

    for s in samples:
        task = s.get("task", "unknown")
        tasks[task]["scores"].append(s.get("score", 0))
        tasks[task]["total"] += 1
        tasks[task]["in_tokens"].append(s["_in_tok"])
        tasks[task]["out_tokens"].append(s["_out_tok"])
        tasks[task]["pred_tokens"].append(s["_pred_tokens"])
        tasks[task]["failure_modes"][s["_failure_mode"]] += 1

    task_order = ["mcq", "niah", "qa", "fwe", "cwe"]
    header = f"{'Task':<6} {'Score':>6} {'Perfect':>8} {'None':>6} {'Collapse':>9} {'AvgIn':>8} {'AvgOut':>8} {'AvgPred':>8}"
    print(f"\n{header}")
    print("-" * 90)

    all_scores = []
    for task in task_order:
        if task not in tasks:
            continue
        t = tasks[task]
        avg = sum(t["scores"]) / len(t["scores"]) * 100
        perfect = sum(1 for sc in t["scores"] if sc >= 0.99)
        none_c = t["failure_modes"].get("extraction_failed", 0) + t["failure_modes"].get("empty_response", 0) + t["failure_modes"].get("token0_collapse", 0)
        collapse = t["failure_modes"].get("token0_collapse", 0) + t["failure_modes"].get("max_tokens_exhaust", 0)
        avg_in = sum(t["in_tokens"]) // max(len(t["in_tokens"]), 1)
        avg_out = sum(t["out_tokens"]) // max(len(t["out_tokens"]), 1)
        avg_pred = sum(t["pred_tokens"]) // max(len(t["pred_tokens"]), 1)
        print(f"{task:<6} {avg:>5.1f}% {perfect:>5}/{t['total']:<3} {none_c:>6} {collapse:>9} {avg_in:>8,} {avg_out:>8,} {avg_pred:>8,}")
        all_scores.extend(t["scores"])

    overall = sum(all_scores) / len(all_scores) * 100
    print("-" * 90)
    print(f"{'TOTAL':<6} {overall:>5.1f}%")

    # ---- Failure mode summary ----
    print(f"\n  Failure mode breakdown:")
    all_modes = Counter()
    for s in samples:
        all_modes[s["_failure_mode"]] += 1
    for mode, cnt in all_modes.most_common():
        pct = cnt / total * 100
        bar = "#" * (cnt // 2)
        print(f"    {mode:<22} {cnt:>4} ({pct:>5.1f}%)  {bar}")

    # ---- Per-task failure modes ----
    print(f"\n  Per-task failure modes:")
    for task in task_order:
        if task not in tasks:
            continue
        t = tasks[task]
        modes_str = ", ".join(f"{m}={c}" for m, c in t["failure_modes"].most_common())
        print(f"    {task}: {modes_str}")

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
        bar = "#" * (c)
        print(f"    {b:<16} {c:>4}  {bar}")

    # ---- Token-0 / unk analysis ----
    if tokenizer:
        unk_samples = [s for s in samples if s["_unk_count"] > 10]
        print(f"\n  Token-0 (<unk>) analysis:")
        print(f"    Samples with >10 <unk> tokens: {len(unk_samples)}")
        if unk_samples:
            print(f"    {'Idx':>4} {'Task':<6} {'Score':>6} {'UNK':>6} {'Ratio':>7} {'OutTok':>8}")
            for s in sorted(unk_samples, key=lambda x: x["_unk_count"], reverse=True)[:15]:
                idx = samples.index(s)
                print(f"    {idx:>4} {s.get('task','?'):<6} {s.get('score',0):>5.2f} {s['_unk_count']:>6} {s['_unk_ratio']:>6.1%} {s['_out_tok']:>8,}")

    # ---- Bottom 10 samples ----
    print(f"\n  Bottom 10 samples (lowest scores):")
    sorted_samples = sorted(enumerate(samples), key=lambda x: x[1].get("score", 0))
    for idx, s in sorted_samples[:10]:
        task = s.get("task", "?")
        score = s.get("score", 0)
        mode = s["_failure_mode"]
        extracted = str(s.get("extracted", "None"))[:40]
        print(f"    [{idx:>3}] {task:<5} score={score:.2f} mode={mode:<20} extracted={extracted}")

    # ---- Top 10 samples ----
    print(f"\n  Top 10 samples (highest scores, non-perfect):")
    non_perfect = [(i, s) for i, s in enumerate(samples) if s.get("score", 0) < 0.99 and s.get("score", 0) > 0]
    non_perfect.sort(key=lambda x: x[1].get("score", 0), reverse=True)
    for idx, s in non_perfect[:10]:
        task = s.get("task", "?")
        score = s.get("score", 0)
        extracted = str(s.get("extracted", "None"))[:40]
        gold = str(s.get("gold", ""))[:40]
        print(f"    [{idx:>3}] {task:<5} score={score:.2f} extracted={extracted}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--tokenizer-path", default="/opt/model", help="Path to tokenizer")
    args = parser.parse_args()
    analyze(args.predictions, args.tokenizer_path)
