#!/usr/bin/env python3
"""Analyze eval_model.py output to find straggler requests (long output, low score).

Usage:
    python scripts/analyze_eval_results.py <predictions.jsonl>

Example:
    python scripts/analyze_eval_results.py outputs/20260408_064127/predictions.jsonl
"""

import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_eval_results.py <predictions.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Total samples: {len(results)}")
    total_out = sum(r["output_tokens"] for r in results)
    print(f"Total output tokens: {total_out}")
    print(f"Average output tokens: {total_out / len(results):.0f}")
    print()

    # Sort by output tokens descending
    by_out = sorted(results, key=lambda r: r["output_tokens"], reverse=True)

    # Top 20 longest outputs
    print("=" * 90)
    print(f"{'Rank':>4} {'Idx':>4} {'Task':>8} {'Score':>6} {'InTok':>8} {'OutTok':>8} {'ThinkTok':>9} {'Answer':>20}")
    print("=" * 90)

    for rank, r in enumerate(by_out[:30], 1):
        pred = r.get("prediction", "")
        # Count thinking tokens (rough: chars before </think>)
        parts = pred.split("</think>")
        if len(parts) > 1:
            think_len = len(parts[0])
            answer = parts[-1].strip()[:60]
        else:
            think_len = 0
            answer = pred.strip()[:60]

        # Estimate thinking tokens (rough: chars / 4)
        think_tokens_est = think_len // 4

        print(f"{rank:>4} {r['index']:>4} {r['task']:>8} {r['score']:>6.2f} "
              f"{r['input_tokens']:>8} {r['output_tokens']:>8} "
              f"{think_tokens_est:>9} {answer[:20]:>20}")

    print()

    # Summary by output token buckets
    buckets = [
        (0, 1000, "< 1K"),
        (1000, 4000, "1K-4K"),
        (4000, 8000, "4K-8K"),
        (8000, 16000, "8K-16K"),
        (16000, 32000, "16K-32K"),
        (32000, 65536, "32K-64K"),
        (65536, 999999, "> 64K"),
    ]

    print("=" * 70)
    print(f"{'Bucket':>10} {'Count':>6} {'TotalOut':>10} {'AvgScore':>9} {'AvgOut':>8}")
    print("=" * 70)

    for lo, hi, label in buckets:
        group = [r for r in results if lo <= r["output_tokens"] < hi]
        if not group:
            continue
        total = sum(r["output_tokens"] for r in group)
        avg_score = sum(r["score"] for r in group) / len(group) * 100
        avg_out = total / len(group)
        print(f"{label:>10} {len(group):>6} {total:>10} {avg_score:>8.1f}% {avg_out:>8.0f}")

    print()

    # Key insight: what % of total output tokens come from top N requests
    print("=" * 70)
    print("Cumulative output token concentration:")
    print("=" * 70)
    cum = 0
    for rank, r in enumerate(by_out, 1):
        cum += r["output_tokens"]
        pct = cum / total_out * 100
        if rank in [1, 2, 3, 5, 10, 15, 20, 30, 50]:
            print(f"  Top {rank:>3} requests: {cum:>10} tokens = {pct:>5.1f}% of total output")

    print()

    # Suggestions
    long_low = [r for r in results if r["output_tokens"] > 16000 and r["score"] < 0.5]
    if long_low:
        waste = sum(r["output_tokens"] for r in long_low)
        print(f"⚠  {len(long_low)} requests have >16K output AND score < 50%")
        print(f"   They waste {waste} output tokens ({waste/total_out*100:.1f}% of total)")
        print(f"   Capping these at 8K would save ~{waste - len(long_low)*8000} tokens")
        print(f"   Tasks: {[r['task'] for r in long_low]}")

if __name__ == "__main__":
    main()
