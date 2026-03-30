#!/usr/bin/env python3
"""
Compare NVFP4 (80.53%) vs GPTQ INT4 (78.69%) eval runs side-by-side.

Usage:
    python compare_nvfp4_vs_gptq.py \
        --nvfp4 SOAR-Toolkit/outputs/20260329_065254 \
        --gptq  SOAR-Toolkit/outputs/20260330_063815
"""

import json
import argparse
import os
from collections import defaultdict


def load_run(run_dir):
    preds = []
    with open(os.path.join(run_dir, "predictions.jsonl"), "r") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line.strip()))
    summary = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)
    return preds, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvfp4", required=True)
    parser.add_argument("--gptq", required=True)
    args = parser.parse_args()

    nvfp4, nvfp4_sum = load_run(args.nvfp4)
    gptq, gptq_sum = load_run(args.gptq)

    assert len(nvfp4) == len(gptq), f"Sample count mismatch: {len(nvfp4)} vs {len(gptq)}"
    N = len(nvfp4)

    tasks = ["mcq", "niah", "qa", "fwe", "cwe"]

    # ======================================================================
    # Section 1: Per-task accuracy comparison
    # ======================================================================
    print("=" * 90)
    print("  NVFP4 (80.53%) vs GPTQ INT4 (78.69%) — Per-Task Accuracy")
    print("=" * 90)

    print(f"\n  {'Task':<6} {'NVFP4':>8} {'GPTQ':>8} {'Delta':>8} {'NVFP4 Perfect':>14} {'GPTQ Perfect':>13}")
    print("  " + "-" * 60)

    for task in tasks:
        nv_task = [s for s in nvfp4 if s.get("task") == task]
        gp_task = [s for s in gptq if s.get("task") == task]
        nv_acc = sum(s["score"] for s in nv_task) / len(nv_task) * 100 if nv_task else 0
        gp_acc = sum(s["score"] for s in gp_task) / len(gp_task) * 100 if gp_task else 0
        nv_perfect = sum(1 for s in nv_task if s["score"] >= 0.99)
        gp_perfect = sum(1 for s in gp_task if s["score"] >= 0.99)
        delta = nv_acc - gp_acc
        sign = "+" if delta > 0 else ""
        print(f"  {task:<6} {nv_acc:>7.1f}% {gp_acc:>7.1f}% {sign}{delta:>6.1f}pp  {nv_perfect:>7}/{len(nv_task):<5} {gp_perfect:>6}/{len(gp_task)}")

    nv_total = sum(s["score"] for s in nvfp4) / N * 100
    gp_total = sum(s["score"] for s in gptq) / N * 100
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<6} {nv_total:>7.2f}% {gp_total:>7.2f}% {nv_total-gp_total:>+7.2f}pp")

    # ======================================================================
    # Section 2: Per-sample comparison — where do they disagree?
    # ======================================================================
    print("\n" + "=" * 90)
    print("  SECTION 2: PER-SAMPLE DISAGREEMENTS")
    print("=" * 90)

    nvfp4_wins = []  # NVFP4 better
    gptq_wins = []   # GPTQ better
    both_perfect = 0
    both_zero = 0
    both_partial = 0

    for i in range(N):
        nv_s = nvfp4[i]["score"]
        gp_s = gptq[i]["score"]
        diff = nv_s - gp_s

        if nv_s >= 0.99 and gp_s >= 0.99:
            both_perfect += 1
        elif nv_s < 0.01 and gp_s < 0.01:
            both_zero += 1
        elif abs(diff) < 0.01:
            both_partial += 1
        elif diff > 0:
            nvfp4_wins.append(i)
        else:
            gptq_wins.append(i)

    print(f"\n  Both perfect (>=0.99):  {both_perfect}")
    print(f"  Both zero (<0.01):     {both_zero}")
    print(f"  Both same (partial):   {both_partial}")
    print(f"  NVFP4 wins:            {len(nvfp4_wins)}")
    print(f"  GPTQ wins:             {len(gptq_wins)}")

    # NVFP4 wins detail
    if nvfp4_wins:
        print(f"\n  --- NVFP4 wins ({len(nvfp4_wins)} samples) ---")
        print(f"  {'Idx':>4} {'Task':<5} {'NVFP4':>6} {'GPTQ':>6} {'Delta':>7}  Gold (truncated)")
        print("  " + "-" * 70)
        nvfp4_wins.sort(key=lambda i: nvfp4[i]["score"] - gptq[i]["score"], reverse=True)
        for i in nvfp4_wins:
            gold = str(nvfp4[i].get("gold", ""))[:40]
            print(f"  {i:>4} {nvfp4[i].get('task','?'):<5} {nvfp4[i]['score']:>5.2f} {gptq[i]['score']:>5.2f} {nvfp4[i]['score']-gptq[i]['score']:>+6.2f}  {gold}")

    # GPTQ wins detail
    if gptq_wins:
        print(f"\n  --- GPTQ wins ({len(gptq_wins)} samples) ---")
        print(f"  {'Idx':>4} {'Task':<5} {'NVFP4':>6} {'GPTQ':>6} {'Delta':>7}  Gold (truncated)")
        print("  " + "-" * 70)
        gptq_wins.sort(key=lambda i: gptq[i]["score"] - nvfp4[i]["score"], reverse=True)
        for i in gptq_wins:
            gold = str(gptq[i].get("gold", ""))[:40]
            print(f"  {i:>4} {gptq[i].get('task','?'):<5} {nvfp4[i]['score']:>5.2f} {gptq[i]['score']:>5.2f} {gptq[i]['score']-nvfp4[i]['score']:>+6.2f}  {gold}")

    # ======================================================================
    # Section 3: Both-zero analysis — what's always broken?
    # ======================================================================
    print("\n" + "=" * 90)
    print("  SECTION 3: BOTH-ZERO SAMPLES (always broken, model-inherent)")
    print("=" * 90)

    both_zero_indices = [i for i in range(N) if nvfp4[i]["score"] < 0.01 and gptq[i]["score"] < 0.01]
    by_task = defaultdict(list)
    for i in both_zero_indices:
        by_task[nvfp4[i].get("task", "?")].append(i)

    print(f"\n  Total: {len(both_zero_indices)} samples always score 0 in both models")
    for task in tasks:
        if task in by_task:
            indices = by_task[task]
            print(f"  {task}: {len(indices)} — indices: {indices}")

    # ======================================================================
    # Section 4: Token usage comparison
    # ======================================================================
    print("\n" + "=" * 90)
    print("  SECTION 4: TOKEN USAGE COMPARISON")
    print("=" * 90)

    for task in tasks:
        nv_task = [s for s in nvfp4 if s.get("task") == task]
        gp_task = [s for s in gptq if s.get("task") == task]
        nv_out = sum(s.get("completion_tokens", s.get("output_tokens", 0)) for s in nv_task)
        gp_out = sum(s.get("completion_tokens", s.get("output_tokens", 0)) for s in gp_task)
        nv_avg = nv_out // max(len(nv_task), 1)
        gp_avg = gp_out // max(len(gp_task), 1)
        print(f"  {task:<6}  NVFP4 avg_out={nv_avg:>6}  GPTQ avg_out={gp_avg:>6}  diff={gp_avg-nv_avg:>+6}")

    # ======================================================================
    # Section 5: Extracted answer comparison for disagreements
    # ======================================================================
    print("\n" + "=" * 90)
    print("  SECTION 5: EXTRACTED ANSWERS FOR DISAGREEMENTS")
    print("=" * 90)

    all_disagree = sorted(nvfp4_wins + gptq_wins)
    for i in all_disagree[:30]:
        nv = nvfp4[i]
        gp = gptq[i]
        task = nv.get("task", "?")
        gold = str(nv.get("gold", ""))[:50]
        nv_ext = str(nv.get("extracted", "None"))[:50]
        gp_ext = str(gp.get("extracted", "None"))[:50]
        winner = "NVFP4" if nv["score"] > gp["score"] else "GPTQ"
        print(f"\n  [{i:>3}] task={task} winner={winner} (NV={nv['score']:.2f} GP={gp['score']:.2f})")
        print(f"        gold:      {gold}")
        print(f"        nvfp4_ext: {nv_ext}")
        print(f"        gptq_ext:  {gp_ext}")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    nv_margin = nv_total - 77.6  # 97% of 80 baseline
    gp_margin = gp_total - 77.6
    print(f"""
  NVFP4: {nv_total:.2f}% (margin above 97% threshold: {nv_margin:+.2f}pp)
  GPTQ:  {gp_total:.2f}% (margin above 97% threshold: {gp_margin:+.2f}pp)

  Agreement: {both_perfect} always-right, {len(both_zero_indices)} always-wrong, {both_partial} same-partial
  Disagreement: NVFP4 wins {len(nvfp4_wins)}, GPTQ wins {len(gptq_wins)}

  NVFP4 total score delta: {sum(nvfp4[i]['score'] - gptq[i]['score'] for i in nvfp4_wins):.2f} points gained
  GPTQ  total score delta: {sum(gptq[i]['score'] - nvfp4[i]['score'] for i in gptq_wins):.2f} points gained
""")


if __name__ == "__main__":
    main()
