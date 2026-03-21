#!/usr/bin/env python3
"""
Run this on your machine:
python analyze_predictions.py \
    --predictions /opt/SOAR-Toolkit/outputs/20260321_163020/predictions.jsonl \
    --eval-data /opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl
"""
import json
import argparse
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--eval-data", default=None, help="Path to perf_public_set.jsonl (optional)")
    args = parser.parse_args()

    # Load predictions
    preds = []
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line.strip()))

    # Per-task analysis
    task_scores = defaultdict(list)
    task_failures = defaultdict(list)  # samples with score < 0.5
    task_tokens_out = defaultdict(list)
    task_extracted_none = defaultdict(int)
    task_degenerate = defaultdict(int)  # output > 10k tokens

    for i, p in enumerate(preds):
        task = p.get("task", p.get("Task", "unknown"))
        score = p.get("score", p.get("Score", 0))
        tokens_out = p.get("tokens_out", p.get("Tokens_out", p.get("out_tokens", 0)))
        extracted = p.get("extracted", p.get("Extracted", "exists"))

        task_scores[task].append(score)
        task_tokens_out[task].append(tokens_out)

        if score < 0.5:
            task_failures[task].append({
                "sample_idx": i + 1,
                "score": score,
                "tokens_out": tokens_out,
                "gold": p.get("gold", p.get("Gold", "?")),
                "extracted": str(extracted)[:200],
            })

        if extracted is None or extracted == "None" or str(extracted).strip() == "":
            task_extracted_none[task] += 1

        if isinstance(tokens_out, (int, float)) and tokens_out > 10000:
            task_degenerate[task] += 1

    # Print summary
    print("=" * 80)
    print("PER-TASK ACCURACY BREAKDOWN")
    print("=" * 80)
    
    total_score = 0
    total_count = 0
    for task in sorted(task_scores.keys()):
        scores = task_scores[task]
        avg = sum(scores) / len(scores) * 100
        num_fail = len(task_failures[task])
        none_count = task_extracted_none[task]
        degen_count = task_degenerate[task]
        avg_out = sum(task_tokens_out[task]) / max(len(task_tokens_out[task]), 1)
        
        total_score += sum(scores)
        total_count += len(scores)
        
        print(f"\n  {task.upper()}: {avg:.1f}% ({len(scores)} samples)")
        print(f"    Failures (score<0.5): {num_fail}")
        print(f"    Extracted=None:       {none_count}")
        print(f"    Degenerate (>10k):    {degen_count}")
        print(f"    Avg output tokens:    {avg_out:.0f}")

    overall = total_score / max(total_count, 1) * 100
    print(f"\n  OVERALL: {overall:.2f}% ({total_count} samples)")

    # Detailed failure analysis
    print("\n" + "=" * 80)
    print("DETAILED FAILURE ANALYSIS (score < 0.5)")
    print("=" * 80)
    
    for task in sorted(task_failures.keys()):
        if not task_failures[task]:
            continue
        print(f"\n--- {task.upper()} ---")
        for fail in task_failures[task]:
            print(f"  Sample {fail['sample_idx']}: score={fail['score']:.2f}, "
                  f"out_tokens={fail['tokens_out']}, "
                  f"gold={str(fail['gold'])[:80]}")
            if fail['extracted'] in ('None', '', 'null'):
                print(f"    -> EXTRACTION FAILED (model output not parseable)")

    # Quantization damage heatmap
    print("\n" + "=" * 80)
    print("QUANTIZATION DAMAGE ASSESSMENT")
    print("=" * 80)
    
    # Expected BF16 baselines (user said ~80% overall)
    # Estimate per-task targets assuming even distribution
    bf16_overall = 0.80
    
    for task in sorted(task_scores.keys()):
        scores = task_scores[task]
        avg = sum(scores) / len(scores)
        
        # Calculate the gap
        # If the task has long prompts (NIAH, QA, FWE, CWE), the damage is
        # likely from linear attention quantization
        if task.lower() in ("niah", "qa", "fwe", "cwe"):
            print(f"  {task.upper()}: {avg*100:.1f}% -- LONG-CONTEXT (lightning-attn layers critical)")
        elif task.lower() == "mcq":
            print(f"  {task.upper()}: {avg*100:.1f}% -- SHORT-CONTEXT (mainly MLP/attention damage)")
        else:
            print(f"  {task.upper()}: {avg*100:.1f}%")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    long_ctx_tasks = ["niah", "qa", "fwe", "cwe"]
    long_ctx_avg = []
    short_ctx_avg = []
    for task, scores in task_scores.items():
        avg = sum(scores) / len(scores)
        if task.lower() in long_ctx_tasks:
            long_ctx_avg.append(avg)
        else:
            short_ctx_avg.append(avg)
    
    if long_ctx_avg:
        lc = sum(long_ctx_avg) / len(long_ctx_avg) * 100
        print(f"\n  Long-context avg: {lc:.1f}%")
    if short_ctx_avg:
        sc = sum(short_ctx_avg) / len(short_ctx_avg) * 100
        print(f"  Short-context avg: {sc:.1f}%")
    
    if long_ctx_avg and short_ctx_avg:
        lc = sum(long_ctx_avg) / len(long_ctx_avg) * 100
        sc = sum(short_ctx_avg) / len(short_ctx_avg) * 100
        if lc < sc - 5:
            print(f"\n  >> Long-context tasks are {sc-lc:.1f}% worse than short-context!")
            print(f"  >> This confirms lightning-attn quantization is the primary issue.")
            print(f"  >> FIX: Better long-context calibration data + skip sensitive layers")
        elif sc < lc - 5:
            print(f"\n  >> Short-context tasks are {lc-sc:.1f}% worse than long-context!")
            print(f"  >> This suggests MLP quantization damage.")
            print(f"  >> FIX: Check MLP block scales, try larger search range")
        else:
            print(f"\n  >> Damage is spread evenly across all tasks.")
            print(f"  >> FIX: Improve calibration data quality + skip last 2-3 lightning layers")

if __name__ == "__main__":
    main()
