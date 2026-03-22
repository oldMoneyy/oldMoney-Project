#!/usr/bin/env python3
"""
Side-by-side comparison: Original BF16 vs AWQ NVFP4

Run:
  python compare_models.py \
    --original /opt/SOAR-Toolkit/outputs/20260322_020604/predictions.jsonl \
    --awq /opt/SOAR-Toolkit/outputs/20260321_163020/predictions.jsonl
"""
import json
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--awq", required=True)
    args = parser.parse_args()

    orig = []
    with open(args.original) as f:
        for line in f:
            if line.strip(): orig.append(json.loads(line.strip()))
    awq = []
    with open(args.awq) as f:
        for line in f:
            if line.strip(): awq.append(json.loads(line.strip()))

    assert len(orig) == len(awq), f"Length mismatch: {len(orig)} vs {len(awq)}"

    # Per-task scores
    from collections import defaultdict
    orig_scores = defaultdict(list)
    awq_scores = defaultdict(list)
    for o, a in zip(orig, awq):
        task = o.get("task", "").lower()
        orig_scores[task].append(o.get("score", 0))
        awq_scores[task].append(a.get("score", 0))

    print("=" * 80)
    print("PER-TASK COMPARISON")
    print("=" * 80)
    print(f"  {'Task':<8} {'Original':>10} {'AWQ':>10} {'Delta':>10} {'Lost samples':>15}")
    total_lost = 0
    for task in sorted(orig_scores.keys()):
        o_avg = sum(orig_scores[task]) / len(orig_scores[task]) * 100
        a_avg = sum(awq_scores[task]) / len(awq_scores[task]) * 100
        delta = a_avg - o_avg
        lost = sum(1 for o, a in zip(orig_scores[task], awq_scores[task]) 
                   if o > 0.5 and a < 0.5)  # samples that went from pass to fail
        total_lost += lost
        print(f"  {task.upper():<8} {o_avg:>9.1f}% {a_avg:>9.1f}% {delta:>+9.1f}% {lost:>10}")
    
    o_total = sum(sum(v) for v in orig_scores.values()) / len(orig) * 100
    a_total = sum(sum(v) for v in awq_scores.values()) / len(awq) * 100
    print(f"  {'OVERALL':<8} {o_total:>9.1f}% {a_total:>9.1f}% {a_total-o_total:>+9.1f}% {total_lost:>10}")

    # MCQ: Sample-level comparison
    print("\n" + "=" * 80)
    print("MCQ SAMPLE-LEVEL COMPARISON (30 samples)")
    print("=" * 80)
    print(f"  {'#':<4} {'Orig':>6} {'AWQ':>6} {'Change':>8} {'Orig_len':>10} {'AWQ_len':>10} {'Orig_extr':>10} {'AWQ_extr':>10} {'Gold':>6}")
    
    mcq_degraded = []
    mcq_improved = []
    for o, a in zip(orig, awq):
        if o.get("task", "").lower() != "mcq":
            continue
        idx = o.get("index", "?")
        o_score = o.get("score", 0)
        a_score = a.get("score", 0)
        o_len = len(o.get("prediction", ""))
        a_len = len(a.get("prediction", ""))
        o_extr = o.get("extracted", "NONE")
        a_extr = a.get("extracted", "NONE")
        gold = o.get("gold", "?")
        
        if o_score > a_score:
            change = "WORSE"
            mcq_degraded.append((idx, o, a))
        elif a_score > o_score:
            change = "BETTER"
            mcq_improved.append((idx, o, a))
        else:
            change = "same"
        
        if o_extr is None: o_extr = "NONE"
        if a_extr is None: a_extr = "NONE"
        
        print(f"  {idx:<4} {o_score:>6.1f} {a_score:>6.1f} {change:>8} {o_len:>10} {a_len:>10} {str(o_extr):>10} {str(a_extr):>10} {gold:>6}")

    print(f"\n  MCQ samples that DEGRADED (orig pass -> AWQ fail): {len(mcq_degraded)}")
    print(f"  MCQ samples that IMPROVED (orig fail -> AWQ pass): {len(mcq_improved)}")

    # For degraded MCQ samples: show what happened
    print("\n" + "=" * 80)
    print("MCQ DEGRADED SAMPLES (passed in BF16, failed in AWQ)")
    print("=" * 80)
    for idx, o, a in mcq_degraded:
        o_pred = o.get("prediction", "")
        a_pred = a.get("prediction", "")
        gold = o.get("gold", "?")
        o_extr = o.get("extracted", "NONE")
        a_extr = a.get("extracted", "NONE")
        
        o_has_think_close = '</think>' in o_pred
        a_has_think_close = '</think>' in a_pred
        o_has_answer = bool(re.search(r'ANSWER:\s*[A-D]', o_pred))
        a_has_answer = bool(re.search(r'ANSWER:\s*[A-D]', a_pred))
        
        print(f"\n  Sample {idx}: gold={gold}")
        print(f"    Original: {len(o_pred)} chars, extr={o_extr}, </think>={'Y' if o_has_think_close else 'N'}, ANSWER={'Y' if o_has_answer else 'N'}")
        print(f"    AWQ:      {len(a_pred)} chars, extr={a_extr}, </think>={'Y' if a_has_think_close else 'N'}, ANSWER={'Y' if a_has_answer else 'N'}")
        
        # Classify AWQ failure
        if a_has_answer and a_extr and a_extr != gold:
            print(f"    -> AWQ chose {a_extr} instead of {gold}: WRONG ANSWER (hallucination)")
        elif not a_has_think_close:
            print(f"    -> AWQ never closed </think>: REASONING LOOP / TRUNCATION")
            # Show last 200 chars of AWQ output
            print(f"    -> AWQ ending: ...{a_pred[-200:]}")
        elif not a_has_answer:
            print(f"    -> AWQ closed </think> but no ANSWER line")
            after_think = a_pred.split('</think>')[-1].strip()
            print(f"    -> After </think>: {after_think[:200]}")

    # QA comparison
    print("\n" + "=" * 80)
    print("QA DEGRADED SAMPLES")
    print("=" * 80)
    for o, a in zip(orig, awq):
        if o.get("task", "").lower() != "qa":
            continue
        o_score = o.get("score", 0)
        a_score = a.get("score", 0)
        if o_score <= a_score:
            continue  # Not degraded
        
        idx = o.get("index", "?")
        gold = o.get("gold", "?")
        
        # Get actual answers (after </think>)
        o_pred = o.get("prediction", "")
        a_pred = a.get("prediction", "")
        o_answer = o_pred.split('</think>')[-1].strip() if '</think>' in o_pred else o_pred[-200:]
        a_answer = a_pred.split('</think>')[-1].strip() if '</think>' in a_pred else a_pred[-200:]
        
        print(f"\n  Sample {idx}: gold={str(gold)[:80]}")
        print(f"    Original answer: {o_answer[:150]}")
        print(f"    AWQ answer:      {a_answer[:150]}")

if __name__ == "__main__":
    main()