#!/usr/bin/env python3
"""
Deep failure mode classification.
Categorizes every failure into: hallucination, reasoning loop, truncation,
near-miss extraction, or wrong answer.

Usage:
  python opt/oldMoney-Project/quantization/calibration/classify_failures.py --predictions /opt/SOAR-Toolkit/outputs/20260321_163020/predictions.jsonl
"""
import json
import re
import argparse
from collections import defaultdict, Counter

def classify_mcq_failure(pred):
    """Classify an MCQ failure into specific failure mode."""
    output = pred.get("prediction", "")
    gold = pred.get("gold", "")
    extracted = pred.get("extracted", None)
    score = pred.get("score", 0)
    
    # Check if output has ANSWER line at all
    answer_match = re.search(r'ANSWER:\s*([A-D])', output)
    has_think_open = '<think>' in output
    has_think_close = '</think>' in output
    output_len = len(output)
    
    # Check for repetition patterns (reasoning loops)
    # Look for repeated phrases
    sentences = output.split('.')
    if len(sentences) > 10:
        last_20 = sentences[-20:]
        # Check if similar phrases repeat
        phrase_counts = Counter()
        for s in last_20:
            s_clean = s.strip()[:50]
            if len(s_clean) > 20:
                phrase_counts[s_clean] += 1
        max_repeat = max(phrase_counts.values()) if phrase_counts else 0
    else:
        max_repeat = 0
    
    # Check if model is still mid-calculation at the end
    last_200 = output[-200:] if len(output) > 200 else output
    ends_mid_sentence = not last_200.rstrip().endswith(('.', '!', '?', ')', '"', "'", 'D'))
    ends_mid_math = any(c in last_200[-50:] for c in ['=', '+', '-', '*', '/', '^']) if len(last_200) > 50 else False
    
    if answer_match and extracted and extracted != gold:
        # Model gave a definitive answer, but it's WRONG
        return "WRONG_ANSWER", f"chose {extracted}, gold={gold}"
    
    elif not answer_match and has_think_open and not has_think_close:
        # Never closed <think> tag - still reasoning
        if max_repeat >= 3:
            return "REASONING_LOOP", f"stuck in loop, {output_len} chars, never closed </think>"
        elif ends_mid_math or ends_mid_sentence:
            return "TRUNCATED_MID_REASONING", f"hit token limit at {output_len} chars, still calculating"
        else:
            return "INCONCLUSIVE_REASONING", f"reasoning petered out at {output_len} chars"
    
    elif not answer_match and has_think_close:
        # Closed think but never wrote ANSWER line
        # Check what's after </think>
        after_think = output.split('</think>')[-1].strip()
        if len(after_think) < 10:
            return "FORGOT_ANSWER_LINE", f"closed </think> but wrote nothing after: '{after_think[:100]}'"
        else:
            return "WRONG_FORMAT", f"wrote after </think> but no ANSWER line: '{after_think[:100]}'"
    
    elif not answer_match:
        return "NO_ANSWER_PRODUCED", f"{output_len} chars, no ANSWER line found"
    
    else:
        return "OTHER", f"score={score}, extracted={extracted}"


def classify_qa_failure(pred):
    """Classify a QA failure."""
    output = pred.get("prediction", "")
    gold = pred.get("gold", [])
    extracted = pred.get("extracted", None)
    
    # Get the actual answer (after </think>)
    if '</think>' in output:
        answer_part = output.split('</think>')[-1].strip()
    else:
        answer_part = output.strip()
    
    gold_strs = gold if isinstance(gold, list) else [gold]
    
    # Check for near-misses
    answer_lower = answer_part.lower()
    for g in gold_strs:
        g_lower = g.lower().strip()
        if g_lower in answer_lower:
            return "NEAR_MISS_CONTAINED", f"answer '{answer_part[:80]}' contains gold '{g}' but extraction failed"
        # Check without articles/quotes
        g_clean = re.sub(r'^(the|a|an|")\s*', '', g_lower).rstrip('"')
        a_clean = re.sub(r'^(the|a|an|")\s*', '', answer_lower).rstrip('"')
        if g_clean in a_clean or a_clean in g_clean:
            return "NEAR_MISS_FUZZY", f"answer '{answer_part[:80]}' ~= gold '{g}'"
    
    # Check if answer is topically related but wrong
    if len(answer_part) < 200:
        return "WRONG_ANSWER_SHORT", f"answer: '{answer_part[:150]}', gold: '{gold_strs[0][:80]}'"
    else:
        return "WRONG_ANSWER_LONG", f"answer ({len(answer_part)} chars): '{answer_part[:100]}...', gold: '{gold_strs[0][:80]}'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    preds = []
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line.strip()))

    # ==========================================
    # MCQ Analysis
    # ==========================================
    print("=" * 80)
    print("MCQ FAILURE CLASSIFICATION (17 failures)")
    print("=" * 80)
    
    mcq_modes = defaultdict(list)
    mcq_all = []
    for i, p in enumerate(preds):
        if p.get("task", "").lower() != "mcq":
            continue
        mcq_all.append((i+1, p))
        if p.get("score", 1) >= 0.5:
            continue
        mode, detail = classify_mcq_failure(p)
        mcq_modes[mode].append((i+1, detail))
    
    for mode in sorted(mcq_modes.keys()):
        items = mcq_modes[mode]
        print(f"\n  [{mode}] ({len(items)} samples)")
        for idx, detail in items:
            print(f"    Sample {idx}: {detail}")
    
    # Summary
    print(f"\n  MCQ FAILURE SUMMARY:")
    total_fail = sum(len(v) for v in mcq_modes.values())
    for mode, items in sorted(mcq_modes.items(), key=lambda x: -len(x[1])):
        pct = len(items) / total_fail * 100
        print(f"    {mode}: {len(items)} ({pct:.0f}%)")

    # MCQ output length distribution
    print(f"\n  MCQ OUTPUT LENGTH DISTRIBUTION:")
    for idx, p in mcq_all:
        output = p.get("prediction", "")
        score = p.get("score", 0)
        extracted = p.get("extracted", None)
        gold = p.get("gold", "?")
        has_answer = bool(re.search(r'ANSWER:\s*[A-D]', output))
        has_think_close = '</think>' in output
        marker = "✓" if score > 0 else "✗"
        extr_str = extracted if extracted else "NONE"
        print(f"    {marker} Sample {idx}: {len(output):>6} chars | gold={gold} extr={extr_str} | </think>={'Y' if has_think_close else 'N'} ANSWER={'Y' if has_answer else 'N'}")

    # ==========================================
    # QA Analysis
    # ==========================================
    print("\n" + "=" * 80)
    print("QA FAILURE CLASSIFICATION (15 failures)")
    print("=" * 80)
    
    qa_modes = defaultdict(list)
    for i, p in enumerate(preds):
        if p.get("task", "").lower() != "qa":
            continue
        if p.get("score", 1) >= 0.5:
            continue
        mode, detail = classify_qa_failure(p)
        qa_modes[mode].append((i+1, detail))
    
    for mode in sorted(qa_modes.keys()):
        items = qa_modes[mode]
        print(f"\n  [{mode}] ({len(items)} samples)")
        for idx, detail in items:
            print(f"    Sample {idx}: {detail}")
    
    print(f"\n  QA FAILURE SUMMARY:")
    total_fail = sum(len(v) for v in qa_modes.values())
    for mode, items in sorted(qa_modes.items(), key=lambda x: -len(x[1])):
        pct = len(items) / total_fail * 100
        print(f"    {mode}: {len(items)} ({pct:.0f}%)")

    # ==========================================
    # POINTS RECOVERY ESTIMATE
    # ==========================================
    print("\n" + "=" * 80)
    print("POINTS RECOVERY ESTIMATE")
    print("=" * 80)
    
    # MCQ: which failures are potentially recoverable?
    recoverable_mcq = 0
    for mode, items in mcq_modes.items():
        if mode in ("TRUNCATED_MID_REASONING", "REASONING_LOOP", "INCONCLUSIVE_REASONING", 
                     "FORGOT_ANSWER_LINE", "NO_ANSWER_PRODUCED"):
            # These might be fixed by allowing more tokens or better quantization
            recoverable_mcq += len(items)
        # WRONG_ANSWER is harder - model actually got it wrong
    
    # QA: which are near-misses?
    recoverable_qa = 0
    for mode, items in qa_modes.items():
        if "NEAR_MISS" in mode:
            recoverable_qa += len(items)
    
    current = 74.44
    mcq_possible = recoverable_mcq / 150 * 100  # as % of total
    qa_possible = recoverable_qa / 150 * 100
    
    print(f"  Current score:          {current:.1f}%")
    print(f"  MCQ recoverable:        {recoverable_mcq} samples = +{mcq_possible:.1f}% if all fixed")
    print(f"  QA near-misses:         {recoverable_qa} samples = +{qa_possible:.1f}% if extraction fixed")
    print(f"  Theoretical ceiling:    {current + mcq_possible + qa_possible:.1f}%")
    print(f"  Target:                 79.0%")
    print(f"  Gap to close:           {79.0 - current:.1f}%")

if __name__ == "__main__":
    main()