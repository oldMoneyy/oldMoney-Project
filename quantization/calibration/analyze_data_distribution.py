#!/usr/bin/env python3
"""Analyze token length distributions of eval and calib datasets."""

import json
import sys
import numpy as np

def analyze_file(path, tokenizer, label, max_samples=None):
    lengths = []
    task_lengths = {}
    
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            row = json.loads(line.strip())
            
            # Get text field (eval uses "question", calib uses "question" too)
            text = row.get("question", row.get("prompt", row.get("text", "")))
            
            # Get task type if available
            task = row.get("task", "unknown")
            
            # Tokenize
            tokens = tokenizer(text, return_tensors="pt", truncation=False)
            n_tokens = tokens["input_ids"].shape[1]
            lengths.append(n_tokens)
            
            if task not in task_lengths:
                task_lengths[task] = []
            task_lengths[task].append(n_tokens)
    
    lengths = np.array(lengths)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  File: {path}")
    print(f"{'='*60}")
    print(f"  Total samples:  {len(lengths)}")
    print(f"  Total tokens:   {lengths.sum():,}")
    print(f"  Mean length:    {lengths.mean():,.0f}")
    print(f"  Median length:  {np.median(lengths):,.0f}")
    print(f"  Min length:     {lengths.min():,}")
    print(f"  Max length:     {lengths.max():,}")
    print(f"  Std length:     {lengths.std():,.0f}")
    
    # Distribution buckets
    buckets = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, float('inf')]
    bucket_labels = ["<512", "512-1K", "1K-2K", "2K-4K", "4K-8K", "8K-16K", "16K-32K", "32K-65K", "65K-131K", ">131K"]
    
    print(f"\n  Length distribution:")
    for j in range(len(bucket_labels)):
        count = ((lengths >= buckets[j]) & (lengths < buckets[j+1])).sum()
        pct = count / len(lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"    {bucket_labels[j]:>10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Per-task breakdown
    if len(task_lengths) > 1 or "unknown" not in task_lengths:
        print(f"\n  Per-task breakdown:")
        for task, tl in sorted(task_lengths.items()):
            tl = np.array(tl)
            print(f"    {task:>10s}: n={len(tl):3d}, "
                  f"mean={tl.mean():,.0f}, "
                  f"min={tl.min():,}, "
                  f"max={tl.max():,}")
    
    # Samples truncated at various max_len
    print(f"\n  Samples that would be truncated at:")
    for ml in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        n_trunc = (lengths > ml).sum()
        pct = n_trunc / len(lengths) * 100
        print(f"    max_len={ml:>6d}: {n_trunc:4d}/{len(lengths)} ({pct:.1f}%) truncated")
    
    return lengths


def main():
    from transformers import AutoTokenizer
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/model"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"
    calib_path = sys.argv[3] if len(sys.argv) > 3 else "/opt/oldMoney-Project/quantization/calibration/calib_dataset.jsonl"
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    eval_lens = analyze_file(eval_path, tokenizer, "EVAL DATA (perf_public_set)")
    calib_lens = analyze_file(calib_path, tokenizer, "CALIB DATA (calib_dataset)")
    
    # Coverage analysis
    print(f"\n{'='*60}")
    print(f"  COVERAGE ANALYSIS: Does calib cover eval?")
    print(f"{'='*60}")
    
    eval_p90 = np.percentile(eval_lens, 90)
    eval_p50 = np.percentile(eval_lens, 50)
    calib_max = calib_lens.max()
    calib_p90 = np.percentile(calib_lens, 90)
    
    print(f"  Eval P50 (median):   {eval_p50:,.0f} tokens")
    print(f"  Eval P90:            {eval_p90:,.0f} tokens")
    print(f"  Eval max:            {eval_lens.max():,} tokens")
    print(f"  Calib max:           {calib_max:,} tokens")
    print(f"  Calib P90:           {calib_p90:,.0f} tokens")
    
    coverage_ratio = calib_max / eval_p90 if eval_p90 > 0 else 0
    print(f"\n  Coverage ratio (calib_max / eval_P90): {coverage_ratio:.2f}x")
    
    if coverage_ratio < 0.5:
        print(f"  SEVERE MISMATCH: Calib data is way too short!")
        print(f"  The Hessian will never see long-range patterns.")
        print(f"  Recommendation: max_len >= {int(eval_p90)}")
    elif coverage_ratio < 1.0:
        print(f"  PARTIAL MISMATCH: Calib doesn't fully cover eval distribution.")
        print(f"  Recommendation: max_len >= {int(eval_p90)}")
    else:
        print(f"  OK: Calib covers eval distribution well.")
    
    # Content type analysis
    print(f"\n  Content type concern:")
    print(f"  - Eval: NIAH/QA/FWE/CWE are needle-in-haystack / long-doc tasks")
    print(f"  - If calib is WikiText+GSM8K, it has NO long-context structure")
    print(f"  - Ideal calib would use ACTUAL eval-like prompts (long docs)")
    print(f"  - At minimum, use the eval set itself (without answers) as calib")


if __name__ == "__main__":
    main()