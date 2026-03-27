"""Analyze eval and calibration datasets for token statistics and repetition patterns.

Compares: eval (perf_public_set) vs calib (optimal_64) vs calib (optimal_96)
"""
import json
import statistics
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/opt/model/')

def extract_text(line):
    if 'messages' in line:
        return ' '.join([m.get('content','') for m in line['messages'] if m.get('content')])
    elif 'prompt' in line:
        return line['prompt']
    elif 'text' in line:
        return line['text']
    elif 'input' in line:
        return line['input']
    elif 'question' in line:
        return line['question']
    return json.dumps(line)

def analyze_file(path, label):
    print(f"\n{'='*80}")
    print(f"  {label}: {path}")
    print(f"{'='*80}")

    with open(path) as f:
        lines = [json.loads(l) for l in f]

    print(f"Total samples: {len(lines)}")
    print(f"Keys: {list(lines[0].keys())}")
    print(f"Sample entry (first 500 chars): {json.dumps(lines[0], ensure_ascii=False)[:500]}")

    token_counts = []
    unique_ratios = []
    char_counts = []

    for line in lines:
        text = extract_text(line)
        tokens = tok.encode(text)
        token_counts.append(len(tokens))
        char_counts.append(len(text))
        unique = len(set(tokens))
        unique_ratios.append(unique / len(tokens) if tokens else 0)

    print(f"\n--- Token count statistics ---")
    print(f"  Min:    {min(token_counts)}")
    print(f"  Max:    {max(token_counts)}")
    print(f"  Mean:   {statistics.mean(token_counts):.0f}")
    print(f"  Median: {statistics.median(token_counts):.0f}")
    if len(token_counts) > 1:
        print(f"  Stdev:  {statistics.stdev(token_counts):.0f}")
    print(f"  Total:  {sum(token_counts)}")

    print(f"\n--- Character count statistics ---")
    print(f"  Min:    {min(char_counts)}")
    print(f"  Max:    {max(char_counts)}")
    print(f"  Mean:   {statistics.mean(char_counts):.0f}")

    # Distribution buckets
    print(f"\n--- Token count distribution ---")
    buckets = [0, 1000, 5000, 10000, 20000, 50000, 100000, 131072, 200000]
    for i in range(len(buckets)-1):
        count = sum(1 for t in token_counts if buckets[i] <= t < buckets[i+1])
        if count > 0:
            print(f"  [{buckets[i]:>7}-{buckets[i+1]:>7}): {count}")
    count = sum(1 for t in token_counts if t >= buckets[-1])
    if count > 0:
        print(f"  [{buckets[-1]:>7}+): {count}")

    # Repetition analysis
    print(f"\n--- Token uniqueness ratio (unique/total, lower = more repetitive) ---")
    print(f"  Min ratio:  {min(unique_ratios):.4f}")
    print(f"  Max ratio:  {max(unique_ratios):.4f}")
    print(f"  Mean ratio: {statistics.mean(unique_ratios):.4f}")

    # Find most repetitive samples
    indexed = sorted(enumerate(unique_ratios), key=lambda x: x[0])
    print(f"\n--- Per-sample details (idx | tokens | unique | ratio | first 80 chars) ---")
    for idx, ratio in indexed:
        text = extract_text(lines[idx])
        preview = text[:80].replace('\n', ' ')
        print(f"  [{idx:>3}] {token_counts[idx]:>7} tok | {int(unique_ratios[idx]*token_counts[idx]):>6} uniq | ratio={unique_ratios[idx]:.4f} | {preview}")

    # Find the MOST repetitive samples
    most_rep = sorted(enumerate(unique_ratios), key=lambda x: x[1])[:5]
    print(f"\n--- Top 5 most repetitive samples ---")
    for idx, ratio in most_rep:
        text = extract_text(lines[idx])
        preview = text[:120].replace('\n', ' ')
        print(f"  [{idx:>3}] {token_counts[idx]:>7} tok | ratio={ratio:.4f} | {preview}")

    return {
        'n_samples': len(lines),
        'token_counts': token_counts,
        'unique_ratios': unique_ratios,
        'mean_tokens': statistics.mean(token_counts),
        'max_tokens': max(token_counts),
    }

# Analyze all three datasets
eval_stats = analyze_file('/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl', 'EVAL DATASET')
calib64_stats = analyze_file('/opt/oldMoney-Project/quantization/calibration/optimal_64.jsonl', 'CALIB DATASET (optimal_64)')
calib96_stats = analyze_file('/opt/oldMoney-Project/quantization/calibration/optimal_96.jsonl', 'CALIB DATASET (optimal_96)')

# Comparison
print(f"\n{'='*80}")
print(f"  COMPARISON: EVAL vs CALIB_64 vs CALIB_96")
print(f"{'='*80}")
print(f"  {'':>20} {'EVAL':>12} {'CALIB_64':>12} {'CALIB_96':>12}")
print(f"  {'Samples':>20} {eval_stats['n_samples']:>12} {calib64_stats['n_samples']:>12} {calib96_stats['n_samples']:>12}")
print(f"  {'Mean tokens':>20} {eval_stats['mean_tokens']:>12.0f} {calib64_stats['mean_tokens']:>12.0f} {calib96_stats['mean_tokens']:>12.0f}")
print(f"  {'Max tokens':>20} {eval_stats['max_tokens']:>12} {calib64_stats['max_tokens']:>12} {calib96_stats['max_tokens']:>12}")
print(f"  {'Mean uniqueness':>20} {statistics.mean(eval_stats['unique_ratios']):>12.4f} {statistics.mean(calib64_stats['unique_ratios']):>12.4f} {statistics.mean(calib96_stats['unique_ratios']):>12.4f}")
print(f"  {'Min uniqueness':>20} {min(eval_stats['unique_ratios']):>12.4f} {min(calib64_stats['unique_ratios']):>12.4f} {min(calib96_stats['unique_ratios']):>12.4f}")

# Coverage analysis: eval vs calib_96
eval_tc = eval_stats['token_counts']
c96_tc = calib96_stats['token_counts']
c96_max = max(c96_tc)
c96_min = min(c96_tc)
print(f"\n  --- Coverage: EVAL vs CALIB_96 ---")
print(f"  Eval samples LONGER than calib_96 max ({c96_max}): {sum(1 for t in eval_tc if t > c96_max)}")
print(f"  Eval samples SHORTER than calib_96 min ({c96_min}): {sum(1 for t in eval_tc if t < c96_min)}")

# Coverage analysis: eval vs calib_64
c64_tc = calib64_stats['token_counts']
c64_max = max(c64_tc)
c64_min = min(c64_tc)
print(f"\n  --- Coverage: EVAL vs CALIB_64 ---")
print(f"  Eval samples LONGER than calib_64 max ({c64_max}): {sum(1 for t in eval_tc if t > c64_max)}")
print(f"  Eval samples SHORTER than calib_64 min ({c64_min}): {sum(1 for t in eval_tc if t < c64_min)}")

# What's new in calib_96 vs calib_64?
print(f"\n  --- CALIB_96 vs CALIB_64 differences ---")
print(f"  Extra samples: {calib96_stats['n_samples'] - calib64_stats['n_samples']}")
print(f"  Token range 64: [{min(c64_tc)}, {max(c64_tc)}]")
print(f"  Token range 96: [{min(c96_tc)}, {max(c96_tc)}]")

# Distribution comparison
print(f"\n  --- Token bucket distribution comparison ---")
buckets = [0, 1000, 5000, 10000, 20000, 50000, 100000, 131072, 200000]
print(f"  {'Bucket':>20} {'EVAL':>8} {'CAL64':>8} {'CAL96':>8}")
for i in range(len(buckets)-1):
    e = sum(1 for t in eval_tc if buckets[i] <= t < buckets[i+1])
    c64 = sum(1 for t in c64_tc if buckets[i] <= t < buckets[i+1])
    c96 = sum(1 for t in c96_tc if buckets[i] <= t < buckets[i+1])
    if e > 0 or c64 > 0 or c96 > 0:
        print(f"  [{buckets[i]:>7}-{buckets[i+1]:>7}) {e:>8} {c64:>8} {c96:>8}")
