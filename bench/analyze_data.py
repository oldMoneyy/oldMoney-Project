"""Analyze eval and calibration datasets for token statistics and repetition patterns."""
import json
import statistics
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/tmp/model_nvfp4_smoothed/')

def extract_text(line):
    if 'messages' in line:
        return ' '.join([m.get('content','') for m in line['messages'] if m.get('content')])
    elif 'prompt' in line:
        return line['prompt']
    elif 'text' in line:
        return line['text']
    elif 'input' in line:
        return line['input']
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

# Analyze both datasets
eval_stats = analyze_file('/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl', 'EVAL DATASET')
calib_stats = analyze_file('/opt/oldMoney-Project/quantization/calibration/optimal_64.jsonl', 'CALIB DATASET (optimal_64)')

# Comparison
print(f"\n{'='*80}")
print(f"  COMPARISON: EVAL vs CALIB")
print(f"{'='*80}")
print(f"  {'':>20} {'EVAL':>12} {'CALIB':>12}")
print(f"  {'Samples':>20} {eval_stats['n_samples']:>12} {calib_stats['n_samples']:>12}")
print(f"  {'Mean tokens':>20} {eval_stats['mean_tokens']:>12.0f} {calib_stats['mean_tokens']:>12.0f}")
print(f"  {'Max tokens':>20} {eval_stats['max_tokens']:>12} {calib_stats['max_tokens']:>12}")
print(f"  {'Mean uniqueness':>20} {statistics.mean(eval_stats['unique_ratios']):>12.4f} {statistics.mean(calib_stats['unique_ratios']):>12.4f}")

# Check if eval has samples in token ranges not covered by calib
eval_tc = eval_stats['token_counts']
calib_tc = calib_stats['token_counts']
calib_max = max(calib_tc)
calib_min = min(calib_tc)
eval_out_of_range = sum(1 for t in eval_tc if t > calib_max)
print(f"\n  Eval samples LONGER than calib max ({calib_max}): {eval_out_of_range}")
eval_shorter = sum(1 for t in eval_tc if t < calib_min)
print(f"  Eval samples SHORTER than calib min ({calib_min}): {eval_shorter}")
