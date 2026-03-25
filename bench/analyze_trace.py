#!/usr/bin/env python3
"""
Analyze SGLang profile trace for MiniCPM-SALA NVFP4 AWQ model.
Focuses on: computation bottlenecks, attention backends, quantization overhead.
python3 /opt/oldMoney-Project/bench/analyze_trace.py /tmp/1774407205.3815544-TP-0.trace.json.gz 2>&1 | tee /opt/oldMoney-Project/logs/trace_analysis.txt
"""
import json
import gzip
import sys
from collections import defaultdict

TRACE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/1774407205.3815544-TP-0.trace.json.gz"

print(f"Loading trace: {TRACE_PATH}")
if TRACE_PATH.endswith('.gz'):
    with gzip.open(TRACE_PATH, 'rt') as f:
        data = json.load(f)
else:
    with open(TRACE_PATH) as f:
        data = json.load(f)

events = data if isinstance(data, list) else data.get('traceEvents', data.get('events', []))
print(f"Total events: {len(events):,}")

# ═══════════════════════════════════════════════════════════════
# 1. TOP OPERATIONS BY TOTAL TIME
# ═══════════════════════════════════════════════════════════════

durations = defaultdict(lambda: {'count': 0, 'total_us': 0, 'max_us': 0, 'min_us': float('inf')})

for e in events:
    ph = e.get('ph', '')
    if ph == 'X' and 'dur' in e:
        name = e.get('name', 'unknown')
        dur = e['dur']  # microseconds
        d = durations[name]
        d['count'] += 1
        d['total_us'] += dur
        d['max_us'] = max(d['max_us'], dur)
        d['min_us'] = min(d['min_us'], dur)

total_time_us = sum(d['total_us'] for d in durations.values())
total_time_ms = total_time_us / 1000

print(f"\n{'='*100}")
print(f"TOP 30 OPERATIONS BY TOTAL GPU TIME (total traced: {total_time_ms:.0f} ms)")
print(f"{'='*100}")
print(f"{'Op':<60} {'Count':>6} {'Total ms':>10} {'%':>6} {'Avg ms':>10} {'Max ms':>10}")
print(f"{'-'*100}")

sorted_ops = sorted(durations.items(), key=lambda x: -x[1]['total_us'])
for name, d in sorted_ops[:30]:
    avg = d['total_us'] / d['count'] / 1000
    pct = d['total_us'] / total_time_us * 100 if total_time_us > 0 else 0
    print(f"{name[:60]:<60} {d['count']:>6} {d['total_us']/1000:>10.1f} {pct:>5.1f}% {avg:>10.3f} {d['max_us']/1000:>10.3f}")

# ═══════════════════════════════════════════════════════════════
# 2. CATEGORIZED BREAKDOWN
# ═══════════════════════════════════════════════════════════════

categories = {
    'GEMM/Linear': ['gemm', 'cutlass', 'matmul', 'cublas', 'fp4', 'fp8', 'quantize', 'dequant', 'scaled_mm', 'e2m1', 'nvfp4'],
    'Attention/FlashInfer': ['flash', 'attn', 'attention', 'flashinfer', 'gla', 'lightning', 'mamba', 'ssm', 'conv1d', 'selective_scan', 'simple_gla'],
    'Normalization': ['norm', 'rmsnorm', 'layernorm', 'layer_norm'],
    'Activation/Elementwise': ['silu', 'gelu', 'relu', 'sigmoid', 'mul', 'add', 'elementwise', 'fused'],
    'Communication': ['nccl', 'allreduce', 'broadcast', 'scatter', 'gather', 'all_to_all'],
    'Memory': ['memcpy', 'memset', 'copy_', 'to(', 'contiguous', 'reshape', 'view', 'transpose', 'cat', 'split', 'chunk'],
    'Embedding/Vocab': ['embed', 'vocab', 'lm_head', 'logit'],
    'RoPE': ['rope', 'rotary'],
    'Softmax/TopK': ['softmax', 'topk', 'argmax', 'sort'],
}

cat_stats = defaultdict(lambda: {'count': 0, 'total_us': 0, 'ops': defaultdict(int)})

for name, d in durations.items():
    name_lower = name.lower()
    matched = False
    for cat, keywords in categories.items():
        if any(kw in name_lower for kw in keywords):
            cat_stats[cat]['count'] += d['count']
            cat_stats[cat]['total_us'] += d['total_us']
            cat_stats[cat]['ops'][name] += d['count']
            matched = True
            break
    if not matched:
        cat_stats['Other']['count'] += d['count']
        cat_stats['Other']['total_us'] += d['total_us']
        cat_stats['Other']['ops'][name] += d['count']

print(f"\n{'='*100}")
print(f"CATEGORIZED BREAKDOWN")
print(f"{'='*100}")
print(f"{'Category':<30} {'Count':>8} {'Total ms':>12} {'%':>7} {'Top operations'}")
print(f"{'-'*100}")

for cat, stats in sorted(cat_stats.items(), key=lambda x: -x[1]['total_us']):
    pct = stats['total_us'] / total_time_us * 100 if total_time_us > 0 else 0
    top_ops = sorted(stats['ops'].items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(f"{n[:30]}({c})" for n, c in top_ops)
    print(f"{cat:<30} {stats['count']:>8} {stats['total_us']/1000:>12.1f} {pct:>6.1f}% {top_str[:60]}")

# ═══════════════════════════════════════════════════════════════
# 3. ATTENTION BACKEND ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"ATTENTION / LINEAR ATTENTION BACKEND ANALYSIS")
print(f"{'='*100}")

attn_ops = {}
for name, d in durations.items():
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['attn', 'flash', 'gla', 'lightning', 'mamba', 'ssm', 'conv1d', 'scan', 'recurrent', 'linear_attn', 'chunk']):
        attn_ops[name] = d

if attn_ops:
    print(f"\n{'Op':<70} {'Count':>6} {'Total ms':>10} {'Avg ms':>10}")
    print(f"{'-'*100}")
    for name, d in sorted(attn_ops.items(), key=lambda x: -x[1]['total_us']):
        avg = d['total_us'] / d['count'] / 1000
        print(f"{name[:70]:<70} {d['count']:>6} {d['total_us']/1000:>10.1f} {avg:>10.3f}")
else:
    print("  No attention-specific ops found in trace.")

# ═══════════════════════════════════════════════════════════════
# 4. NVFP4 / QUANTIZATION OPS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"QUANTIZATION / FP4 / FP8 OPERATIONS")
print(f"{'='*100}")

quant_ops = {}
for name, d in durations.items():
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['fp4', 'fp8', 'quant', 'dequant', 'scale', 'e2m1', 'modelopt', 'nvfp', 'int8', 'int4', 'w4a']):
        quant_ops[name] = d

if quant_ops:
    print(f"\n{'Op':<70} {'Count':>6} {'Total ms':>10} {'Avg ms':>10}")
    print(f"{'-'*100}")
    for name, d in sorted(quant_ops.items(), key=lambda x: -x[1]['total_us']):
        avg = d['total_us'] / d['count'] / 1000
        print(f"{name[:70]:<70} {d['count']:>6} {d['total_us']/1000:>10.1f} {avg:>10.3f}")
else:
    print("  No quantization-specific ops found. Checking for GEMM kernels...")

# ═══════════════════════════════════════════════════════════════
# 5. GEMM / LINEAR LAYER ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"GEMM / MATRIX MULTIPLY OPERATIONS")
print(f"{'='*100}")

gemm_ops = {}
for name, d in durations.items():
    name_lower = name.lower()
    if any(kw in name_lower for kw in ['gemm', 'cutlass', 'matmul', 'cublas', 'mm', 'linear', 'gemv', 'dot', 'scaled_mm', 'deep_gemm']):
        gemm_ops[name] = d

if gemm_ops:
    total_gemm_us = sum(d['total_us'] for d in gemm_ops.values())
    print(f"\nTotal GEMM time: {total_gemm_us/1000:.1f} ms ({total_gemm_us/total_time_us*100:.1f}% of traced time)")
    print(f"\n{'Op':<70} {'Count':>6} {'Total ms':>10} {'Avg ms':>10} {'Max ms':>10}")
    print(f"{'-'*100}")
    for name, d in sorted(gemm_ops.items(), key=lambda x: -x[1]['total_us']):
        avg = d['total_us'] / d['count'] / 1000
        print(f"{name[:70]:<70} {d['count']:>6} {d['total_us']/1000:>10.1f} {avg:>10.3f} {d['max_us']/1000:>10.3f}")
else:
    print("  No GEMM ops found by name.")

# ═══════════════════════════════════════════════════════════════
# 6. KERNEL LAUNCH / OVERHEAD ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"LONG-RUNNING OPS (potential bottlenecks, >5ms avg)")
print(f"{'='*100}")

print(f"\n{'Op':<70} {'Count':>6} {'Avg ms':>10} {'Max ms':>10} {'Total ms':>10}")
print(f"{'-'*100}")
for name, d in sorted_ops:
    avg_ms = d['total_us'] / d['count'] / 1000
    if avg_ms > 5.0:
        print(f"{name[:70]:<70} {d['count']:>6} {avg_ms:>10.2f} {d['max_us']/1000:>10.2f} {d['total_us']/1000:>10.1f}")

# ═══════════════════════════════════════════════════════════════
# 7. UNIQUE KERNEL NAMES (for discovery)
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"ALL UNIQUE KERNEL/OP NAMES ({len(durations)} unique)")
print(f"{'='*100}")
for name in sorted(durations.keys()):
    d = durations[name]
    if d['total_us'] > 100:  # Only show ops with >0.1ms total
        print(f"  {d['total_us']/1000:>10.1f} ms  {d['count']:>6}x  {name}")

print(f"\n{'='*100}")
print("ANALYSIS COMPLETE")
print(f"{'='*100}")