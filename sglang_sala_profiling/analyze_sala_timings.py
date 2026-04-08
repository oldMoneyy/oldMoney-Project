#!/usr/bin/env python3
"""
Analyze sala_timings.csv from SALA profiling runs.

Usage:
    python analyze_sala_timings.py <path_to_sala_timings.csv> [--nsys <path_to_nsys.txt>]

What this script does:
    1. Parses the sala_timings.csv (CUDA event timings, no sync overhead in hot path)
    2. Decomposes to LEAF-LEVEL components with ZERO double-counting
       - attn wraps {qkv_proj, attn_kernel, o_gate, o_proj} -> only count leaves
       - fla_kernel wraps {fla_state_load, fused_recurrent_gla, chunk_simple_gla, fla_state_save}
       - Overhead = wrapper - sum(children), tracked separately
    3. Reports top bottlenecks for decode, prefill, and combined
    4. Optionally cross-references with nsys kernel summary

Profiling methodology (for context):
    - Timing instrument: torch.cuda.Event(enable_timing=True) pairs
    - record() is non-blocking; no torch.cuda.synchronize() in hot path
    - Sync only happens in print_summary() every N steps (outside forward pass)
    - CUDA graphs must be disabled (--disable-cuda-graph) for measurements
    - Values are GPU-side elapsed ms, NOT CPU wall-clock
    - Includes inter-kernel gaps (CPU dispatch overhead) within each region
    - NSYS measures kernel-only time; SALA measures region time (always >= nsys)
"""

import csv
import sys
import argparse
import re
from collections import defaultdict


def parse_csv(path):
    """Parse sala_timings.csv into a dict keyed by the 'key' column."""
    by_key = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            by_key[r['key']] = {
                'total_ms': float(r['total_ms']),
                'count': int(r['count']),
                'avg_ms': float(r['avg_ms']),
                'min_ms': float(r['min_ms']),
                'max_ms': float(r['max_ms']),
            }
    return by_key


def get(by_key, key):
    """Get total_ms for a key, defaulting to 0 if missing."""
    return by_key.get(key, {'total_ms': 0})['total_ms']


def get_count(by_key, key):
    """Get count for a key, defaulting to 0 if missing."""
    return by_key.get(key, {'count': 0})['count']


def decompose_to_leaves(by_key):
    """
    Decompose nested profiling regions into leaf-level totals.
    Returns dict: leaf[(name, mode)] = total_ms

    Nesting structure:
      MiniCPM4 layers:
        L{id}/minicpm4/attn/{mode}          PARENT
          L{id}/minicpm4/qkv_proj/{mode}      child
          L{id}/minicpm4/attn_kernel/{mode}   child (the actual FlashInfer kernel)
          L{id}/minicpm4/o_gate/{mode}        child
          L{id}/minicpm4/o_proj/{mode}        child
        L{id}/minicpm4/input_norm/{mode}    standalone
        L{id}/minicpm4/post_attn_norm/{mode} standalone
        L{id}/minicpm4/mlp/{mode}           standalone

      Lightning layers:
        L{id}/lightning-attn/attn/{mode}    PARENT (wraps ALL lightning/* children)
          L{id}/lightning/qkv_proj/{mode}     child
          L{id}/lightning/qk_norm/{mode}      child
          L{id}/lightning/rope/{mode}         child
          L{id}/lightning/fla_kernel/{mode}   child (PARENT)
            L{id}/lightning/fla_state_load/{mode}    grandchild
            L{id}/lightning/fused_recurrent_gla/{mode} grandchild
            L{id}/lightning/chunk_simple_gla/{mode}    grandchild (prefill only)
            L{id}/lightning/fla_state_save/{mode}    grandchild
          L{id}/lightning/o_norm/{mode}       child
          L{id}/lightning/sigmoid_gate/{mode} child
          L{id}/lightning/o_proj/{mode}       child
        L{id}/lightning-attn/input_norm/{mode}    standalone
        L{id}/lightning-attn/post_attn_norm/{mode} standalone
        L{id}/lightning-attn/mlp/{mode}           standalone

      Model-level:
        model/embed/{mode}
        model/final_norm/{mode}
        model/logits/{mode}
    """
    # Auto-detect layer types from keys
    minicpm4_layers = set()
    lightning_layers = set()
    for key in by_key:
        m = re.match(r'L(\d+)/minicpm4/', key)
        if m:
            minicpm4_layers.add(int(m.group(1)))
        m = re.match(r'L(\d+)/lightning', key)
        if m:
            lightning_layers.add(int(m.group(1)))

    leaf = defaultdict(float)

    def add(name, mode, val):
        leaf[(name, mode)] += val

    # MiniCPM4 layers
    for lid in sorted(minicpm4_layers):
        p = f'L{lid:02d}/minicpm4'
        for mode in ['decode', 'prefill']:
            attn = get(by_key, f'{p}/attn/{mode}')
            ak = get(by_key, f'{p}/attn_kernel/{mode}')
            qkv = get(by_key, f'{p}/qkv_proj/{mode}')
            og = get(by_key, f'{p}/o_gate/{mode}')
            op = get(by_key, f'{p}/o_proj/{mode}')
            overhead = attn - ak - qkv - og - op

            add('minicpm4/attn_kernel', mode, ak)
            add('minicpm4/qkv_proj', mode, qkv)
            add('minicpm4/o_gate', mode, og)
            add('minicpm4/o_proj', mode, op)
            add('minicpm4/attn_overhead', mode, overhead)
            add('minicpm4/mlp', mode, get(by_key, f'{p}/mlp/{mode}'))
            add('minicpm4/input_norm', mode, get(by_key, f'{p}/input_norm/{mode}'))
            add('minicpm4/post_attn_norm', mode, get(by_key, f'{p}/post_attn_norm/{mode}'))

    # Lightning layers
    for lid in sorted(lightning_layers):
        la = f'L{lid:02d}/lightning-attn'
        l = f'L{lid:02d}/lightning'
        for mode in ['decode', 'prefill']:
            # Standalone top-level components
            add('lightning/input_norm', mode, get(by_key, f'{la}/input_norm/{mode}'))
            add('lightning/post_attn_norm', mode, get(by_key, f'{la}/post_attn_norm/{mode}'))
            add('lightning/mlp', mode, get(by_key, f'{la}/mlp/{mode}'))

            # Attn wrapper and its children
            attn_w = get(by_key, f'{la}/attn/{mode}')
            qkv = get(by_key, f'{l}/qkv_proj/{mode}')
            qk_n = get(by_key, f'{l}/qk_norm/{mode}')
            rope = get(by_key, f'{l}/rope/{mode}')
            sig = get(by_key, f'{l}/sigmoid_gate/{mode}')
            op = get(by_key, f'{l}/o_proj/{mode}')
            on = get(by_key, f'{l}/o_norm/{mode}')

            # fla_kernel and its children
            fla_k = get(by_key, f'{l}/fla_kernel/{mode}')
            st_l = get(by_key, f'{l}/fla_state_load/{mode}')
            st_s = get(by_key, f'{l}/fla_state_save/{mode}')
            rec = get(by_key, f'{l}/fused_recurrent_gla/{mode}')
            chk = get(by_key, f'{l}/chunk_simple_gla/{mode}')
            fla_oh = fla_k - st_l - st_s - rec - chk

            attn_inner = qkv + qk_n + rope + sig + op + on + fla_k
            attn_oh = attn_w - attn_inner

            add('lightning/fused_recurrent_gla', mode, rec)
            add('lightning/chunk_simple_gla', mode, chk)
            add('lightning/fla_state_load', mode, st_l)
            add('lightning/fla_state_save', mode, st_s)
            add('lightning/fla_kernel_overhead', mode, fla_oh)
            add('lightning/qkv_proj', mode, qkv)
            add('lightning/qk_norm', mode, qk_n)
            add('lightning/rope', mode, rope)
            add('lightning/sigmoid_gate', mode, sig)
            add('lightning/o_proj', mode, op)
            add('lightning/o_norm', mode, on)
            add('lightning/attn_overhead', mode, attn_oh)

    # Model-level components
    for mode in ['decode', 'prefill']:
        for comp in ['embed', 'final_norm', 'logits']:
            add(f'model/{comp}', mode, get(by_key, f'model/{comp}/{mode}'))

    return dict(leaf)


def print_top_n(leaf, mode, n=10):
    """Print top N bottlenecks for a given mode."""
    items = sorted([(name, val) for (name, m), val in leaf.items() if m == mode],
                   key=lambda x: -x[1])
    total = sum(v for _, v in items)

    # Detect step count from the data
    # Use any high-count key to determine decode steps
    if mode == 'decode':
        desc = f'DECODE (leaf-level, zero double-counting, {total / 1000:.1f}s total)'
    else:
        desc = f'PREFILL (leaf-level, zero double-counting, {total / 1000:.1f}s total)'

    print(f'\n{"=" * 90}')
    print(f'TOP {n} BOTTLENECKS: {desc}')
    print(f'{"=" * 90}')
    for i, (name, val) in enumerate(items[:n]):
        pct = val / total * 100
        print(f'  #{i + 1:2d}  {name:<40s}  {val:>10.1f}ms  ({pct:>5.1f}%)')
    print(f'       {"SUM of all":<40s}  {total:>10.1f}ms')


def print_combined(leaf, n=10):
    """Print top N combined bottlenecks."""
    combined = defaultdict(float)
    for (name, mode), val in leaf.items():
        combined[name] += val
    total = sum(combined.values())
    items = sorted(combined.items(), key=lambda x: -x[1])

    print(f'\n{"=" * 90}')
    print(f'TOP {n} BOTTLENECKS: COMBINED ({total / 1000:.1f}s total)')
    print(f'{"=" * 90}')
    for i, (name, val) in enumerate(items[:n]):
        d = leaf.get((name, 'decode'), 0)
        p = leaf.get((name, 'prefill'), 0)
        pct = val / total * 100
        print(f'  #{i + 1:2d}  {name:<40s}  {val:>10.1f}ms  ({pct:>5.1f}%)  [D:{d:>9.1f}  P:{p:>9.1f}]')
    print(f'       {"SUM of all":<40s}  {total:>10.1f}ms')


def print_grouped(leaf):
    """Print grouped view merging related components."""
    combined = defaultdict(float)
    for (name, mode), val in leaf.items():
        combined[name] += val
    total = sum(combined.values())

    groups = [
        ('ALL MLP (32 layers)',              ['lightning/mlp', 'minicpm4/mlp']),
        ('minicpm4 dense attn_kernel (8L)',  ['minicpm4/attn_kernel']),
        ('ALL qkv_proj (32L)',              ['lightning/qkv_proj', 'minicpm4/qkv_proj']),
        ('lightning recurrent_gla (24L)',    ['lightning/fused_recurrent_gla']),
        ('ALL overhead gaps',               ['lightning/attn_overhead', 'lightning/fla_kernel_overhead',
                                             'minicpm4/attn_overhead']),
        ('lightning state I/O (24L)',        ['lightning/fla_state_load', 'lightning/fla_state_save']),
        ('ALL norms (input+post+o)',         ['lightning/input_norm', 'lightning/post_attn_norm',
                                             'lightning/o_norm', 'minicpm4/input_norm',
                                             'minicpm4/post_attn_norm']),
        ('lightning sigmoid_gate (24L)',     ['lightning/sigmoid_gate']),
        ('ALL o_proj (32L)',                ['lightning/o_proj', 'minicpm4/o_proj']),
        ('lightning rope (24L)',            ['lightning/rope']),
        ('lightning qk_norm (24L)',         ['lightning/qk_norm']),
        ('model (embed+logits+norm)',       ['model/embed', 'model/logits', 'model/final_norm']),
        ('minicpm4 o_gate (8L)',            ['minicpm4/o_gate']),
        ('lightning chunk_gla (prefill)',   ['lightning/chunk_simple_gla']),
    ]

    print(f'\n{"=" * 90}')
    print(f'GROUPED VIEW (related components merged, {total / 1000:.1f}s total)')
    print(f'{"=" * 90}')

    for gname, keys in sorted(groups, key=lambda g: -sum(combined.get(k, 0) for k in g[1])):
        t = sum(combined.get(k, 0) for k in keys)
        d = sum(leaf.get((k, 'decode'), 0) for k in keys)
        p = sum(leaf.get((k, 'prefill'), 0) for k in keys)
        pct = t / total * 100
        print(f'  {gname:<42s}  {t:>10.1f}ms ({pct:>5.1f}%)  [D:{d:>9.1f}  P:{p:>9.1f}]')
    print(f'  {"TOTAL":<42s}  {total:>10.1f}ms')


def print_per_layer(by_key):
    """Print per-layer decode totals."""
    layer_totals = defaultdict(lambda: {'decode': 0, 'prefill': 0, 'type': ''})
    for key, data in by_key.items():
        m = re.match(r'L(\d+)/(minicpm4|lightning-attn|lightning)/', key)
        if not m:
            continue
        lid = int(m.group(1))
        ltype = m.group(2)
        if key.endswith('/decode'):
            layer_totals[lid]['decode'] += data['total_ms']
        elif key.endswith('/prefill'):
            layer_totals[lid]['prefill'] += data['total_ms']
        if ltype in ('minicpm4', 'lightning-attn'):
            layer_totals[lid]['type'] = ltype.replace('lightning-attn', 'lightning')

    total_d = sum(v['decode'] for v in layer_totals.values())
    total_p = sum(v['prefill'] for v in layer_totals.values())

    print(f'\n{"=" * 90}')
    print(f'PER-LAYER TOTALS (decode + prefill)')
    print(f'{"=" * 90}')
    for lid in sorted(layer_totals.keys()):
        v = layer_totals[lid]
        d_pct = v['decode'] / total_d * 100 if total_d else 0
        p_pct = v['prefill'] / total_p * 100 if total_p else 0
        print(f'  L{lid:02d} ({v["type"]:>10s}):  D={v["decode"]:>10.1f}ms ({d_pct:>4.1f}%)  '
              f'P={v["prefill"]:>10.1f}ms ({p_pct:>4.1f}%)')


def print_mode_split(leaf):
    """Print decode vs prefill split."""
    decode_total = sum(v for (_, m), v in leaf.items() if m == 'decode')
    prefill_total = sum(v for (_, m), v in leaf.items() if m == 'prefill')
    grand = decode_total + prefill_total

    print(f'\n{"=" * 90}')
    print(f'MODE SPLIT')
    print(f'{"=" * 90}')
    print(f'  Decode:  {decode_total:>12.1f}ms  ({decode_total / grand * 100:.1f}%)  = {decode_total / 1000:.1f}s')
    print(f'  Prefill: {prefill_total:>12.1f}ms  ({prefill_total / grand * 100:.1f}%)  = {prefill_total / 1000:.1f}s')
    print(f'  Total:   {grand:>12.1f}ms              = {grand / 1000:.1f}s')


def print_sanity_check(by_key, leaf):
    """Verify leaf decomposition sums match raw totals (catch bugs)."""
    raw_decode = sum(d['total_ms'] for k, d in by_key.items() if k.endswith('/decode'))
    raw_prefill = sum(d['total_ms'] for k, d in by_key.items() if k.endswith('/prefill'))
    leaf_decode = sum(v for (_, m), v in leaf.items() if m == 'decode')
    leaf_prefill = sum(v for (_, m), v in leaf.items() if m == 'prefill')

    # Raw totals include parent wrappers (double-counted).
    # Leaf totals should be LESS than raw totals (parents removed, only overhead kept).
    print(f'\n{"=" * 90}')
    print(f'SANITY CHECK')
    print(f'{"=" * 90}')
    print(f'  Raw CSV decode (with nesting):  {raw_decode:>12.1f}ms')
    print(f'  Leaf decode (no overlap):       {leaf_decode:>12.1f}ms')
    print(f'  Removed (parent wrappers):      {raw_decode - leaf_decode:>12.1f}ms')
    print(f'  Raw CSV prefill:                {raw_prefill:>12.1f}ms')
    print(f'  Leaf prefill:                   {leaf_prefill:>12.1f}ms')
    print(f'  Removed:                        {raw_prefill - leaf_prefill:>12.1f}ms')


def parse_nsys(path):
    """Parse nsys kernel summary from nsys.txt (the stats output section)."""
    kernels = []
    in_table = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if 'Time (%)' in line and 'Total Time' in line:
                in_table = True
                continue
            if in_table and line.startswith('--------'):
                continue
            if in_table and line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        time_pct = float(parts[0])
                        total_ms = float(parts[1])
                        instances = int(parts[2])
                        avg_ms = float(parts[3])
                        # Name is everything after the 7th column
                        name = ' '.join(parts[7:])
                        kernels.append({
                            'time_pct': time_pct,
                            'total_ms': total_ms,
                            'instances': instances,
                            'avg_ms': avg_ms,
                            'name': name,
                        })
                    except (ValueError, IndexError):
                        pass
    return kernels


def print_nsys_comparison(leaf, nsys_kernels):
    """Cross-reference nsys kernel data with SALA leaf data."""
    nsys_total = sum(k['total_ms'] for k in nsys_kernels)
    sala_total = sum(v for v in leaf.values())

    # Classify nsys kernels
    nsys_groups = defaultdict(float)
    for k in nsys_kernels:
        name = k['name']
        if 'marlin::Marlin' in name or 'gptq_marlin' in name:
            nsys_groups['Marlin GEMM (linear layers)'] += k['total_ms']
        elif 'BatchPrefill' in name or 'MergeState' in name or 'PersistentVariable' in name:
            nsys_groups['FlashInfer attention'] += k['total_ms']
        elif 'chunk_fwd_kernel' in name:
            nsys_groups['GLA chunk kernel'] += k['total_ms']
        elif 'fused_recurrent_fwd' in name:
            nsys_groups['GLA fused recurrent'] += k['total_ms']
        elif 'RMSNorm' in name:
            nsys_groups['RMSNorm'] += k['total_ms']
        elif 'act_and_mul' in name:
            nsys_groups['SiLU activation'] += k['total_ms']
        elif 'RotaryPos' in name:
            nsys_groups['RoPE'] += k['total_ms']
        elif 'cutlass' in name.lower():
            nsys_groups['Cutlass GEMM'] += k['total_ms']
        else:
            nsys_groups['Other (copy/index/elementwise)'] += k['total_ms']

    # SALA equivalent groups
    sala_groups = {
        'Marlin GEMM (linear layers)': sum(leaf.get((k, m), 0)
            for k in ['lightning/mlp', 'minicpm4/mlp', 'lightning/qkv_proj',
                       'minicpm4/qkv_proj', 'lightning/o_proj', 'minicpm4/o_proj',
                       'minicpm4/o_gate', 'lightning/sigmoid_gate']
            for m in ['decode', 'prefill']),
        'FlashInfer attention': sum(leaf.get(('minicpm4/attn_kernel', m), 0)
            for m in ['decode', 'prefill']),
        'GLA chunk kernel': sum(leaf.get(('lightning/chunk_simple_gla', m), 0)
            for m in ['decode', 'prefill']),
        'GLA fused recurrent': sum(leaf.get(('lightning/fused_recurrent_gla', m), 0)
            for m in ['decode', 'prefill']),
        'RMSNorm': sum(leaf.get((k, m), 0)
            for k in ['lightning/input_norm', 'lightning/post_attn_norm', 'lightning/o_norm',
                       'lightning/qk_norm', 'minicpm4/input_norm', 'minicpm4/post_attn_norm']
            for m in ['decode', 'prefill']),
    }

    print(f'\n{"=" * 90}')
    print(f'NSYS vs SALA CROSS-REFERENCE')
    print(f'{"=" * 90}')
    print(f'  NSYS: kernel-only execution, {nsys_total / 1000:.1f}s total')
    print(f'  SALA: region time (incl. dispatch gaps), {sala_total / 1000:.1f}s total')
    print(f'  Gap: {(sala_total - nsys_total) / 1000:.1f}s ({(sala_total - nsys_total) / sala_total * 100:.1f}% = CPU dispatch overhead)')
    print()
    print(f'  {"Group":<35s}  {"NSYS":>10s} {"NSYS%":>6s}  {"SALA~":>10s} {"SALA%":>6s}  {"Ratio":>6s}')
    print(f'  {"-" * 35}  {"-" * 10} {"-" * 6}  {"-" * 10} {"-" * 6}  {"-" * 6}')

    for gname in sorted(nsys_groups.keys(), key=lambda g: -nsys_groups[g]):
        n = nsys_groups[gname]
        s = sala_groups.get(gname, 0)
        n_pct = n / nsys_total * 100
        s_pct = s / sala_total * 100 if s else 0
        ratio = s / n if n > 0 else 0
        s_str = f'{s:>9.0f}ms' if s else '       N/A'
        s_pct_str = f'({s_pct:>4.1f}%)' if s else '  (N/A)'
        r_str = f'{ratio:>5.2f}x' if s else '   N/A'
        print(f'  {gname:<35s}  {n:>9.0f}ms ({n_pct:>4.1f}%)  {s_str} {s_pct_str}  {r_str}')

    print()
    print('  Ratio interpretation: SALA/NSYS ratio > 1.0 means SALA includes')
    print('  inter-kernel dispatch overhead. Higher ratio = more small kernel launches.')
    print('  With CUDA graphs in production, this overhead largely disappears.')


def main():
    parser = argparse.ArgumentParser(description='Analyze SALA profiling timings CSV')
    parser.add_argument('csv_path', help='Path to sala_timings.csv')
    parser.add_argument('--nsys', help='Path to nsys.txt for cross-reference')
    parser.add_argument('--top', type=int, default=10, help='Number of top bottlenecks to show')
    parser.add_argument('--per-layer', action='store_true', help='Show per-layer breakdown')
    args = parser.parse_args()

    by_key = parse_csv(args.csv_path)
    leaf = decompose_to_leaves(by_key)

    print_sanity_check(by_key, leaf)
    print_mode_split(leaf)
    print_top_n(leaf, 'decode', args.top)
    print_top_n(leaf, 'prefill', args.top)
    print_combined(leaf, args.top)
    print_grouped(leaf)

    if args.per_layer:
        print_per_layer(by_key)

    if args.nsys:
        nsys_kernels = parse_nsys(args.nsys)
        if nsys_kernels:
            print_nsys_comparison(leaf, nsys_kernels)
        else:
            print(f'\nWARNING: Could not parse nsys kernel data from {args.nsys}')


if __name__ == '__main__':
    main()
