#!/usr/bin/env python3
"""
Analyze nvidia-smi dmon and memory CSV data for GPU utilization profiling.

Usage: python3 analyze_gpu.py <dmon.txt> <mem.csv>
"""
import sys

def analyze_dmon(filepath):
    """Analyze nvidia-smi dmon output (columns: gpu sm mem enc dec)"""
    print("=" * 80)
    print("GPU SM & MEMORY UTILIZATION (nvidia-smi dmon)")
    print("=" * 80)

    sm_vals = []
    mem_vals = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('gpu'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        sm = int(parts[1])
                        mem = int(parts[2])
                        if 0 <= sm <= 100 and 0 <= mem <= 100:
                            sm_vals.append(sm)
                            mem_vals.append(mem)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"  File not found: {filepath}")
        return

    if not sm_vals:
        print("  No valid samples found")
        return

    n = len(sm_vals)
    avg_sm = sum(sm_vals) / n
    avg_mem = sum(mem_vals) / n

    # Distribution
    sm_100 = sum(1 for v in sm_vals if v >= 95)
    sm_50_95 = sum(1 for v in sm_vals if 50 <= v < 95)
    sm_1_50 = sum(1 for v in sm_vals if 0 < v < 50)
    sm_0 = sum(1 for v in sm_vals if v == 0)

    print(f"\n  Samples:       {n}")
    print(f"  Avg SM%:       {avg_sm:.1f}%")
    print(f"  Avg Mem%:      {avg_mem:.1f}%")
    print(f"  Max SM%:       {max(sm_vals)}%")
    print(f"  Min SM%:       {min(sm_vals)}%")
    print(f"\n  SM Distribution:")
    print(f"    >= 95%:      {sm_100:>4} samples ({sm_100*100/n:.0f}%) -- GPU fully busy")
    print(f"    50-95%:      {sm_50_95:>4} samples ({sm_50_95*100/n:.0f}%) -- partial utilization")
    print(f"    1-50%:       {sm_1_50:>4} samples ({sm_1_50*100/n:.0f}%) -- low utilization")
    print(f"    0%:          {sm_0:>4} samples ({sm_0*100/n:.0f}%) -- idle")

    # Time series summary (first/middle/last thirds)
    third = n // 3
    if third > 0:
        print(f"\n  Time Phases:")
        for label, start, end in [("Early", 0, third), ("Mid", third, 2*third), ("Late", 2*third, n)]:
            phase_sm = sm_vals[start:end]
            phase_mem = mem_vals[start:end]
            print(f"    {label:>5}:  SM={sum(phase_sm)/len(phase_sm):.0f}%  Mem={sum(phase_mem)/len(phase_mem):.0f}%")


def analyze_mem(filepath):
    """Analyze nvidia-smi memory CSV"""
    print(f"\n{'='*80}")
    print("GPU MEMORY USAGE OVER TIME")
    print("=" * 80)

    mem_used_vals = []
    try:
        with open(filepath) as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        mem_str = parts[1].strip().replace(' MiB', '').replace(' MB', '')
                        mem_used = int(mem_str)
                        mem_used_vals.append(mem_used)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"  File not found: {filepath}")
        return

    if not mem_used_vals:
        print("  No valid memory samples found")
        return

    print(f"  Samples:     {len(mem_used_vals)}")
    print(f"  Peak:        {max(mem_used_vals)} MiB")
    print(f"  Min:         {min(mem_used_vals)} MiB")
    print(f"  Avg:         {sum(mem_used_vals)/len(mem_used_vals):.0f} MiB")
    print(f"  Range:       {max(mem_used_vals) - min(mem_used_vals)} MiB")


if __name__ == "__main__":
    dmon_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gpu_dmon.txt"
    mem_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gpu_mem.csv"
    analyze_dmon(dmon_file)
    analyze_mem(mem_file)
