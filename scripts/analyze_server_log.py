#!/usr/bin/env python3
"""Analyze SGLang server logs to understand throughput vs batch size patterns.

Usage:
    python scripts/analyze_server_log.py <server.log>

Example:
    python scripts/analyze_server_log.py /opt/server.log
"""

import re
import sys
from collections import defaultdict

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_server_log.py <server.log>")
        sys.exit(1)

    pattern = re.compile(
        r'\[([^\]]+)\] Decode batch, '
        r'#running-req: (\d+), '
        r'#full token: (\d+), '
        r'full token usage: ([0-9.]+), '
        r'mamba num: (\d+), '
        r'mamba usage: ([0-9.]+), '
        r'cuda graph: (True|False), '
        r'gen throughput \(token/s\): ([0-9.]+), '
        r'#queue-req: (\d+)'
    )

    # Collect data per batch size
    batch_data = defaultdict(lambda: {"count": 0, "throughput_sum": 0, "token_sum": 0, "queue_sum": 0})
    all_entries = []

    with open(sys.argv[1], "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            ts, running, full_tok, usage, mamba, mamba_usage, cuda_graph, tps, queue = m.groups()
            entry = {
                "ts": ts,
                "running": int(running),
                "full_token": int(full_tok),
                "usage": float(usage),
                "tps": float(tps),
                "queue": int(queue),
            }
            all_entries.append(entry)

            bs = int(running)
            batch_data[bs]["count"] += 1
            batch_data[bs]["throughput_sum"] += float(tps)
            batch_data[bs]["token_sum"] += int(full_tok)
            batch_data[bs]["queue_sum"] += int(queue)

    if not all_entries:
        print("No decode batch entries found in log.")
        sys.exit(1)

    total_steps = len(all_entries)
    print(f"Total decode steps: {total_steps}")
    print()

    # 1. Throughput vs batch size
    print("=" * 70)
    print(f"{'BatchSize':>9} {'Steps':>7} {'%Time':>7} {'AvgTPS':>9} {'TPS/Req':>9} {'AvgKVTok':>10}")
    print("=" * 70)

    for bs in sorted(batch_data.keys()):
        d = batch_data[bs]
        n = d["count"]
        avg_tps = d["throughput_sum"] / n
        avg_tok = d["token_sum"] / n
        pct = n / total_steps * 100
        per_req = avg_tps / bs if bs > 0 else 0
        print(f"{bs:>9} {n:>7} {pct:>6.1f}% {avg_tps:>9.1f} {per_req:>9.1f} {avg_tok:>10.0f}")

    print()

    # 2. Time wasted on low batch sizes (tail)
    low_batch_steps = sum(d["count"] for bs, d in batch_data.items() if bs <= 8)
    low_batch_pct = low_batch_steps / total_steps * 100
    high_batch_steps = sum(d["count"] for bs, d in batch_data.items() if bs > 32)
    high_batch_pct = high_batch_steps / total_steps * 100

    print("=" * 70)
    print("Time distribution:")
    print(f"  Batch ≤ 8 (tail/straggler):  {low_batch_steps:>6} steps = {low_batch_pct:>5.1f}%")
    print(f"  Batch 9-32 (medium):         {total_steps - low_batch_steps - high_batch_steps:>6} steps")
    print(f"  Batch > 32 (full load):      {high_batch_steps:>6} steps = {high_batch_pct:>5.1f}%")
    print()

    # 3. Throughput efficiency
    peak_tps = max(d["throughput_sum"] / d["count"] for d in batch_data.values())
    avg_tps_all = sum(d["throughput_sum"] for d in batch_data.values()) / total_steps
    print(f"  Peak throughput:    {peak_tps:>8.1f} tokens/s")
    print(f"  Average throughput: {avg_tps_all:>8.1f} tokens/s")
    print(f"  Efficiency:         {avg_tps_all/peak_tps*100:>7.1f}%")
    print()

    # 4. KV cache usage stats
    max_usage = max(e["usage"] for e in all_entries)
    avg_usage = sum(e["usage"] for e in all_entries) / len(all_entries)
    max_token = max(e["full_token"] for e in all_entries)
    print(f"  Peak KV cache usage:  {max_usage:.2%}  ({max_token:,} tokens)")
    print(f"  Avg KV cache usage:   {avg_usage:.2%}")
    print()

    # 5. Straggler analysis: find transitions from high to low batch size
    print("=" * 70)
    print("Batch size drop events (straggler indicators):")
    print("=" * 70)
    drops = []
    for i in range(1, len(all_entries)):
        prev_bs = all_entries[i-1]["running"]
        curr_bs = all_entries[i]["running"]
        if prev_bs > 16 and curr_bs <= 8:
            drops.append((i, all_entries[i]["ts"], prev_bs, curr_bs))

    if drops:
        for idx, ts, prev, curr in drops[:10]:
            print(f"  Step {idx}: {prev} → {curr} requests at {ts}")
    else:
        print("  No sudden drops detected (gradual wind-down)")

if __name__ == "__main__":
    main()
