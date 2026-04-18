#!/usr/bin/env python3
"""Measure prefill vs decode wall-clock time from an sglang server log.

Strategy: Each log line carries a timestamp and is either a Prefill batch
or Decode batch event. The phase at any instant is "whatever was last
logged." We sum gaps between consecutive timestamps, attributing each gap
to whichever phase owned its *start*.

This gives a fair breakdown even when prefill and decode interleave (which
they do heavily at 64-way concurrency with chunked prefill).

Usage:
    python phase_breakdown.py /path/to/server_sparse.log
    python phase_breakdown.py /path/to/server_sparse.log --start "2026-04-17 08:39:23"
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
PREFILL_RE = re.compile(r"Prefill batch.*#new-token:\s*(\d+)")
DECODE_RE = re.compile(r"Decode batch.*gen throughput \(token/s\):\s*([\d.]+)")


def parse_ts(line):
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def classify(line):
    """Return ('prefill', new_tokens) | ('decode', gen_tps) | None."""
    m = PREFILL_RE.search(line)
    if m:
        return ("prefill", int(m.group(1)))
    m = DECODE_RE.search(line)
    if m:
        return ("decode", float(m.group(1)))
    return None


def analyze(path, start=None, end=None):
    events = []  # list of (timestamp, phase, metric)
    with open(path) as f:
        for line in f:
            ts = parse_ts(line)
            if ts is None:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            cls = classify(line)
            if cls is None:
                continue
            events.append((ts, cls[0], cls[1]))

    if not events:
        print("No prefill/decode events found.", file=sys.stderr)
        return

    # Phase wall-clock: attribute gap [t_i, t_{i+1}) to phase of event i.
    prefill_s = 0.0
    decode_s = 0.0
    prefill_batches = 0
    decode_batches = 0
    prefill_tokens_seen = 0
    decode_step_count = 0
    decode_tps_sum = 0.0

    for i in range(len(events) - 1):
        t0, phase, metric = events[i]
        t1 = events[i + 1][0]
        dt = (t1 - t0).total_seconds()
        if phase == "prefill":
            prefill_s += dt
            prefill_batches += 1
            prefill_tokens_seen += metric
        else:
            decode_s += dt
            decode_batches += 1
            decode_step_count += 1
            decode_tps_sum += metric

    # Count the final event's batch but we can't attribute time to it
    last_phase = events[-1][1]
    if last_phase == "prefill":
        prefill_batches += 1
        prefill_tokens_seen += events[-1][2]
    else:
        decode_batches += 1
        decode_step_count += 1
        decode_tps_sum += events[-1][2]

    total_s = prefill_s + decode_s
    span_s = (events[-1][0] - events[0][0]).total_seconds()

    print(f"Log file       : {path}")
    print(f"First event    : {events[0][0]}")
    print(f"Last event     : {events[-1][0]}")
    print(f"Wall-clock span: {span_s:.1f}s ({span_s/60:.1f} min)")
    print(f"Total events   : {len(events)}  (prefill batches: {prefill_batches}, decode batches: {decode_batches})")
    print()
    print("=== Phase breakdown (attributed by last-logged phase) ===")
    if total_s > 0:
        print(f"  Prefill : {prefill_s:8.1f}s  ({prefill_s/total_s*100:5.1f}%)  "
              f"over {prefill_batches} batches, {prefill_tokens_seen:,} new tokens")
        print(f"  Decode  : {decode_s:8.1f}s  ({decode_s/total_s*100:5.1f}%)  "
              f"over {decode_batches} batches")
    print()

    if prefill_tokens_seen > 0 and prefill_s > 0:
        print(f"Prefill token throughput : {prefill_tokens_seen / prefill_s:,.0f} tok/s (averaged over attributed prefill time)")
    if decode_step_count > 0:
        avg_tps = decode_tps_sum / decode_step_count
        print(f"Avg decode gen throughput: {avg_tps:,.1f} tok/s (per-step mean across {decode_step_count} steps)")
        # Note: at concurrency 56, this is aggregate across all in-flight requests,
        # not per-request.
        print(f"  (aggregate across concurrent requests, not per-request)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--start", help="ISO timestamp, ignore events before this")
    ap.add_argument("--end", help="ISO timestamp, ignore events after this")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S") if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S") if args.end else None

    analyze(args.log, start=start, end=end)


if __name__ == "__main__":
    main()
