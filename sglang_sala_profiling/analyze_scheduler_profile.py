#!/usr/bin/env python3
"""Analyze scheduler profiling output from SGLANG_SCHEDULER_PROFILE=1.

Usage:
    python3 analyze_scheduler_profile.py /tmp/scheduler_profile.jsonl
"""

import json
import sys
from collections import Counter


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <scheduler_profile.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    if not steps:
        print("No steps recorded.")
        sys.exit(1)

    decode_steps = [s for s in steps if s["mode"] == "decode"]
    extend_steps = [s for s in steps if s["mode"] == "extend"]

    total_wall = steps[-1]["wall_end"] - steps[0]["wall_start"]
    total_forward = sum(s["forward_ms"] for s in steps)
    total_schedule = sum(s["schedule_ms"] for s in steps)
    total_process = sum(s["process_ms"] for s in steps)
    total_other = total_wall * 1000 - total_forward - total_schedule - total_process

    print("=" * 70)
    print(f"SCHEDULER PROFILE: {len(steps)} steps, {total_wall:.1f}s wall time")
    print("=" * 70)
    print()

    # --- Overall time breakdown ---
    print("--- WHERE DOES WALL-CLOCK TIME GO? ---")
    print(f"  model forward:     {total_forward/1000:8.1f}s  ({total_forward/total_wall/10:5.1f}%)")
    print(f"  scheduling:        {total_schedule/1000:8.1f}s  ({total_schedule/total_wall/10:5.1f}%)")
    print(f"  result processing: {total_process/1000:8.1f}s  ({total_process/total_wall/10:5.1f}%)")
    print(f"  other/gap:         {total_other/1000:8.1f}s  ({total_other/total_wall/10:5.1f}%)")
    print(f"  TOTAL:             {total_wall:8.1f}s")
    print()

    # --- Decode steps ---
    print(f"--- DECODE STEPS ({len(decode_steps)}) ---")
    if decode_steps:
        fwd = sorted([s["forward_ms"] for s in decode_steps])
        sched = sorted([s["schedule_ms"] for s in decode_steps])
        proc = sorted([s["process_ms"] for s in decode_steps])
        bs = [s["batch_size"] for s in decode_steps]

        print(f"  forward_ms:   avg={sum(fwd)/len(fwd):.2f}  "
              f"p50={fwd[len(fwd)//2]:.2f}  p95={fwd[int(len(fwd)*0.95)]:.2f}  "
              f"p99={fwd[int(len(fwd)*0.99)]:.2f}  max={fwd[-1]:.2f}")
        print(f"  schedule_ms:  avg={sum(sched)/len(sched):.3f}  "
              f"p50={sched[len(sched)//2]:.3f}  p99={sched[int(len(sched)*0.99)]:.3f}  "
              f"max={sched[-1]:.3f}")
        print(f"  process_ms:   avg={sum(proc)/len(proc):.3f}  "
              f"p50={proc[len(proc)//2]:.3f}  p99={proc[int(len(proc)*0.99)]:.3f}  "
              f"max={proc[-1]:.3f}")
        print(f"  batch_size:   avg={sum(bs)/len(bs):.1f}  min={min(bs)}  max={max(bs)}")

        # Batch size histogram
        bsc = Counter(bs)
        print(f"  batch_size distribution:")
        for k in sorted(bsc.keys()):
            bar = "#" * min(50, bsc[k] * 50 // max(bsc.values()))
            print(f"    bs={k:3d}: {bsc[k]:6d} steps  {bar}")

        # Decode time breakdown
        decode_fwd_total = sum(fwd)
        decode_sched_total = sum(sched)
        decode_proc_total = sum(proc)
        print(f"  time totals:  forward={decode_fwd_total/1000:.1f}s  "
              f"sched={decode_sched_total/1000:.1f}s  proc={decode_proc_total/1000:.1f}s")
    print()

    # --- Extend steps ---
    print(f"--- EXTEND (PREFILL) STEPS ({len(extend_steps)}) ---")
    if extend_steps:
        fwd = sorted([s["forward_ms"] for s in extend_steps])
        sched = sorted([s["schedule_ms"] for s in extend_steps])
        proc = sorted([s["process_ms"] for s in extend_steps])
        pfx = [s.get("prefill_tokens", 0) for s in extend_steps]
        bs = [s["batch_size"] for s in extend_steps]

        print(f"  forward_ms:   avg={sum(fwd)/len(fwd):.1f}  "
              f"min={fwd[0]:.1f}  p50={fwd[len(fwd)//2]:.1f}  max={fwd[-1]:.1f}")
        print(f"  schedule_ms:  avg={sum(sched)/len(sched):.2f}  max={sched[-1]:.2f}")
        print(f"  process_ms:   avg={sum(proc)/len(proc):.2f}  max={proc[-1]:.2f}")
        print(f"  prefill_tokens: total={sum(pfx):,}  avg={sum(pfx)/len(pfx):.0f}")
        print(f"  batch_size:   avg={sum(bs)/len(bs):.1f}  min={min(bs)}  max={max(bs)}")

        extend_fwd_total = sum(fwd)
        print(f"  time totals:  forward={extend_fwd_total/1000:.1f}s  "
              f"sched={sum(sched)/1000:.1f}s  proc={sum(proc)/1000:.1f}s")
    print()

    # --- Timeline ---
    print("--- TIMELINE (batch size & mode over time) ---")
    n_samples = min(30, len(steps))
    for i in range(0, len(steps), max(1, len(steps) // n_samples)):
        s = steps[i]
        t = s["wall_start"] - steps[0]["wall_start"]
        pfx = s.get("prefill_tokens", 0)
        pfx_str = f"  pfx={pfx}" if pfx > 0 else ""
        print(f"  t={t:7.1f}s  step={i:6d}  {s['mode']:7s}  "
              f"bs={s['batch_size']:3d}  fwd={s['forward_ms']:8.1f}ms{pfx_str}")
    print()

    # --- Prefill impact analysis ---
    if extend_steps and decode_steps:
        print("--- PREFILL IMPACT ON THROUGHPUT ---")
        # How much time was spent in extend vs decode
        extend_wall = sum(s["wall_end"] - s["wall_start"] for s in extend_steps)
        decode_wall = sum(s["wall_end"] - s["wall_start"] for s in decode_steps)
        print(f"  Wall time in extend steps: {extend_wall:.1f}s ({extend_wall/total_wall*100:.1f}%)")
        print(f"  Wall time in decode steps: {decode_wall:.1f}s ({decode_wall/total_wall*100:.1f}%)")
        print()

        # Decode tokens generated during extend phase vs pure decode phase
        # (approximate: assume 1 output token per request per step)
        extend_phase_end = max(s["wall_end"] for s in extend_steps)
        extend_phase_start = min(s["wall_start"] for s in extend_steps)
        decode_during_extend = [s for s in decode_steps
                                if s["wall_start"] < extend_phase_end]
        decode_after_extend = [s for s in decode_steps
                               if s["wall_start"] >= extend_phase_end]

        tokens_during_extend = sum(s["batch_size"] for s in decode_during_extend)
        tokens_after_extend = sum(s["batch_size"] for s in decode_after_extend)
        total_output_tokens = tokens_during_extend + tokens_after_extend

        if total_output_tokens > 0:
            print(f"  Prefill phase: {extend_phase_start - steps[0]['wall_start']:.0f}s "
                  f"to {extend_phase_end - steps[0]['wall_start']:.0f}s "
                  f"({extend_phase_end - extend_phase_start:.0f}s)")
            print(f"  Decode steps during prefill phase: {len(decode_during_extend)}")
            print(f"  Decode steps after prefill phase:  {len(decode_after_extend)}")
            print(f"  Output tokens during prefill: {tokens_during_extend:,} "
                  f"({tokens_during_extend/total_output_tokens*100:.1f}%)")
            print(f"  Output tokens after prefill:  {tokens_after_extend:,} "
                  f"({tokens_after_extend/total_output_tokens*100:.1f}%)")

            if decode_during_extend:
                avg_bs_during = sum(s["batch_size"] for s in decode_during_extend) / len(decode_during_extend)
                print(f"  Avg batch size during prefill: {avg_bs_during:.1f}")
            if decode_after_extend:
                avg_bs_after = sum(s["batch_size"] for s in decode_after_extend) / len(decode_after_extend)
                print(f"  Avg batch size after prefill:  {avg_bs_after:.1f}")
        print()

    # --- Verdict ---
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    fwd_pct = total_forward / total_wall / 10
    sched_pct = total_schedule / total_wall / 10
    proc_pct = total_process / total_wall / 10
    other_pct = total_other / total_wall / 10

    if fwd_pct > 90:
        print("  Model forward dominates (>90%). Scheduling is NOT a bottleneck.")
        print("  Focus on kernel-level optimizations.")
    elif fwd_pct > 80:
        print(f"  Model forward is {fwd_pct:.0f}%. Some overhead exists but forward dominates.")
        print(f"  Scheduling: {sched_pct:.1f}%, Processing: {proc_pct:.1f}%, Other: {other_pct:.1f}%")
    else:
        print(f"  WARNING: Only {fwd_pct:.0f}% of time in model forward!")
        print(f"  Scheduling: {sched_pct:.1f}%, Processing: {proc_pct:.1f}%, Other: {other_pct:.1f}%")
        print("  Scheduling/processing overhead IS a significant bottleneck.")

    if decode_steps:
        avg_bs = sum(s["batch_size"] for s in decode_steps) / len(decode_steps)
        if avg_bs < 20:
            print(f"  WARNING: Avg decode batch size = {avg_bs:.1f} (out of 64).")
            print("  Prefill ramp-up is severely limiting throughput.")
        elif avg_bs < 40:
            print(f"  NOTE: Avg decode batch size = {avg_bs:.1f}. Moderate prefill impact.")
        else:
            print(f"  Decode batch size avg = {avg_bs:.1f}. Batch utilization is good.")


if __name__ == "__main__":
    main()
