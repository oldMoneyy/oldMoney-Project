#!/usr/bin/env python3
"""
Compare profiling results across backends × concurrency tiers.
Computes SOAR 2026 competition-weighted scores.

Directory structure:
  results/<backend>/<tier>/bench_baseline.txt
  results/<backend>/<tier>/sala_timings.csv
  results/<backend>/<tier>/gpu_analysis.txt

Usage: python3 compare_backends.py <results_dir>
"""
import os
import re
import sys
from collections import defaultdict

TIERS = ["c1", "c8", "c64"]
TIER_WEIGHTS = {"c1": 0.40, "c8": 0.30, "c64": 0.30}
TIER_LABELS = {
    "c1": "concurrent=1 (40%)",
    "c8": "concurrent=8 (30%)",
    "c64": "unlimited (30%)",
}
BOUNTY_TARGETS = {"c1": 420, "c8": 420, "c64": 720}


def extract_bench_metrics(filepath):
    metrics = {}
    for candidate in [filepath,
                      filepath.replace("bench_baseline", "bench_sala"),
                      filepath.replace("bench_baseline", "bench_phase1")]:
        try:
            with open(candidate) as f:
                text = f.read()
            break
        except FileNotFoundError:
            text = ""
            continue

    if not text:
        return metrics

    patterns = {
        "total_duration_s": r"[Tt]otal\s+duration[:\s]+([0-9.]+)",
        "input_throughput": r"[Ii]nput\s+throughput[:\s]+([0-9.]+)",
        "output_throughput": r"[Oo]utput\s+throughput[:\s]+([0-9.]+)",
        "avg_ttft_ms": r"(?:[Aa]vg|[Mm]ean)\s+TTFT[:\s]+([0-9.]+)",
        "p99_ttft_ms": r"P99\s+TTFT[:\s]+([0-9.]+)",
        "avg_itl_ms": r"(?:[Aa]vg|[Mm]ean)\s+ITL[:\s]+([0-9.]+)",
        "p99_itl_ms": r"P99\s+ITL[:\s]+([0-9.]+)",
        "avg_tpot_ms": r"(?:[Aa]vg|[Mm]ean)\s+TPOT[:\s]+([0-9.]+)",
        "completed_requests": r"(?:[Cc]ompleted|[Ss]uccessful)\s+requests[:\s]+(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.I)
        if m:
            metrics[key] = float(m.group(1))

    metrics["_raw"] = text
    return metrics


def extract_gpu_summary(filepath):
    data = {}
    try:
        with open(filepath) as f:
            text = f.read()
        for pattern, key in [
            (r"Avg SM%:\s+([0-9.]+)", "avg_sm"),
            (r"Avg Mem%:\s+([0-9.]+)", "avg_mem"),
            (r"Peak:\s+(\d+)", "peak_mem_mib"),
        ]:
            m = re.search(pattern, text)
            if m:
                data[key] = float(m.group(1))
    except FileNotFoundError:
        pass
    return data


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/oldMoney-Project/sglang_sala_profiling/results"

    backends = ["flashinfer", "minicpm_flashinfer"]

    # Collect all data: backend -> tier -> {bench, gpu}
    all_data = {}
    for backend in backends:
        all_data[backend] = {}
        for tier in TIERS:
            tdir = os.path.join(results_dir, backend, tier)
            if os.path.isdir(tdir):
                all_data[backend][tier] = {
                    "bench": extract_bench_metrics(os.path.join(tdir, "bench_baseline.txt")),
                    "gpu": extract_gpu_summary(os.path.join(tdir, "gpu_analysis.txt")),
                }

    # ═══════════════════════════════════════════════════════════════
    # Per-tier comparison
    # ═══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("BACKEND COMPARISON: flashinfer vs minicpm_flashinfer")
    print("=" * 100)

    for tier in TIERS:
        print(f"\n{'─'*100}")
        print(f"  TIER: {TIER_LABELS[tier]}")
        print(f"{'─'*100}")

        fi = all_data.get("flashinfer", {}).get(tier, {})
        mcfi = all_data.get("minicpm_flashinfer", {}).get(tier, {})

        if not fi and not mcfi:
            print("  No data available for this tier.")
            continue

        print(f"  {'Metric':<35} {'flashinfer':>18} {'minicpm_fi':>18} {'Delta':>12}")
        print(f"  {'-'*85}")

        bench_keys = [
            ("total_duration_s", "Duration (s)", True),
            ("input_throughput", "Input Tput (tok/s)", False),
            ("output_throughput", "Output Tput (tok/s)", False),
            ("avg_ttft_ms", "Avg TTFT (ms)", True),
            ("p99_ttft_ms", "P99 TTFT (ms)", True),
            ("avg_itl_ms", "Avg ITL (ms)", True),
            ("p99_itl_ms", "P99 ITL (ms)", True),
            ("completed_requests", "Completed Reqs", False),
        ]
        for key, label, lower_better in bench_keys:
            v1 = fi.get("bench", {}).get(key)
            v2 = mcfi.get("bench", {}).get(key)
            if v1 is not None and v2 is not None:
                delta_pct = (v2 - v1) / v1 * 100 if v1 != 0 else 0
                if lower_better:
                    marker = " BETTER" if v2 < v1 else " worse" if v2 > v1 else ""
                else:
                    marker = " BETTER" if v2 > v1 else " worse" if v2 < v1 else ""
                print(f"  {label:<35} {v1:>18.2f} {v2:>18.2f} {delta_pct:>+10.1f}%{marker}")
            else:
                v1s = f"{v1:.2f}" if v1 is not None else "N/A"
                v2s = f"{v2:.2f}" if v2 is not None else "N/A"
                print(f"  {label:<35} {v1s:>18} {v2s:>18}")

        # GPU
        for key, label in [("avg_sm", "Avg SM%"), ("peak_mem_mib", "Peak Mem (MiB)")]:
            v1 = fi.get("gpu", {}).get(key)
            v2 = mcfi.get("gpu", {}).get(key)
            if v1 is not None or v2 is not None:
                v1s = f"{v1:.1f}" if v1 is not None else "N/A"
                v2s = f"{v2:.1f}" if v2 is not None else "N/A"
                print(f"  {label:<35} {v1s:>18} {v2s:>18}")

    # ═══════════════════════════════════════════════════════════════
    # Competition-weighted score
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("SOAR 2026 COMPETITION SCORE ESTIMATE")
    print(f"{'='*100}")
    print(f"  Scoring: S_N = (best_duration / your_duration) × 100")
    print(f"  Final = S1×40% + S8×30% + S∞×30%")
    print()

    # Find best duration per tier across backends
    best_per_tier = {}
    for tier in TIERS:
        durations = []
        for backend in backends:
            d = all_data.get(backend, {}).get(tier, {}).get("bench", {}).get("total_duration_s")
            if d is not None:
                durations.append(d)
        if durations:
            best_per_tier[tier] = min(durations)

    for backend in backends:
        scores = {}
        weighted = 0
        print(f"  {backend}:")
        for tier in TIERS:
            dur = all_data.get(backend, {}).get(tier, {}).get("bench", {}).get("total_duration_s")
            best = best_per_tier.get(tier)
            if dur is not None and best is not None:
                s = (best / dur) * 100
                scores[tier] = s
                weighted += s * TIER_WEIGHTS[tier]
                bounty = BOUNTY_TARGETS[tier]
                vs_bounty = "BEATS BOUNTY" if dur <= bounty else f"need -{dur-bounty:.0f}s"
                print(f"    {TIER_LABELS[tier]:>30}:  dur={dur:>8.1f}s  score={s:>6.1f}  ({vs_bounty})")
            else:
                print(f"    {TIER_LABELS[tier]:>30}:  N/A")
        if scores:
            print(f"    {'Weighted Performance Score':>30}:  {weighted:>8.1f}")
        print()

    # Bounty analysis
    print(f"  Bounty targets: c1={BOUNTY_TARGETS['c1']}s, c8={BOUNTY_TARGETS['c8']}s, c64={BOUNTY_TARGETS['c64']}s")

    print(f"\n{'='*100}")
    print("COMPARISON COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
