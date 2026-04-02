"""
Fine-grained CUDA event + NVTX profiler for MiniCPM-SALA.

Enable with: export SGLANG_SALA_PROFILE=1
Summary interval: export SGLANG_SALA_PROFILE_INTERVAL=50  (every N model forward calls)
CSV output:      export SGLANG_SALA_PROFILE_LOG=/tmp/sala_profile.csv

Provides:
  - NVTX range markers for Nsight Systems (nsys) visualization
  - CUDA event-based async timing (no synchronize in the hot path)
  - torch.profiler.record_function annotations for torch traces
  - Layer-level breakdown (which layer costs how much, lightning vs minicpm4)
  - Component-level within each layer (norm, qkv_proj, attn_kernel, o_proj, gate/z_proj, mlp)
  - Sub-component within FLA (qk_norm, rope, fla_kernel, o_norm, output_gate)
  - Sub-component within sparse attention (kv_save, topk, flash_attn)
  - Prefill vs decode separation
  - Step-level aggregation with periodic summaries

Architecture:
  - NVTX markers: always pushed when SGLANG_SALA_PROFILE=1, zero cost otherwise
  - CUDA events: recorded async (no sync), flushed lazily when events complete
  - Sync happens ONCE per summary interval, not per-op
"""

import os
import threading
import atexit
from collections import defaultdict
from contextlib import contextmanager

import torch

# ═══════════════════════════════════════════════════════════════════════════
# Global config
# ═══════════════════════════════════════════════════════════════════════════

_ENABLED = os.environ.get("SGLANG_SALA_PROFILE", "0") == "1"
_INTERVAL = int(os.environ.get("SGLANG_SALA_PROFILE_INTERVAL", "50"))
_LOG_FILE = os.environ.get("SGLANG_SALA_PROFILE_LOG", "")

# Accumulated timings: key -> list of elapsed_ms
_timings = defaultdict(list)
_step_count = 0
_lock = threading.Lock()

# Pending CUDA event pairs: list of (key, start_event, end_event)
_pending_events = []

if _ENABLED:
    print("[SALA-PROFILE] Fine-grained profiling ENABLED "
          f"(interval={_INTERVAL}, log={_LOG_FILE or 'stdout'})")


def is_enabled():
    return _ENABLED


# ═══════════════════════════════════════════════════════════════════════════
# CUDA event collection
# ═══════════════════════════════════════════════════════════════════════════

def _flush_pending():
    """Collect elapsed times from completed CUDA event pairs (non-blocking)."""
    global _pending_events
    if not _pending_events:
        return

    still_pending = []
    for key, start_ev, end_ev in _pending_events:
        if end_ev.query():
            elapsed_ms = start_ev.elapsed_time(end_ev)
            with _lock:
                _timings[key].append(elapsed_ms)
        else:
            still_pending.append((key, start_ev, end_ev))
    _pending_events = still_pending


def _flush_all():
    """Force-sync and flush all pending events."""
    if _pending_events:
        torch.cuda.synchronize()
        _flush_pending()


# ═══════════════════════════════════════════════════════════════════════════
# Profiling primitives
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def profile_region(key: str):
    """Context manager: NVTX marker + CUDA event timing + record_function.

    Zero overhead when SGLANG_SALA_PROFILE != "1".
    Skips instrumentation during CUDA graph capture.
    """
    if not _ENABLED:
        yield
        return

    if torch.cuda.is_current_stream_capturing():
        yield
        return

    # NVTX for nsys
    torch.cuda.nvtx.range_push(key)

    # CUDA events for timing
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    # record_function for torch profiler
    from torch.profiler import record_function
    with record_function(key):
        yield

    end_ev.record()
    _pending_events.append((key, start_ev, end_ev))

    torch.cuda.nvtx.range_pop()


def nvtx_push(name: str):
    """Push an NVTX range. Use for manual push/pop when context manager doesn't fit."""
    if _ENABLED and not torch.cuda.is_current_stream_capturing():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    """Pop an NVTX range."""
    if _ENABLED and not torch.cuda.is_current_stream_capturing():
        torch.cuda.nvtx.range_pop()


# ═══════════════════════════════════════════════════════════════════════════
# Step tracking and summary
# ═══════════════════════════════════════════════════════════════════════════

def step_completed(mode: str = "unknown", num_tokens: int = 0):
    """Call after each model forward to flush events and maybe print summary."""
    global _step_count
    if not _ENABLED:
        return

    # Don't touch CUDA events during graph capture
    if torch.cuda.is_current_stream_capturing():
        return

    _flush_pending()
    _step_count += 1

    if _step_count % _INTERVAL == 0:
        print_summary(tag=f"step={_step_count} mode={mode} tokens={num_tokens}")


def print_summary(tag: str = ""):
    """Print accumulated profiling summary grouped by layer and component."""
    if not _ENABLED:
        return

    _flush_all()

    with _lock:
        if not _timings:
            print(f"[SALA-PROFILE] No timings collected yet. {tag}")
            return

        print(f"\n{'='*120}")
        print(f"[SALA-PROFILE] Summary {tag}")
        print(f"{'='*120}")

        # Build component_data from accumulated timings
        component_data = {}
        for key, vals in _timings.items():
            total = sum(vals)
            count = len(vals)
            component_data[key] = {
                "total_ms": total, "count": count,
                "avg_ms": total / count if count else 0,
                "min_ms": min(vals) if vals else 0,
                "max_ms": max(vals) if vals else 0,
            }

        # ── 1. Per-layer summary ──
        layer_totals = defaultdict(lambda: {"total_ms": 0, "count": 0, "type": ""})
        for key, d in component_data.items():
            parts = key.split("/")
            if len(parts) >= 2 and parts[0].startswith("L"):
                layer_key = parts[0]
                layer_totals[layer_key]["total_ms"] += d["total_ms"]
                layer_totals[layer_key]["count"] = max(layer_totals[layer_key]["count"], d["count"])
                layer_totals[layer_key]["type"] = parts[1] if len(parts) > 1 else ""

        grand_total = sum(d["total_ms"] for d in layer_totals.values())
        if grand_total > 0:
            print(f"\n  PER-LAYER SUMMARY (total: {grand_total:.1f} ms across {len(layer_totals)} layers)")
            print(f"  {'Layer':<8} {'Type':<15} {'Total ms':>10} {'%':>6} {'Avg ms':>10} {'Calls':>8}")
            print(f"  {'-'*65}")
            for layer_key in sorted(layer_totals.keys(), key=lambda x: int(x[1:])):
                d = layer_totals[layer_key]
                pct = d["total_ms"] * 100 / grand_total if grand_total else 0
                avg = d["total_ms"] / d["count"] if d["count"] else 0
                print(f"  {layer_key:<8} {d['type']:<15} {d['total_ms']:>10.1f} {pct:>5.1f}% {avg:>10.3f} {d['count']:>8}")

        # ── 2. Component breakdown per layer type ──
        for ltype in ["lightning", "minicpm4"]:
            type_components = defaultdict(lambda: {"total_ms": 0, "count": 0})
            for key, d in component_data.items():
                parts = key.split("/")
                if len(parts) >= 3 and parts[1] == ltype:
                    comp = parts[2]
                    mode_suffix = f" ({parts[3]})" if len(parts) > 3 else ""
                    comp_key = comp + mode_suffix
                    type_components[comp_key]["total_ms"] += d["total_ms"]
                    type_components[comp_key]["count"] += d["count"]

            if type_components:
                type_total = sum(d["total_ms"] for d in type_components.values())
                print(f"\n  [{ltype.upper()}] Component breakdown (total: {type_total:.1f} ms)")
                print(f"  {'Component':<40} {'Total ms':>10} {'%':>6} {'Avg ms':>10} {'Calls':>8}")
                print(f"  {'-'*80}")
                for comp, d in sorted(type_components.items(), key=lambda x: -x[1]["total_ms"]):
                    pct = d["total_ms"] * 100 / type_total if type_total else 0
                    avg = d["total_ms"] / d["count"] if d["count"] else 0
                    print(f"  {comp:<40} {d['total_ms']:>10.1f} {pct:>5.1f}% {avg:>10.3f} {d['count']:>8}")

        # ── 3. Non-layer items (embed, final_norm, logits, etc.) ──
        other_items = {k: d for k, d in component_data.items() if not k.startswith("L")}
        if other_items:
            print(f"\n  [OTHER COMPONENTS]")
            print(f"  {'Component':<40} {'Total ms':>10} {'Avg ms':>10} {'Calls':>8}")
            print(f"  {'-'*70}")
            for comp, d in sorted(other_items.items(), key=lambda x: -x[1]["total_ms"]):
                avg = d["total_ms"] / d["count"] if d["count"] else 0
                print(f"  {comp:<40} {d['total_ms']:>10.1f} {avg:>10.3f} {d['count']:>8}")

        # ── 4. Per-layer detailed breakdown (ALL layers) ──
        if layer_totals:
            print(f"\n  PER-LAYER DETAILED BREAKDOWN")
            print(f"  {'Layer':<6} {'Type':<10} {'Component':<25} "
                  f"{'Total ms':>10} {'%':>6} {'Avg ms':>10} {'Min ms':>10} {'Max ms':>10} {'Calls':>6}")
            print(f"  {'-'*105}")
            for layer_key in sorted(layer_totals.keys(), key=lambda x: int(x[1:])):
                ltype = layer_totals[layer_key]["type"]
                layer_comps = {k: d for k, d in component_data.items()
                               if k.startswith(layer_key + "/")}
                layer_total = sum(d["total_ms"] for d in layer_comps.values())
                for key, d in sorted(layer_comps.items(), key=lambda x: -x[1]["total_ms"]):
                    short_key = "/".join(key.split("/")[2:])
                    pct = d["total_ms"] * 100 / layer_total if layer_total else 0
                    print(f"  {layer_key:<6} {ltype:<10} {short_key:<25} "
                          f"{d['total_ms']:>10.1f} {pct:>5.1f}% {d['avg_ms']:>10.3f} "
                          f"{d['min_ms']:>10.3f} {d['max_ms']:>10.3f} {d['count']:>6}")

        # ── 5. Aggregate: lightning vs minicpm4 total ──
        print(f"\n  AGGREGATE COMPARISON")
        for ltype in ["lightning", "minicpm4"]:
            type_total = sum(d["total_ms"] for k, d in component_data.items()
                            if k.split("/")[1] == ltype if k.startswith("L"))
            type_layers = len([k for k in layer_totals if layer_totals[k]["type"] == ltype])
            if type_total > 0:
                print(f"  {ltype:<15}: {type_total:>10.1f} ms total across {type_layers} layers "
                      f"({type_total*100/grand_total:.1f}% of layer time)" if grand_total else "")

        print(f"\n{'='*120}")

        if _LOG_FILE:
            _write_csv_log()


def _write_csv_log():
    """Write raw timings to CSV for external analysis."""
    with open(_LOG_FILE, "w") as f:
        f.write("key,total_ms,count,avg_ms,min_ms,max_ms\n")
        for key, vals in sorted(_timings.items()):
            total = sum(vals)
            count = len(vals)
            avg = total / count if count else 0
            mn = min(vals) if vals else 0
            mx = max(vals) if vals else 0
            f.write(f"{key},{total:.3f},{count},{avg:.6f},{mn:.6f},{mx:.6f}\n")
    print(f"[SALA-PROFILE] Wrote CSV to {_LOG_FILE}")


def reset():
    """Reset all accumulated timings."""
    global _step_count
    with _lock:
        _timings.clear()
        _step_count = 0
        _pending_events.clear()


# Print summary on exit
if _ENABLED:
    atexit.register(lambda: print_summary(tag="[final]"))
