"""CUDA event-based profiling for MiniCPM-SALA inference.

Enable by setting SGLANG_PROFILE=1 environment variable.
Results are dumped to _official_profiling/ after SGLANG_PROFILE_STEPS decode steps.
"""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

PROFILING_ENABLED = os.environ.get("SGLANG_PROFILE", "0") == "1"
PROFILE_STEPS = int(os.environ.get("SGLANG_PROFILE_STEPS", "50"))
PROFILE_WARMUP = int(os.environ.get("SGLANG_PROFILE_WARMUP", "5"))
PROFILE_OUTPUT_DIR = os.environ.get(
    "SGLANG_PROFILE_OUTPUT",
    str(Path(__file__).resolve().parent.parent / "_official_profiling"),
)


class ProfilingCollector:
    """Collects CUDA event-based timing for profiling inference bottlenecks."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.enabled = PROFILING_ENABLED
        self.step_count = 0
        self.warmup = PROFILE_WARMUP
        self.max_steps = PROFILE_STEPS
        self.dumped = False

        # {op_name: [(start_event, end_event), ...]}
        self._pending_events = defaultdict(list)
        # {op_name: [elapsed_ms, ...]}
        self._timings = defaultdict(list)
        # Track per-step total to compute percentages
        self._step_starts = []
        self._step_ends = []

    @property
    def active(self):
        return (
            self.enabled
            and self.step_count >= self.warmup
            and self.step_count < self.warmup + self.max_steps
            and not self.dumped
        )

    def step_begin(self):
        """Call at the start of each model forward (decode step)."""
        if not self.enabled or self.dumped:
            return
        if self.active:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._step_starts.append(e)

    def step_end(self):
        """Call at the end of each model forward (decode step)."""
        if not self.enabled or self.dumped:
            return
        if self.active:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._step_ends.append(e)
        self.step_count += 1
        if self.step_count >= self.warmup + self.max_steps:
            self._flush_and_dump()

    @contextmanager
    def timer(self, name):
        """Context manager to time a CUDA operation.

        Usage:
            with profiler.timer("layer.0.attn.qkv_proj"):
                qkv, _ = self.qkv_proj(hidden_states)
        """
        if not self.active:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._pending_events[name].append((start, end))

    def record_start(self, name):
        """Record start event for an operation (non-context-manager style)."""
        if not self.active:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return (name, e)

    def record_end(self, handle):
        """Record end event. handle is the return value of record_start."""
        if handle is None:
            return
        name, start = handle
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending_events[name].append((start, end))

    def _flush_and_dump(self):
        """Synchronize, compute timings, and dump results."""
        if self.dumped:
            return
        self.dumped = True
        torch.cuda.synchronize()

        # Compute per-op timings
        for name, events in self._pending_events.items():
            for start, end in events:
                elapsed = start.elapsed_time(end)
                self._timings[name].append(elapsed)

        # Compute per-step total
        step_totals = []
        for s, e in zip(self._step_starts, self._step_ends):
            step_totals.append(s.elapsed_time(e))

        self._save_results(step_totals)

    def _save_results(self, step_totals):
        os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)

        # Build summary
        avg_step = sum(step_totals) / len(step_totals) if step_totals else 0

        # Group timings by category
        # Naming convention: "layer.{id}.{mixer_type}.{op}" or "layer.{id}.{op}"
        op_summary = {}
        category_totals = defaultdict(float)

        for name, times in sorted(self._timings.items()):
            avg = sum(times) / len(times)
            total = sum(times)
            count = len(times)
            op_summary[name] = {
                "avg_ms": round(avg, 4),
                "total_ms": round(total, 2),
                "count": count,
                "pct_of_step": round(avg / avg_step * 100, 2) if avg_step > 0 else 0,
            }

            # Extract category (e.g., "qkv_proj", "rope", "attn", "mlp", etc.)
            parts = name.split(".")
            if len(parts) >= 3:
                category = parts[-1]  # Last part is the op
            else:
                category = name
            category_totals[category] += avg

        # Aggregate by layer type
        layer_type_totals = defaultdict(lambda: defaultdict(float))
        for name, times in self._timings.items():
            avg = sum(times) / len(times)
            parts = name.split(".")
            if len(parts) >= 3:
                layer_type = parts[2] if len(parts) > 3 else "common"
                op = parts[-1]
                layer_type_totals[layer_type][op] += avg

        # Per-layer breakdown
        layer_breakdown = {}
        for name, times in sorted(self._timings.items()):
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "layer":
                layer_id = parts[1]
                if layer_id not in layer_breakdown:
                    layer_breakdown[layer_id] = {}
                op = ".".join(parts[2:])
                layer_breakdown[layer_id][op] = {
                    "avg_ms": round(sum(times) / len(times), 4),
                    "count": len(times),
                }

        # Save detailed JSON
        result = {
            "config": {
                "warmup_steps": self.warmup,
                "profile_steps": self.max_steps,
                "total_steps_seen": self.step_count,
            },
            "step_total": {
                "avg_ms": round(avg_step, 4),
                "min_ms": round(min(step_totals), 4) if step_totals else 0,
                "max_ms": round(max(step_totals), 4) if step_totals else 0,
            },
            "op_breakdown": op_summary,
            "category_totals_ms": {
                k: round(v, 4) for k, v in sorted(category_totals.items(), key=lambda x: -x[1])
            },
            "layer_type_breakdown": {
                lt: {k: round(v, 4) for k, v in sorted(ops.items(), key=lambda x: -x[1])}
                for lt, ops in layer_type_totals.items()
            },
        }

        with open(os.path.join(PROFILE_OUTPUT_DIR, "decode_breakdown.json"), "w") as f:
            json.dump(result, f, indent=2)

        with open(os.path.join(PROFILE_OUTPUT_DIR, "layer_breakdown.json"), "w") as f:
            json.dump(layer_breakdown, f, indent=2)

        # Human-readable summary
        lines = []
        lines.append("=" * 70)
        lines.append("MiniCPM-SALA Decode Profiling Summary")
        lines.append("=" * 70)
        lines.append(f"Steps profiled: {self.max_steps} (after {self.warmup} warmup)")
        lines.append(f"Avg step time: {avg_step:.4f} ms")
        lines.append("")
        lines.append("--- Category Breakdown (avg per step) ---")
        for cat, ms in sorted(category_totals.items(), key=lambda x: -x[1]):
            pct = ms / avg_step * 100 if avg_step > 0 else 0
            lines.append(f"  {cat:30s} {ms:10.4f} ms  ({pct:5.1f}%)")
        lines.append("")
        lines.append("--- By Layer Type ---")
        for lt, ops in layer_type_totals.items():
            lt_total = sum(ops.values())
            lines.append(f"\n  [{lt}] total avg: {lt_total:.4f} ms")
            for op, ms in sorted(ops.items(), key=lambda x: -x[1]):
                lines.append(f"    {op:28s} {ms:10.4f} ms")
        lines.append("")
        lines.append("--- Top 20 Operations ---")
        sorted_ops = sorted(op_summary.items(), key=lambda x: -x[1]["avg_ms"])[:20]
        for name, info in sorted_ops:
            lines.append(
                f"  {name:50s} avg={info['avg_ms']:10.4f} ms  ({info['pct_of_step']:5.1f}%)"
            )

        summary = "\n".join(lines)
        with open(os.path.join(PROFILE_OUTPUT_DIR, "summary.txt"), "w") as f:
            f.write(summary)

        print("\n" + summary)
        print(f"\nProfiling results saved to {PROFILE_OUTPUT_DIR}/")


# Global accessor
profiler = ProfilingCollector.get()
