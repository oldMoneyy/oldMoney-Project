"""SALA Profiler - Non-invasive CUDA event timing for MiniCPM-SALA inference.

Enable with: export SGLANG_SALA_PROFILE=1
Output: /tmp/sglang_profile.jsonl (one JSON record per forward step)
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_SALA_PROFILE", "0") == "1"
_PROFILER: Optional["SalaProfiler"] = None


def is_profiling_enabled() -> bool:
    return _ENABLED


def _is_capturing() -> bool:
    """Check if CUDA graph capture is in progress."""
    if hasattr(torch.cuda, "is_current_stream_capturing"):
        return torch.cuda.is_current_stream_capturing()
    # Fallback for older PyTorch
    try:
        s = torch.cuda.current_stream()
        return s.is_capturing() if hasattr(s, "is_capturing") else False
    except Exception:
        return False


def get_profiler() -> Optional["SalaProfiler"]:
    global _PROFILER
    if not _ENABLED:
        return None
    # Cannot use CUDA events during CUDA graph capture
    if _is_capturing():
        return None
    if _PROFILER is None:
        _PROFILER = SalaProfiler()
    return _PROFILER


class CudaTimer:
    """Context manager that records GPU elapsed time using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()

    def sync_and_get_ms(self) -> float:
        self.end_event.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        return self.elapsed_ms


class SalaProfiler:
    """Collects per-step timing data and writes JSONL output."""

    def __init__(self, output_path: str = "/tmp/sglang_profile.jsonl"):
        self.output_path = output_path
        self.step_count = 0
        self.summary_interval = 50

        # Per-step accumulators (reset each step)
        self._current_step = {}
        self._layer_times = []

        # Running averages for summary
        self._avg_accum = defaultdict(lambda: {"total": 0.0, "count": 0})
        self._layer_avg_accum = defaultdict(lambda: defaultdict(lambda: {"total": 0.0, "count": 0}))

        # Clear output file
        with open(self.output_path, "w") as f:
            pass
        logger.info(f"SALA Profiler enabled. Output: {self.output_path}")

    def begin_step(self, forward_mode: str = "", batch_size: int = 0, num_tokens: int = 0):
        """Call at the beginning of each forward step."""
        self._current_step = {
            "step": self.step_count,
            "forward_mode": forward_mode,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
            "wall_start": time.perf_counter(),
        }
        self._layer_times = []

    def record_layer(self, layer_id: int, layer_type: str,
                     layernorm_ms: float, attn_ms: float, mlp_ms: float):
        self._layer_times.append({
            "layer_id": layer_id,
            "type": layer_type,
            "layernorm_ms": round(layernorm_ms, 3),
            "attn_ms": round(attn_ms, 3),
            "mlp_ms": round(mlp_ms, 3),
            "total_ms": round(layernorm_ms + attn_ms + mlp_ms, 3),
        })

        # Update layer averages
        acc = self._layer_avg_accum[layer_id]
        for key, val in [("layernorm_ms", layernorm_ms), ("attn_ms", attn_ms), ("mlp_ms", mlp_ms)]:
            acc[key]["total"] += val
            acc[key]["count"] += 1

    def record_timing(self, name: str, value_ms: float):
        self._current_step[name] = round(value_ms, 3)

        acc = self._avg_accum[name]
        acc["total"] += value_ms
        acc["count"] += 1

    def record_scheduler_timing(self, recv_ms: float, schedule_ms: float,
                                 run_batch_ms: float, process_result_ms: float):
        """Record scheduler loop timing (called outside of begin_step/end_step)."""
        for name, val in [("sched_recv_ms", recv_ms), ("sched_schedule_ms", schedule_ms),
                          ("sched_run_batch_ms", run_batch_ms), ("sched_process_result_ms", process_result_ms)]:
            self._current_step[name] = round(val, 3)
            acc = self._avg_accum[name]
            acc["total"] += val
            acc["count"] += 1

    def end_step(self):
        """Call at the end of each forward step. Writes the record."""
        self._current_step["wall_total_ms"] = round(
            (time.perf_counter() - self._current_step.get("wall_start", 0)) * 1000, 3
        )
        self._current_step.pop("wall_start", None)
        self._current_step["layer_times"] = self._layer_times

        # Write JSONL
        with open(self.output_path, "a") as f:
            f.write(json.dumps(self._current_step) + "\n")

        self.step_count += 1

        # Periodic summary
        if self.step_count % self.summary_interval == 0:
            self._print_summary()

        # Reset
        self._current_step = {}
        self._layer_times = []

    def _print_summary(self):
        lines = [f"\n===== SALA Profile Summary (last {self.summary_interval} steps, step {self.step_count}) ====="]

        # Top-level averages
        for name, acc in sorted(self._avg_accum.items()):
            if acc["count"] > 0:
                avg = acc["total"] / acc["count"]
                lines.append(f"  {name}: {avg:.3f} ms (avg)")

        # Per-layer averages
        if self._layer_avg_accum:
            lines.append("  --- Per-layer averages ---")
            total_attn = 0.0
            total_mlp = 0.0
            total_ln = 0.0
            count = 0
            for lid in sorted(self._layer_avg_accum.keys()):
                acc = self._layer_avg_accum[lid]
                ln_avg = acc["layernorm_ms"]["total"] / max(acc["layernorm_ms"]["count"], 1)
                attn_avg = acc["attn_ms"]["total"] / max(acc["attn_ms"]["count"], 1)
                mlp_avg = acc["mlp_ms"]["total"] / max(acc["mlp_ms"]["count"], 1)
                total = ln_avg + attn_avg + mlp_avg
                lines.append(f"    Layer {lid:2d}: ln={ln_avg:.3f} attn={attn_avg:.3f} mlp={mlp_avg:.3f} total={total:.3f} ms")
                total_attn += attn_avg
                total_mlp += mlp_avg
                total_ln += ln_avg
                count += 1

            if count > 0:
                lines.append(f"    SUM: ln={total_ln:.3f} attn={total_attn:.3f} mlp={total_mlp:.3f} total={total_ln+total_attn+total_mlp:.3f} ms")

        lines.append("=" * 60)
        logger.info("\n".join(lines))

        # Reset accumulators
        self._avg_accum.clear()
        self._layer_avg_accum.clear()
