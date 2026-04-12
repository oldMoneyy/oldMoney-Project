"""In-process decode profiler for SGLang.

Monkey-patches ModelRunner.forward_decode to capture GPU kernel timings
under CUDA graph. Activated by SGLANG_DECODE_PROFILE=1 env var.

Profiles the first N decode steps after WARMUP steps, then writes results
and exits. This captures real production behavior including CUDA graph replays.

Usage:
  export SGLANG_DECODE_PROFILE=1
  # Launch server normally — it will auto-profile and write results after
  # enough decode steps, then keep running.
  # Send a long-context request to trigger decode.
  # Results written to /opt/decode_profile.txt
"""
import os
import logging

logger = logging.getLogger(__name__)

PROFILE_ENABLED = os.environ.get("SGLANG_DECODE_PROFILE", "0") == "1"
WARMUP_STEPS = 50      # skip initial decode steps (CUDA graph capture phase)
PROFILE_STEPS = 200    # profile this many decode steps
OUTPUT_PATH = "/opt/decode_profile.txt"

_step_count = 0
_profiler = None
_profiling_active = False
_done = False


def maybe_patch_forward_decode(model_runner):
    """Call this after model_runner is initialized to install the profiling hook."""
    if not PROFILE_ENABLED:
        return

    original_forward_decode = model_runner.forward_decode.__func__

    def profiled_forward_decode(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
        global _step_count, _profiler, _profiling_active, _done
        _step_count += 1

        if _step_count == WARMUP_STEPS and not _done:
            import torch
            from torch.profiler import profile, ProfilerActivity
            logger.info(f"[DECODE PROFILER] Starting profile after {WARMUP_STEPS} warmup steps")
            _profiler = profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            )
            _profiler.__enter__()
            _profiling_active = True

        result = original_forward_decode(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors)

        if _profiling_active and _step_count == WARMUP_STEPS + PROFILE_STEPS:
            _profiler.__exit__(None, None, None)
            _profiling_active = False
            _done = True

            table = _profiler.key_averages().table(sort_by="cuda_time_total", row_limit=80)
            logger.info(f"[DECODE PROFILER] Results ({PROFILE_STEPS} decode steps):\n{table}")

            with open(OUTPUT_PATH, "w") as f:
                f.write(f"Decode profile: {PROFILE_STEPS} steps after {WARMUP_STEPS} warmup\n")
                f.write(f"Batch sizes seen during profile window: check server log\n\n")
                f.write(table)

            # Also write a cuda-time-only version sorted differently
            table2 = _profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
            with open(OUTPUT_PATH.replace(".txt", "_self_cuda.txt"), "w") as f:
                f.write(f"Sorted by self_cuda_time_total:\n\n")
                f.write(table2)

            logger.info(f"[DECODE PROFILER] Written to {OUTPUT_PATH}")

        return result

    import types
    model_runner.forward_decode = types.MethodType(profiled_forward_decode, model_runner)
    logger.info(f"[DECODE PROFILER] Patched forward_decode. Will profile {PROFILE_STEPS} steps after {WARMUP_STEPS} warmup.")
