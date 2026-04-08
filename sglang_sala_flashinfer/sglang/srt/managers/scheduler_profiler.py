"""Lightweight scheduler profiler — measures WHERE wall-clock time goes.

Enable: SGLANG_SCHEDULER_PROFILE=1
Output: /tmp/scheduler_profile.jsonl (one JSON line per step)

Post-process:
    python3 -c "
import json, sys
steps = [json.loads(l) for l in open('/tmp/scheduler_profile.jsonl')]
decode_steps = [s for s in steps if s['mode'] == 'decode']
extend_steps = [s for s in steps if s['mode'] == 'extend']

total_wall = steps[-1]['wall_end'] - steps[0]['wall_start']
total_forward = sum(s['forward_ms'] for s in steps)
total_schedule = sum(s['schedule_ms'] for s in steps)
total_process = sum(s['process_ms'] for s in steps)
total_other = total_wall*1000 - total_forward - total_schedule - total_process

print(f'=== SCHEDULER PROFILE ({len(steps)} steps, {total_wall:.1f}s wall) ===')
print(f'  model forward:   {total_forward/1000:.1f}s ({total_forward/total_wall/10:.1f}%)')
print(f'  scheduling:      {total_schedule/1000:.1f}s ({total_schedule/total_wall/10:.1f}%)')
print(f'  result process:  {total_process/1000:.1f}s ({total_process/total_wall/10:.1f}%)')
print(f'  other/gap:       {total_other/1000:.1f}s ({total_other/total_wall/10:.1f}%)')
print()
print(f'Decode steps: {len(decode_steps)}')
if decode_steps:
    fwd = [s['forward_ms'] for s in decode_steps]
    bs = [s['batch_size'] for s in decode_steps]
    print(f'  forward: avg={sum(fwd)/len(fwd):.2f}ms, p50={sorted(fwd)[len(fwd)//2]:.2f}ms, p99={sorted(fwd)[int(len(fwd)*0.99)]:.2f}ms')
    print(f'  batch_size: avg={sum(bs)/len(bs):.1f}, min={min(bs)}, max={max(bs)}')
    # batch size histogram
    from collections import Counter
    bsc = Counter(bs)
    print(f'  batch_size distribution: ' + ', '.join(f'{k}:{v}' for k,v in sorted(bsc.items())))
print()
print(f'Extend steps: {len(extend_steps)}')
if extend_steps:
    fwd = [s['forward_ms'] for s in extend_steps]
    bs = [s['batch_size'] for s in extend_steps]
    pfx = [s.get('prefill_tokens',0) for s in extend_steps]
    print(f'  forward: avg={sum(fwd)/len(fwd):.1f}ms, min={min(fwd):.1f}ms, max={max(fwd):.1f}ms')
    print(f'  prefill_tokens: avg={sum(pfx)/len(pfx):.0f}, total={sum(pfx)}')
    print(f'  decode_batch_size: avg={sum(bs)/len(bs):.1f}')
print()
# Time series: batch size over time
print('Batch size over time (every 1000 steps):')
for i in range(0, len(steps), max(1, len(steps)//20)):
    s = steps[i]
    t = s['wall_start'] - steps[0]['wall_start']
    print(f'  t={t:7.1f}s  step={i:6d}  mode={s[\"mode\"]:7s}  bs={s[\"batch_size\"]:3d}  fwd={s[\"forward_ms\"]:.1f}ms')
"
"""

import json
import os
import time
from contextlib import contextmanager

_ENABLED = os.environ.get("SGLANG_SCHEDULER_PROFILE", "0") == "1"
_outfile = None
_step_count = 0


def is_enabled():
    return _ENABLED


def _get_outfile():
    global _outfile
    if _outfile is None:
        path = os.environ.get("SGLANG_SCHEDULER_PROFILE_PATH", "/tmp/scheduler_profile.jsonl")
        _outfile = open(path, "w")
    return _outfile


def record_step(
    mode: str,          # "decode" or "extend"
    batch_size: int,    # number of requests in batch
    forward_ms: float,  # model forward time
    schedule_ms: float, # get_next_batch_to_run time
    process_ms: float,  # process_batch_result time
    wall_start: float,  # time.perf_counter() at step start
    wall_end: float,    # time.perf_counter() at step end
    prefill_tokens: int = 0,  # number of new prefill tokens in this step
):
    global _step_count
    _step_count += 1
    f = _get_outfile()
    f.write(json.dumps({
        "step": _step_count,
        "mode": mode,
        "batch_size": batch_size,
        "forward_ms": round(forward_ms, 3),
        "schedule_ms": round(schedule_ms, 3),
        "process_ms": round(process_ms, 3),
        "wall_start": round(wall_start, 6),
        "wall_end": round(wall_end, 6),
        "prefill_tokens": prefill_tokens,
    }) + "\n")
    # Flush every 100 steps
    if _step_count % 100 == 0:
        f.flush()


def flush():
    if _outfile is not None:
        _outfile.flush()
