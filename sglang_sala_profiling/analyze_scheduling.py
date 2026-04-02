#!/usr/bin/env python3
"""
Analyze SGLang server log for scheduling patterns: prefill/decode ratio, batch sizes, etc.

Usage: python3 analyze_scheduling.py <server.log>
"""
import re
import sys
from collections import Counter, defaultdict

def main():
    logfile = sys.argv[1] if len(sys.argv) > 1 else "/opt/server.log"

    print("=" * 80)
    print("SCHEDULING ANALYSIS")
    print("=" * 80)

    prefill_count = 0
    decode_count = 0
    batch_sizes = []
    extend_lens = []
    decode_lens = []
    memory_warnings = []
    errors = []

    # Patterns to match various log formats
    re_running = re.compile(r'#running-req:\s*(\d+)')
    re_batch = re.compile(r'batch[_\s]?size[=:\s]+(\d+)', re.I)
    re_extend = re.compile(r'(extend|prefill|is_extend|new_fill)', re.I)
    re_decode = re.compile(r'(is_decode|decode_batch|decode_forward)', re.I)
    re_memory = re.compile(r'(memory|oom|evict|swap|out.of.memory)', re.I)
    re_error = re.compile(r'(ERROR|Exception|Traceback)', re.I)
    re_input_len = re.compile(r'input[_\s]?len[=:\s]+(\d+)', re.I)
    re_output_len = re.compile(r'output[_\s]?len[=:\s]+(\d+)', re.I)

    try:
        with open(logfile) as f:
            for line in f:
                if re_extend.search(line):
                    prefill_count += 1
                if re_decode.search(line):
                    decode_count += 1

                m = re_running.search(line)
                if m:
                    batch_sizes.append(int(m.group(1)))

                m = re_batch.search(line)
                if m:
                    batch_sizes.append(int(m.group(1)))

                if re_memory.search(line):
                    memory_warnings.append(line.strip()[:150])

                if re_error.search(line):
                    errors.append(line.strip()[:150])
    except FileNotFoundError:
        print(f"  File not found: {logfile}")
        return

    print(f"\n  Prefill/extend steps:  {prefill_count}")
    print(f"  Decode steps:          {decode_count}")
    if prefill_count + decode_count > 0:
        print(f"  Prefill ratio:         {prefill_count/(prefill_count+decode_count):.1%}")
        print(f"  Decode ratio:          {decode_count/(prefill_count+decode_count):.1%}")

    if batch_sizes:
        print(f"\n  Batch size stats (from #running-req):")
        print(f"    Samples:  {len(batch_sizes)}")
        print(f"    Avg:      {sum(batch_sizes)/len(batch_sizes):.1f}")
        print(f"    Max:      {max(batch_sizes)}")
        print(f"    Min:      {min(batch_sizes)}")
        # Distribution
        dist = Counter()
        for bs in batch_sizes:
            if bs == 0:
                dist["0"] += 1
            elif bs <= 8:
                dist["1-8"] += 1
            elif bs <= 32:
                dist["9-32"] += 1
            elif bs <= 64:
                dist["33-64"] += 1
            else:
                dist[">64"] += 1
        print(f"    Distribution:")
        for bucket in ["0", "1-8", "9-32", "33-64", ">64"]:
            if bucket in dist:
                print(f"      {bucket:>6}: {dist[bucket]:>4} ({dist[bucket]*100/len(batch_sizes):.0f}%)")

    if memory_warnings:
        print(f"\n  Memory warnings ({len(memory_warnings)}):")
        for w in memory_warnings[:10]:
            print(f"    {w}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    if not memory_warnings:
        print(f"\n  No memory warnings/OOM found (good)")

    if not errors:
        print(f"\n  No errors found (good)")


if __name__ == "__main__":
    main()
