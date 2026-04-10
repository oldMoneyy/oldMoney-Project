#!/usr/bin/env python3
"""Analyze SGLang server log for scheduling efficiency."""
import re
import sys
from collections import defaultdict

def parse_log(filepath):
    prefill_events = []
    decode_events = []
    
    with open(filepath) as f:
        for line in f:
            # Prefill batch lines
            m = re.search(
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Prefill batch.*'
                r'#new-seq:\s*(\d+).*#new-token:\s*(\d+).*#cached-token:\s*(\d+).*'
                r'token usage:\s*([\d.]+).*mamba usage:\s*([\d.]+).*'
                r'#running-req:\s*(\d+).*#queue-req:\s*(\d+)',
                line
            )
            if m:
                prefill_events.append({
                    'time': m.group(1),
                    'new_seq': int(m.group(2)),
                    'new_token': int(m.group(3)),
                    'cached_token': int(m.group(4)),
                    'token_usage': float(m.group(5)),
                    'mamba_usage': float(m.group(6)),
                    'running': int(m.group(7)),
                    'queue': int(m.group(8)),
                })
            
            # Decode batch lines
            m = re.search(
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Decode batch.*'
                r'#running-req:\s*(\d+).*#queue-req:\s*(\d+)',
                line
            )
            if m:
                decode_events.append({
                    'time': m.group(1),
                    'running': int(m.group(2)),
                    'queue': int(m.group(3)),
                })
    
    return prefill_events, decode_events

def analyze(filepath):
    prefills, decodes = parse_log(filepath)
    
    print(f"=== Server Log Analysis: {filepath} ===\n")
    print(f"Total prefill events: {len(prefills)}")
    print(f"Total decode events:  {len(decodes)}")
    
    if not prefills and not decodes:
        print("No scheduling events found. Is this the right log?")
        return
    
    # --- Prefill analysis ---
    if prefills:
        new_tokens = [e['new_token'] for e in prefills]
        cached_tokens = [e['cached_token'] for e in prefills]
        running_at_prefill = [e['running'] for e in prefills]
        queue_at_prefill = [e['queue'] for e in prefills]
        token_usage = [e['token_usage'] for e in prefills]
        mamba_usage = [e['mamba_usage'] for e in prefills]
        
        print(f"\n--- Prefill Events ({len(prefills)}) ---")
        print(f"New tokens per prefill:  min={min(new_tokens)}, max={max(new_tokens)}, avg={sum(new_tokens)/len(new_tokens):.0f}, total={sum(new_tokens)}")
        print(f"Cached tokens:           min={min(cached_tokens)}, max={max(cached_tokens)}, avg={sum(cached_tokens)/len(cached_tokens):.0f}, total={sum(cached_tokens)}")
        print(f"Running reqs at prefill: min={min(running_at_prefill)}, max={max(running_at_prefill)}, avg={sum(running_at_prefill)/len(running_at_prefill):.1f}")
        print(f"Queued reqs at prefill:  min={min(queue_at_prefill)}, max={max(queue_at_prefill)}, avg={sum(queue_at_prefill)/len(queue_at_prefill):.1f}")
        print(f"Token usage at prefill:  min={min(token_usage):.3f}, max={max(token_usage):.3f}, avg={sum(token_usage)/len(token_usage):.3f}")
        print(f"Mamba usage at prefill:  min={min(mamba_usage):.3f}, max={max(mamba_usage):.3f}, avg={sum(mamba_usage)/len(mamba_usage):.3f}")
        
        # Chunked prefill distribution
        print(f"\nPrefill token size distribution:")
        buckets = defaultdict(int)
        for t in new_tokens:
            if t <= 1024: buckets['<=1K'] += 1
            elif t <= 4096: buckets['1K-4K'] += 1
            elif t <= 8192: buckets['4K-8K'] += 1
            elif t <= 16384: buckets['8K-16K'] += 1
            elif t <= 32768: buckets['16K-32K'] += 1
            else: buckets['>32K'] += 1
        for b in ['<=1K', '1K-4K', '4K-8K', '8K-16K', '16K-32K', '>32K']:
            if b in buckets:
                print(f"  {b:>8s}: {buckets[b]:4d} ({100*buckets[b]/len(new_tokens):.1f}%)")
    
    # --- Decode analysis ---
    if decodes:
        running = [e['running'] for e in decodes]
        queued = [e['queue'] for e in decodes]
        
        print(f"\n--- Decode Events ({len(decodes)}) ---")
        print(f"Running reqs:  min={min(running)}, max={max(running)}, avg={sum(running)/len(running):.1f}")
        print(f"Queued reqs:   min={min(queued)}, max={max(queued)}, avg={sum(queued)/len(queued):.1f}")
        
        # Running req distribution
        print(f"\nDecode batch size distribution:")
        buckets = defaultdict(int)
        for r in running:
            if r <= 1: buckets['1'] += 1
            elif r <= 4: buckets['2-4'] += 1
            elif r <= 8: buckets['5-8'] += 1
            elif r <= 16: buckets['9-16'] += 1
            elif r <= 32: buckets['17-32'] += 1
            elif r <= 48: buckets['33-48'] += 1
            else: buckets['49-64'] += 1
        for b in ['1', '2-4', '5-8', '9-16', '17-32', '33-48', '49-64']:
            if b in buckets:
                print(f"  {b:>5s}: {buckets[b]:5d} ({100*buckets[b]/len(running):.1f}%)")
        
        # Time with queue > 0 (requests waiting = scheduling bottleneck)
        queued_steps = sum(1 for q in queued if q > 0)
        print(f"\nDecode steps with queued requests: {queued_steps}/{len(decodes)} ({100*queued_steps/len(decodes):.1f}%)")
        
        # Mamba usage saturation
        if prefills:
            high_mamba = sum(1 for e in prefills if e['mamba_usage'] > 0.9)
            print(f"Prefills with mamba usage > 90%: {high_mamba}/{len(prefills)} ({100*high_mamba/len(prefills):.1f}%)")
            
            # Check if mamba cache is the bottleneck
            max_mamba = max(mamba_usage)
            if max_mamba > 0.95:
                print(f"WARNING: Mamba cache saturated at {max_mamba:.3f} — this limits concurrency!")
            
            max_token = max(token_usage)
            if max_token > 0.95:
                print(f"WARNING: KV token cache saturated at {max_token:.3f} — this limits concurrency!")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/opt/server.log"
    analyze(filepath)
