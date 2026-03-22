#!/usr/bin/env python3
"""Generate a benchmark dataset matching SOAR 2026 competition distribution."""
import json, random, sys

random.seed(42)
NUM_REQUESTS = 64  # enough for statistical stability

# Competition input distribution (tokens)
INPUT_DIST = [
    (0.25, 100, 4000),      # 25% short
    (0.10, 4000, 16000),     # 10% medium  
    (0.15, 16000, 32000),    # 15% medium-long
    (0.35, 32000, 128000),   # 35% long
    (0.15, 128000, 160000),  # 15% very long
]

# Competition output distribution (tokens)  
OUTPUT_DIST = [
    (0.35, 10, 512),
    (0.25, 512, 2000),
    (0.10, 2000, 4000),
    (0.15, 4000, 16000),
    (0.15, 16000, 32000),
]

def sample_from_dist(dist):
    r = random.random()
    cumulative = 0
    for prob, lo, hi in dist:
        cumulative += prob
        if r <= cumulative:
            return random.randint(lo, hi)
    return dist[-1][2]

requests = []
for i in range(NUM_REQUESTS):
    input_len = sample_from_dist(INPUT_DIST)
    output_len = sample_from_dist(OUTPUT_DIST)
    # Generate dummy content (~4 chars per token)
    prompt = "x " * input_len
    response = "y " * output_len
    requests.append({
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    })

output_path = "/opt/oldMoney-Project/bench/competition_bench.jsonl"
with open(output_path, "w") as f:
    for r in requests:
        f.write(json.dumps(r) + "\n")

# Print distribution stats
input_lens = [sample_from_dist(INPUT_DIST) for _ in range(NUM_REQUESTS)]
output_lens = [sample_from_dist(OUTPUT_DIST) for _ in range(NUM_REQUESTS)]
print(f"Generated {NUM_REQUESTS} requests to {output_path}")
print(f"Input tokens:  min={min(input_lens)}, median={sorted(input_lens)[len(input_lens)//2]}, max={max(input_lens)}")
print(f"Output tokens: min={min(output_lens)}, median={sorted(output_lens)[len(output_lens)//2]}, max={max(output_lens)}")
