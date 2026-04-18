#!/usr/bin/env python3
"""Generate stratified datasets for sparse-vs-dense concurrency analysis.

Produces:
  - competition_bench_A_long32.jsonl   — 32 prompts, all >128K input
  - competition_bench_B_short64.jsonl  — 64 prompts, all <32K input

Output length distribution matches the competition spec in both files,
so the only variable across A and B is input length.
"""
import json
import random

random.seed(42)

BASE = "/home/work/compass_max_posttrain_1/.cz/sala/oldMoney-Project/bench"
PATH_A = f"{BASE}/competition_bench_A_long32.jsonl"
PATH_B = f"{BASE}/competition_bench_B_short64.jsonl"

# Input distribution for A: only the >128K buckets from the competition spec,
# renormalized. A has 26% + 17% = 43% original mass, so we scale those shares.
A_INPUT_DIST = [
    (0.604, 128000, 256000),  # was 0.26 / 0.43
    (0.396, 256000, 512000),  # was 0.17 / 0.43
]

# Input distribution for B: only the <32K buckets (0-4K, 4K-16K, 16K-32K).
# Original mass 0.16 + 0.08 + 0.08 = 0.32.
B_INPUT_DIST = [
    (0.500, 100,    4000),    # 0.16 / 0.32
    (0.250, 4000,   16000),   # 0.08 / 0.32
    (0.250, 16000,  32000),   # 0.08 / 0.32
]

# Output distribution — same as competition spec, used for both A and B
OUTPUT_DIST = [
    (0.58, 10,    512),
    (0.17, 512,   2000),
    (0.06, 2000,  4000),
    (0.09, 4000,  16000),
    (0.10, 16000, 32000),
]


def sample_from_dist(dist):
    r = random.random()
    cumulative = 0.0
    for prob, lo, hi in dist:
        cumulative += prob
        if r <= cumulative:
            return random.randint(lo, hi)
    return dist[-1][2]


def generate_requests(n, input_dist):
    requests = []
    for _ in range(n):
        input_len = sample_from_dist(input_dist)
        output_len = sample_from_dist(OUTPUT_DIST)
        prompt = "x " * input_len
        response = "y " * output_len
        requests.append({
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    return requests


def write(path, requests):
    with open(path, "w") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")
    input_lens = [len(r["conversations"][0]["content"].split()) for r in requests]
    output_lens = [len(r["conversations"][1]["content"].split()) for r in requests]
    print(f"  {path}")
    print(f"    n={len(requests)}, "
          f"input: min={min(input_lens)} median={sorted(input_lens)[len(input_lens)//2]} max={max(input_lens)}, "
          f"total_input={sum(input_lens):,}, total_output={sum(output_lens):,}")


print("Generating stratified benchmark datasets...")
a = generate_requests(32, A_INPUT_DIST)
b = generate_requests(64, B_INPUT_DIST)
write(PATH_A, a)
write(PATH_B, b)
print("\nNext: run bench_serving against both servers with these files.")
