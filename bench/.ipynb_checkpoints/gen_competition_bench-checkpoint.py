#!/usr/bin/env python3
"""Generate benchmark datasets matching SOAR 2026 competition distribution.
生成两个文件：64条 + 32条，来自同一随机序列（分布完全一致），64条文件和原来脚本一模一样。
"""
import json, random

random.seed(42)

# ================== 配置 ==================
NUM_64 = 64
NUM_32 = 32
PATH_64 = "/opt/oldMoney-Project/bench/competition_bench_64.jsonl"
PATH_32 = "/opt/oldMoney-Project/bench/competition_bench_32.jsonl"   # 我帮你补全了 /opt/

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
    cumulative = 0.0
    for prob, lo, hi in dist:
        cumulative += prob
        if r <= cumulative:
            return random.randint(lo, hi)
    return dist[-1][2]

def generate_requests(n):
    requests = []
    for _ in range(n):
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
    return requests

# ================== 生成并切分 ==================
# 一次生成96条（64+32），保证两个文件来自完全相同的随机序列
all_requests = generate_requests(NUM_64 + NUM_32)

requests_64 = all_requests[:NUM_64]
requests_32 = all_requests[NUM_64:]

# 写入64条（和原来脚本完全一致）
with open(PATH_64, "w") as f:
    for r in requests_64:
        f.write(json.dumps(r) + "\n")

# 写入32条
with open(PATH_32, "w") as f:
    for r in requests_32:
        f.write(json.dumps(r) + "\n")

# ================== 打印结果 ==================
print(f"✅ 生成完成！")
print(f"   {NUM_64} 条 → {PATH_64}")
print(f"   {NUM_32} 条 → {PATH_32}")
print(f"   两个文件分布完全一致（来自同一随机种子）")

# 额外统计（方便你检查）
input_lens = [sample_from_dist(INPUT_DIST) for _ in range(200)]
output_lens = [sample_from_dist(OUTPUT_DIST) for _ in range(200)]
print(f"\n分布统计（采样200次）:")
print(f"   Input tokens:  min={min(input_lens)}, median={sorted(input_lens)[100]}, max={max(input_lens)}")
print(f"   Output tokens: min={min(output_lens)}, median={sorted(output_lens)[100]}, max={max(output_lens)}")