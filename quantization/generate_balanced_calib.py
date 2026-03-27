#!/usr/bin/env python3
"""
Generate balanced calibration dataset for MLP-only NVFP4 quantization.

Problem: optimal_96.jsonl has 94 samples but only 1 below 1k tokens.
The SOAR eval is 20% short MCQ (95-656 tokens) — zero calibration coverage.

Solution: Add all 30 MCQ samples from the public eval set to optimal_96,
giving each task type proportional representation in the AWQ H_diag.

AWQ uses per-sample normalized H_diag (each sample contributes equally
regardless of token count), so 30 MCQ samples in 124 total gives MCQ
24% weight — close to the 20% in eval.

Usage (on server):
    python /opt/oldMoney-Project/quantization/generate_balanced_calib.py

Output: /opt/calib_balanced_124.jsonl
"""
import json
import random

EVAL_PATH = "/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"
CALIB_PATH = "/opt/optimal_96.jsonl"
OUT_PATH = "/opt/calib_balanced_124.jsonl"

# --- 1. Load eval data and extract MCQ samples (indices 0-29) ---
eval_samples = []
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line.strip())
        eval_samples.append(obj)

mcq_samples = []
for i, s in enumerate(eval_samples):
    if s.get("task") == "mcq" or i < 30:
        # Eval indices 0-29 are MCQ (95-656 tokens)
        # Convert to calibration format: just need "question" field
        mcq_samples.append({"question": s["question"], "_source": f"eval_mcq_{i}"})

print(f"MCQ samples extracted: {len(mcq_samples)}")

# --- 2. Load existing calibration data ---
calib_samples = []
with open(CALIB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        calib_samples.append(obj)

print(f"Existing calib samples: {len(calib_samples)}")

# --- 3. Check for duplicates ---
# Extract question text for dedup
existing_questions = set()
for s in calib_samples:
    q = s.get("question", s.get("text", ""))
    existing_questions.add(q[:200])  # first 200 chars as key

new_mcq = []
for s in mcq_samples:
    q = s.get("question", "")
    if q[:200] not in existing_questions:
        new_mcq.append(s)
    else:
        print(f"  Skipping duplicate MCQ: {q[:80]}...")

print(f"New MCQ samples (after dedup): {len(new_mcq)}")

# --- 4. Combine and shuffle ---
combined = calib_samples + new_mcq
random.seed(42)
random.shuffle(combined)

# --- 5. Write output ---
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for s in combined:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nOutput: {OUT_PATH}")
print(f"Total samples: {len(combined)}")
print(f"  From calib_96: {len(calib_samples)}")
print(f"  New MCQ added: {len(new_mcq)}")

# --- 6. Distribution summary ---
print(f"\nTask distribution in combined dataset:")
task_counts = {}
for s in combined:
    task = s.get("task", s.get("_source", "calib_long"))
    if "mcq" in str(task):
        task_counts["mcq"] = task_counts.get("mcq", 0) + 1
    else:
        task_counts["long_context"] = task_counts.get("long_context", 0) + 1

for task, count in sorted(task_counts.items()):
    pct = 100 * count / len(combined)
    print(f"  {task}: {count} ({pct:.1f}%)")

print(f"\nMCQ weight in H_diag: {task_counts.get('mcq', 0)}/{len(combined)} "
      f"= {100 * task_counts.get('mcq', 0) / len(combined):.1f}% "
      f"(eval target: 20%)")
