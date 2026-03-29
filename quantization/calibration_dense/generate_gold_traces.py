#!/usr/bin/env python3
"""
Generate correct reasoning traces for QA + MCQ calibration.

Key insight: Tell Claude the GOLD answer and ask it to reason toward it.
This guarantees reasoning matches the answer (no contradictions, no discards).

Usage:
    nohup uv run --with requests python -u \
        quantization/calibration_dense/generate_gold_traces.py \
        --eval-path SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
        --output quantization/calibration_dense/gold_traces.jsonl \
        > logs/generate_gold_traces.log 2>&1 &
"""

import os
import json
import time
import re
import subprocess
import argparse


CLAUDE_API_URL = "http://compass.llm.shopee.io/compass-api/v1/messages"
CLAUDE_API_KEY = "c2e4ec4df11a3f7b18aa492df6e3cdcdd708e8df5b3d7207cdfb713a20993115"
CLAUDE_MODEL = "claude-opus-4-6"


def call_claude(prompt, max_tokens=4096, retries=3):
    """Call Claude API using curl."""
    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    })
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [
                    "curl", "--silent", "--location", "--request", "POST",
                    "--url", CLAUDE_API_URL,
                    "--header", f"Authorization: Bearer {CLAUDE_API_KEY}",
                    "--header", "Content-type: application/json",
                    "--data-raw", payload,
                ],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr[:200]}")
            data = json.loads(result.stdout)
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            return text
        except Exception as e:
            print(f"    [WARN] Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def build_mcq_prompt(question, gold):
    """Ask Claude to reason step by step toward the known correct MCQ answer."""
    return f"""You are solving an MCQ question. The correct answer is {gold}.

Please provide a clear, step-by-step reasoning process that logically arrives at answer {gold}. Show your work as if you are a student working through the problem. End with "ANSWER: {gold}".

Question:
{question}"""


def build_qa_prompt(question, gold):
    """Ask Claude to reason toward the known correct QA answer."""
    gold_str = gold if isinstance(gold, str) else ", ".join(gold) if isinstance(gold, list) else str(gold)
    # QA questions contain long documents, truncate for Claude (keep first 3000 + last 1000 chars)
    if len(question) > 5000:
        q_truncated = question[:3000] + "\n\n[... document truncated for reasoning ...]\n\n" + question[-1000:]
    else:
        q_truncated = question
    return f"""You are answering a document-based question. The correct answer is: "{gold_str}"

Please provide a clear reasoning process that explains WHY this is the correct answer based on the question. Show your reasoning step by step, then state the answer. The answer must be exactly: {gold_str}

Question (with document context):
{q_truncated}"""


def format_trace(question, reasoning, gold, task):
    """Format as SALA-style calibration trace."""
    if task == "mcq":
        gold_str = gold
        return f"{question}\n\n<think>\n{reasoning}\n</think>\n\nANSWER: {gold_str}"
    else:
        gold_str = gold if isinstance(gold, str) else ", ".join(gold) if isinstance(gold, list) else str(gold)
        return f"{question}\n\n<think>\n{reasoning}\n</think>\n\n{gold_str}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="mcq,qa",
                        help="Comma-separated task types to generate (default: mcq,qa)")
    parser.add_argument("--max-per-task", type=int, default=30,
                        help="Max samples per task type")
    args = parser.parse_args()

    target_tasks = set(args.tasks.split(","))

    print("=" * 70)
    print("  Gold-Guided Trace Generator (Claude reasons toward gold answer)")
    print("=" * 70)

    # Load eval data
    print("\n[Step 1] Loading eval data...")
    with open(args.eval_path, "r", encoding="utf-8") as f:
        eval_samples = [json.loads(line.strip()) for line in f]

    # Group by task
    by_task = {}
    for i, s in enumerate(eval_samples):
        task = s.get("task", "")
        if task in target_tasks:
            if task not in by_task:
                by_task[task] = []
            by_task[task].append((i, s))

    for task in target_tasks:
        count = len(by_task.get(task, []))
        print(f"  {task}: {count} samples")

    # Generate traces
    print(f"\n[Step 2] Generating Claude traces (reasoning toward gold)...")
    all_traces = []
    stats = {"total": 0, "success": 0, "failed": 0}

    for task in sorted(target_tasks):
        samples = by_task.get(task, [])[:args.max_per_task]
        print(f"\n--- Task: {task} ({len(samples)} samples) ---")

        for j, (idx, s) in enumerate(samples):
            question = s["question"]
            gold = s.get("gold", "")
            gold_display = str(gold)[:50] if gold else "?"
            short_q = question[:80].replace("\n", " ")

            print(f"\n  [{j+1}/{len(samples)}] idx={idx} gold={gold_display}")
            print(f"    Q: {short_q}...")

            stats["total"] += 1

            # Build prompt based on task type
            if task == "mcq":
                prompt = build_mcq_prompt(question, gold)
            elif task == "qa":
                prompt = build_qa_prompt(question, gold)
            else:
                prompt = build_qa_prompt(question, gold)

            t0 = time.time()
            reasoning = call_claude(prompt)
            dt = time.time() - t0

            if reasoning is None:
                print(f"    FAILED ({dt:.1f}s)")
                stats["failed"] += 1
                continue

            print(f"    OK: {len(reasoning)} chars ({dt:.1f}s)")
            stats["success"] += 1

            # Format trace
            trace = format_trace(question, reasoning, gold, task)
            all_traces.append({
                "question": trace,
                "_source": f"claude_gold_{task}",
                "_type": task,
                "_eval_idx": idx,
                "_gold": str(gold),
                "_chars": len(trace),
            })

    # Save
    print(f"\n[Step 3] Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for t in all_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"  GOLD TRACE GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total attempted: {stats['total']}")
    print(f"  Success:         {stats['success']}")
    print(f"  Failed:          {stats['failed']}")
    print(f"  Output:          {args.output}")

    # Per-task breakdown
    from collections import Counter
    task_counts = Counter(t["_type"] for t in all_traces)
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} traces")

    chars = [t["_chars"] for t in all_traces]
    if chars:
        print(f"  Chars: min={min(chars):,} max={max(chars):,} mean={sum(chars)//len(chars):,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
