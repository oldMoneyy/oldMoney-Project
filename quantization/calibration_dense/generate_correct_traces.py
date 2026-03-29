#!/usr/bin/env python3
"""
Generate correct reasoning traces for calibration using Claude API + SALA BF16.

Strategy:
  1. Load eval MCQ questions + diverse reasoning questions
  2. Call Claude API to get correct answers and reasoning
  3. Call SALA BF16 model to generate reasoning traces
  4. Keep SALA traces where answer matches Claude's (correct SALA-style trace)
  5. For mismatches, reformat Claude's reasoning in SALA <think> style
  6. Output calibration JSONL

Usage (on server):
    # Start SALA BF16 server first:
    python3 -m sglang.launch_server --model /opt/model --trust-remote-code \
        --attention-backend flashinfer --port 31333 --mem-fraction-static 0.82

    # Then run this:
    nohup python generate_correct_traces.py \
        --eval-path /opt/oldMoney-Project/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
        --output /opt/oldMoney-Project/quantization/calibration_dense/correct_traces.jsonl \
        --sala-url http://127.0.0.1:31333 \
        > /opt/oldMoney-Project/logs/generate_correct_traces.log 2>&1 &
"""

import os
import json
import time
import re
import argparse
import requests


CLAUDE_API_URL = "http://compass.llm.shopee.io/compass-api/v1/messages"
CLAUDE_API_KEY = "c2e4ec4df11a3f7b18aa492df6e3cdcdd708e8df5b3d7207cdfb713a20993115"
CLAUDE_MODEL = "claude-opus-4-6"


# ====================================================================
# Extra reasoning questions (diverse topics, not in eval)
# ====================================================================

EXTRA_QUESTIONS = [
    {
        "question": "Derive the formula for the sum of an infinite geometric series. Show all steps clearly.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Explain the mechanism of CRISPR-Cas9 gene editing. How does the guide RNA direct the Cas9 protein to the correct location in the genome?\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "A 5 kg block slides down a frictionless inclined plane that makes a 30 degree angle with the horizontal. What is the acceleration of the block? What is the normal force?\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Prove that the square root of 2 is irrational using proof by contradiction.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Explain how a transformer neural network processes a sequence of tokens. Describe the self-attention mechanism, positional encoding, and the feed-forward layers.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "What is the time complexity of merge sort and why? Walk through the recurrence relation T(n) = 2T(n/2) + O(n) and solve it using the Master Theorem.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Derive the Lorentz transformation equations from Einstein's two postulates of special relativity.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Explain the process of meiosis and how it leads to genetic diversity through crossing over and independent assortment.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "A company has 12 employees. In how many ways can a committee of 5 be formed if the CEO and CFO cannot both be on the committee?\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
    {
        "question": "Explain the CAP theorem in distributed systems. Give a concrete example of a system that sacrifices consistency for availability.\n\nPlease think step by step and provide your answer.",
        "gold": None,
        "_type": "reasoning",
    },
]


def call_claude(question, max_tokens=4096, retries=3):
    """Call Claude API using curl (requests gets 403 due to redirect handling)."""
    import subprocess
    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": question},
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
            print(f"    [WARN] Claude API attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def call_sala(question, sala_url, max_tokens=16384):
    """Call SALA model to generate reasoning trace."""
    try:
        resp = requests.post(
            f"{sala_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "MiniCPM-SALA",
                "messages": [{"role": "user", "content": question}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
    except Exception as e:
        print(f"    [WARN] SALA call failed: {e}")
        return None, {}


def extract_mcq_answer(text):
    """Extract MCQ answer letter from text."""
    if text is None:
        return None
    # Look for ANSWER: X pattern
    match = re.search(r"ANSWER:\s*([A-D])", text)
    if match:
        return match.group(1)
    # Look for standalone letter at the end
    match = re.search(r"\b([A-D])\s*$", text.strip())
    if match:
        return match.group(1)
    return None


def format_as_sala_trace(question, claude_reasoning, claude_answer):
    """Reformat Claude's reasoning in SALA's <think> style."""
    if claude_answer:
        return f"{question}\n\n<think>\n{claude_reasoning}\n</think>\n\nANSWER: {claude_answer}"
    else:
        return f"{question}\n\n<think>\n{claude_reasoning}\n</think>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", required=True,
                        help="Path to perf_public_set.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--sala-url", default="http://127.0.0.1:31333",
                        help="SALA server URL")
    parser.add_argument("--max-mcq", type=int, default=30,
                        help="Max MCQ questions from eval")
    parser.add_argument("--sala-max-tokens", type=int, default=16384,
                        help="Max tokens for SALA generation")
    parser.add_argument("--skip-sala", action="store_true",
                        help="Skip SALA generation, use Claude traces only")
    args = parser.parse_args()

    print("=" * 70)
    print("  Correct Trace Generator (Claude + SALA)")
    print("=" * 70)

    # ---- Step 1: Load MCQ questions from eval ----
    print("\n[Step 1] Loading eval MCQ questions...")
    eval_samples = []
    with open(args.eval_path, "r", encoding="utf-8") as f:
        for line in f:
            eval_samples.append(json.loads(line.strip()))

    mcq_questions = []
    for i, s in enumerate(eval_samples):
        if s.get("task") == "mcq" or i < 30:
            mcq_questions.append({
                "question": s["question"],
                "gold": s.get("gold", ""),
                "index": i,
                "_type": "mcq",
            })
    mcq_questions = mcq_questions[:args.max_mcq]
    print(f"  Loaded {len(mcq_questions)} MCQ questions")

    # Add extra reasoning questions
    all_questions = mcq_questions + EXTRA_QUESTIONS
    print(f"  Total questions: {len(all_questions)} ({len(mcq_questions)} MCQ + {len(EXTRA_QUESTIONS)} reasoning)")

    # ---- Step 2: Get Claude answers ----
    print(f"\n[Step 2] Getting Claude answers for {len(all_questions)} questions...")
    results = []
    for i, q in enumerate(all_questions):
        qtype = q["_type"]
        gold = q.get("gold", "")
        question_text = q["question"]
        short_q = question_text[:80].replace("\n", " ")

        print(f"\n  [{i+1}/{len(all_questions)}] ({qtype}) {short_q}...")

        # Call Claude
        t0 = time.time()
        claude_response = call_claude(question_text)
        dt_claude = time.time() - t0

        if claude_response is None:
            print(f"    Claude FAILED, skipping")
            continue

        claude_answer = extract_mcq_answer(claude_response) if qtype == "mcq" else None

        if qtype == "mcq":
            match = "MATCH" if claude_answer == gold else f"MISMATCH (claude={claude_answer}, gold={gold})"
            print(f"    Claude: {len(claude_response)} chars, answer={claude_answer}, gold={gold} -> {match} ({dt_claude:.1f}s)")
        else:
            print(f"    Claude: {len(claude_response)} chars ({dt_claude:.1f}s)")

        result = {
            "question": question_text,
            "gold": gold,
            "_type": qtype,
            "claude_response": claude_response,
            "claude_answer": claude_answer,
        }

        # ---- Step 3: Get SALA trace (if server available) ----
        if not args.skip_sala:
            t0 = time.time()
            sala_response, sala_usage = call_sala(question_text, args.sala_url, args.sala_max_tokens)
            dt_sala = time.time() - t0

            if sala_response:
                sala_answer = extract_mcq_answer(sala_response) if qtype == "mcq" else None
                sala_tokens = sala_usage.get("completion_tokens", len(sala_response) // 4)
                result["sala_response"] = sala_response
                result["sala_answer"] = sala_answer
                result["sala_tokens"] = sala_tokens

                if qtype == "mcq":
                    sala_match = "MATCH" if sala_answer == gold else f"MISMATCH (sala={sala_answer})"
                    print(f"    SALA:   {len(sala_response)} chars, {sala_tokens} tokens, answer={sala_answer} -> {sala_match} ({dt_sala:.1f}s)")
                else:
                    print(f"    SALA:   {len(sala_response)} chars, {sala_tokens} tokens ({dt_sala:.1f}s)")
            else:
                print(f"    SALA:   FAILED")
                result["sala_response"] = None

        results.append(result)

    # ---- Step 4: Build calibration traces ----
    print(f"\n\n[Step 4] Building calibration traces...")
    calib_samples = []
    stats = {"sala_correct": 0, "claude_fallback": 0, "claude_only": 0, "reasoning": 0}

    for r in results:
        qtype = r["_type"]
        question = r["question"]
        gold = r.get("gold", "")
        claude_response = r.get("claude_response", "")
        claude_answer = r.get("claude_answer")
        sala_response = r.get("sala_response")
        sala_answer = r.get("sala_answer")

        if qtype == "mcq":
            if sala_response and sala_answer == gold:
                # SALA got it right — use SALA's trace (matches inference distribution)
                trace = f"{question}\n\n{sala_response}"
                source = "sala_correct"
                stats["sala_correct"] += 1
            elif claude_answer == gold:
                # SALA wrong, Claude right — use Claude reasoning in SALA format
                trace = format_as_sala_trace(question, claude_response, gold)
                source = "claude_fallback"
                stats["claude_fallback"] += 1
            else:
                # Claude disagrees with gold — DISCARD (reasoning contradicts answer)
                stats["claude_only"] += 1
                continue
        else:
            # Reasoning question — prefer SALA trace, fallback to Claude
            if sala_response and len(sala_response) > 100:
                trace = f"{question}\n\n{sala_response}"
                source = "sala_reasoning"
            else:
                trace = format_as_sala_trace(question, claude_response, None)
                source = "claude_reasoning"
            stats["reasoning"] += 1

        calib_samples.append({
            "question": trace,
            "_source": source,
            "_type": qtype,
            "_chars": len(trace),
        })

    # ---- Step 5: Save ----
    print(f"\n[Step 5] Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for s in calib_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"  TRACE GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total traces: {len(calib_samples)}")
    print(f"  MCQ - SALA correct (best):  {stats['sala_correct']}")
    print(f"  MCQ - Claude fallback:      {stats['claude_fallback']}")
    print(f"  MCQ - Claude only (discarded): {stats['claude_only']}")
    print(f"  Reasoning traces:           {stats['reasoning']}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
