"""Sparse vs Dense FlashInfer benchmark.

Usage:
  1. Start server with sparse:
     SGLANG_SPARSE_PREFILL=1 SGLANG_SPARSE_DECODE=1 python3 -m sglang.launch_server ...
     python3 bench/sparse_vs_dense_bench.py

  2. Restart server WITHOUT sparse:
     SGLANG_SPARSE_PREFILL=0 SGLANG_SPARSE_DECODE=0 python3 -m sglang.launch_server ...
     python3 bench/sparse_vs_dense_bench.py

  Compare the two runs.
"""

import time
import requests
import json
import sys

API_BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:31333"
TARGET_INPUT_TOKENS = 160_000
MAX_OUTPUT_TOKENS = 32768

# --- Build ~160K token input ---
# ~4 chars per token on average
filler_sentence = "The quick brown fox jumps over the lazy dog near the riverbank. "
chars_needed = TARGET_INPUT_TOKENS * 4
repetitions = chars_needed // len(filler_sentence) + 1
filler_text = filler_sentence * repetitions

# Insert a needle in the middle
needle_pos = len(filler_text) // 2
needle = "IMPORTANT FACT: The secret launch code is DELTA-WHISKEY-9. Remember this. "
filler_text = filler_text[:needle_pos] + needle + filler_text[needle_pos:]

prompt = f"""Below is a very long document. Read it carefully and answer the question at the end.

{filler_text}

Question: What is the secret launch code mentioned in the document above? After answering, write a detailed 2000-word essay about space exploration history."""

print(f"📏 Input size: ~{len(prompt):,} chars (~{len(prompt)//4:,} tokens estimate)")
print(f"📝 Max output tokens: {MAX_OUTPUT_TOKENS:,}")
print(f"🔗 API: {API_BASE}")
print()

# --- Send request and measure ---
payload = {
    "model": "MiniCPM-SALA",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": MAX_OUTPUT_TOKENS,
    "min_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.0,
}

print("⏳ Sending request...")
t_start = time.time()

resp = requests.post(
    f"{API_BASE}/v1/chat/completions",
    json=payload,
    timeout=600,
)
resp.raise_for_status()

t_end = time.time()
total_time = t_end - t_start

result = resp.json()
usage = result["usage"]
prompt_tokens = usage["prompt_tokens"]
completion_tokens = usage["completion_tokens"]
content = result["choices"][0]["message"]["content"]

# --- Report ---
print(f"\n{'='*70}")
print(f"📊 BENCHMARK RESULTS")
print(f"{'='*70}")
print(f"Input tokens      : {prompt_tokens:,}")
print(f"Output tokens     : {completion_tokens:,}")
print(f"Total time        : {total_time:.2f}s")
print(f"Overall tokens/s  : {(prompt_tokens + completion_tokens) / total_time:.1f}")
print(f"Decode tokens/s   : ~{completion_tokens / total_time:.1f} (approx, includes prefill)")
print(f"{'='*70}")
print(f"\n📝 Answer preview (first 200 chars):")
# Strip <think> tags for preview
answer = content
if "<think>" in answer and "</think>" in answer:
    answer = answer[answer.index("</think>") + len("</think>"):].strip()
print(answer[:200])
print(f"\n✅ Found needle: {'DELTA-WHISKEY-9' in content}")
