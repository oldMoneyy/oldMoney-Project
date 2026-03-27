#!/usr/bin/env python3
"""
ANALYZE PERF_PUBLIC_SET.JSONL — WITH REAL TOKENIZER + ONE EXAMPLE PER CATEGORY
=============================================================================

Now prints **exactly one representative example** for every task/category.
• Uses the first sample in the task list (sorted by index for consistency)
• Shows index + truncated question (first 800 chars + …)
• All token counts still use your real /opt/model/tokenizer.model
• Style 100% consistent with build_optimal_64.py

Usage:
  python analyze_perf_public_set_with_tokenizer.py
"""

import os
import sys
import json
from collections import defaultdict, Counter
import sentencepiece as spm

# ════════════════════════════════════════════════════════════════════
# PATHS (absolute)
# ════════════════════════════════════════════════════════════════════

EVAL_PATH = "/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"
TOKENIZER_PATH = "/opt/model/tokenizer.model"

TIER_THRESHOLDS = {
    "short":  (0,      40000),
    "medium": (40000,  80000),
    "long":   (80000,  float('inf')),
}

def main():
    print("=" * 80)
    print("  ANALYZING PERF_PUBLIC_SET.JSONL — WITH REAL TOKENIZER + EXAMPLES")
    print("=" * 80)
    print(f"  Dataset   : {EVAL_PATH}")
    print(f"  Tokenizer : {TOKENIZER_PATH}\n")

    # ── Load tokenizer ──
    print("[0/5] Loading tokenizer...")
    if not os.path.exists(TOKENIZER_PATH):
        print(f"  ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        sys.exit(1)
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    print(f"  ✓ Loaded successfully (vocab size: {tokenizer.get_piece_size():,})")

    # ── Load dataset ──
    print("\n[1/5] Loading JSONL...")
    data = []
    by_task = defaultdict(list)
    by_index = {}

    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                    task = item.get("task", "unknown").lower()
                    by_task[task].append(item)
                    idx = item.get("index")
                    if idx is not None:
                        by_index[idx] = item
                except json.JSONDecodeError:
                    print(f"  [WARN] Invalid JSON on line {line_num}")

    total_samples = len(data)
    print(f"  Loaded {total_samples} samples")

    # ── Re-tokenize with real tokenizer ──
    print("\n[2/5] Re-tokenizing every sample...")
    real_token_counts = {}
    for item in data:
        text = item.get("question", "")
        real_tokens = len(tokenizer.encode(text, out_type=int))
        real_token_counts[item.get("index", -1)] = real_tokens

    # ── Task breakdown ──
    print("\n[3/5] TASK BREAKDOWN (REAL tokens)")
    print("─" * 80)
    task_counts = {task: len(items) for task, items in by_task.items()}
    for task, count in sorted(task_counts.items()):
        pct = count / total_samples * 100
        bar = "█" * int(pct / 2)
        print(f"  {task.upper():<12} {count:>4} samples ({pct:5.1f}%) {bar}")

    # ── Per-task detailed statistics + ONE EXAMPLE PER CATEGORY ──
    print("\n[4/5] PER-TASK DETAILED STATISTICS + ONE EXAMPLE")
    print("─" * 80)

    for task in sorted(by_task.keys()):
        items = by_task[task]
        count = len(items)
        
        # Sort by index for consistent example selection
        items = sorted(items, key=lambda x: x.get("index", 0))
        
        # Real token stats
        real_tokens_list = [real_token_counts.get(item.get("index", -1), 0) for item in items]
        char_lens = [len(item.get("question", "")) for item in items]
        
        token_min = min(real_tokens_list) if real_tokens_list else 0
        token_max = max(real_tokens_list) if real_tokens_list else 0
        token_avg = sum(real_tokens_list) / count if count else 0
        
        char_min = min(char_lens) if char_lens else 0
        char_max = max(char_lens) if char_lens else 0
        char_avg = sum(char_lens) / count if count else 0
        
        # Tiers
        tiers = {"short": 0, "medium": 0, "long": 0}
        for t in real_tokens_list:
            if t < TIER_THRESHOLDS["short"][1]:
                tiers["short"] += 1
            elif t < TIER_THRESHOLDS["medium"][1]:
                tiers["medium"] += 1
            else:
                tiers["long"] += 1
        
        indices = [item.get("index") for item in items if item.get("index") is not None]
        idx_min = min(indices) if indices else "N/A"
        idx_max = max(indices) if indices else "N/A"
        
        print(f"\n  {task.upper()} ({count} samples)")
        print(f"    Index range     : {idx_min} – {idx_max}")
        print(f"    Char length     : min={char_min:>8,}  max={char_max:>8,}  avg={char_avg:>8,.0f}")
        print(f"    REAL tokens     : min={token_min:>8,}  max={token_max:>8,}  avg={token_avg:>8,.0f}")
        print(f"    Tier distribution:")
        for tier_name, tier_count in [("short", tiers["short"]), ("medium", tiers["medium"]), ("long", tiers["long"])]:
            tier_pct = tier_count / count * 100 if count else 0
            print(f"      {tier_name:>8}: {tier_count:>3} ({tier_pct:5.1f}%)")
        
        # ── ONE EXAMPLE PER CATEGORY ──
        if items:
            ex = items[0]  # first after sorting by index = most representative low-index sample
            ex_idx = ex.get("index", "N/A")
            ex_text = ex.get("question", "")
            preview = ex_text[:800]
            if len(ex_text) > 800:
                preview += "..."
            print(f"    EXAMPLE (index {ex_idx}) ─────────────────────────────────────────────────────────────")
            print(f"    {preview}")
            print(f"    ────────────────────────────────────────────────────────────────────────────────────")

    # ── Global summary ──
    print("\n[5/5] GLOBAL SUMMARY")
    print("─" * 80)
    all_real_tokens = [real_token_counts.get(item.get("index", -1), 0) for item in data]
    
    print("  Token buckets (REAL):")
    for label, lo, hi in [("<40k short", 0, 40000), ("40k-80k medium", 40000, 80000), (">80k long", 80000, float('inf'))]:
        cnt = sum(1 for t in all_real_tokens if lo <= t < hi)
        bar = "█" * (cnt // 4)
        print(f"    {label:>18}: {cnt:>4} {bar}")

    print("\nSUMMARY TABLE (with examples ready)")
    print("┌────────────┬──────┬──────────────┬──────────────┐")
    print("│ Task       │ Count│ Avg REAL Tok │ Index Range  │")
    print("├────────────┼──────┼──────────────┼──────────────┤")
    for task in sorted(task_counts.keys()):
        items = by_task[task]
        avg_tok = sum(real_token_counts.get(item.get("index", -1), 0) for item in items) / len(items)
        indices = [item.get("index") for item in items if item.get("index") is not None]
        idx_range = f"{min(indices)}–{max(indices)}" if indices else "N/A"
        print(f"│ {task.upper():<10} │ {len(items):>4} │ {avg_tok:>12,.0f} │ {idx_range:<12} │")
    print("└────────────┴──────┴──────────────┴──────────────┘")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE — ONE EXAMPLE PRINTED PER CATEGORY")
    print("=" * 80)
    print("  All examples are taken from your real dataset.")
    print("  Run again anytime the JSONL or tokenizer changes.")


if __name__ == "__main__":
    main()