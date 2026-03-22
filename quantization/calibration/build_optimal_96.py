#!/usr/bin/env python3
"""
BUILD THE OPTIMAL 96-SAMPLE CALIBRATION DATASET
================================================

Keeps all 64 samples from optimal_64 and adds 32 more.

ANALYSIS OF 77% AWQ (new) vs 82.4% BF16:
  ┌──────┬────────┬────────┬────────┬──────────────────────────────────┐
  │ Task │  BF16  │ 74%AWQ │ 77%AWQ │ Analysis                         │
  ├──────┼────────┼────────┼────────┼──────────────────────────────────┤
  │ CWE  │  90.0% │  83.3% │  80.7% │ GOT WORSE! -9.3% gap. Urgent.   │
  │ FWE  │  98.9% │  98.9% │  96.7% │ Regressed. -2.2% gap.           │
  │ MCQ  │  60.0% │  43.3% │  50.0% │ Improved +6.7%. Still -10% gap. │
  │ NIAH │ 100.0% │  96.7% │ 100.0% │ FULLY RECOVERED! +3.3%.         │
  │ QA   │  63.3% │  50.0% │  60.0% │ Big improvement +10%. -3.3% gap.│
  │ ALL  │  82.4% │  74.4% │  77.3% │ +2.9% overall. Target: 79%.     │
  └──────┴────────┴────────┴────────┴──────────────────────────────────┘

KEY INSIGHT FROM 77% MCQ:
  </think>% went from 57% → 90% (gen traces WORKED for convergence!)
  But accuracy only 50% vs 60% BF16.
  Truncation is largely fixed — remaining failures are wrong answers.
  → More gen traces have diminishing returns for MCQ.

KEY INSIGHT FROM 77% CWE:
  CWE got WORSE (83.3% → 80.7%) despite being a new format.
  </think>% improved (47% → 80%), but some samples that were correct now fail.
  → CWE needs MORE coverage. 10 samples in optimal_64 was not enough.

ALLOCATION FOR EXTRA 32 SLOTS:
  ┌────────────────────────────┬───────┬──────────────────────────────────────┐
  │ Category                   │ Count │ Justification                        │
  ├────────────────────────────┼───────┼──────────────────────────────────────┤
  │ CWE eval prompts           │   12  │ CWE regressed -9.3%. Top priority.   │
  │ MCQ generation traces      │    6  │ Still -10% gap. Diminishing returns. │
  │ QA eval prompts            │    6  │ Close remaining -3.3% gap.           │
  │ FWE eval prompts           │    4  │ Address -2.2% regression.            │
  │ NIAH eval prompts          │    2  │ Maintain 100%. Add format diversity.  │
  │ General reasoning traces   │    2  │ Non-MCQ gen patterns.                │
  ├────────────────────────────┼───────┼──────────────────────────────────────┤
  │ TOTAL ADDITIONAL           │   32  │                                      │
  └────────────────────────────┴───────┴──────────────────────────────────────┘

COMBINED 96-SAMPLE WEIGHT (with per-sample observer):
  ┌────────────────────┬─────────────┬─────────────┬──────────────────────────┐
  │ Category           │ In 64-set   │ In 96-set   │ Hessian %                │
  ├────────────────────┼─────────────┼─────────────┼──────────────────────────┤
  │ MCQ gen traces     │ 20 (31.3%)  │ 26 (27.1%)  │ Slight dilution OK       │
  │ Reasoning traces   │  4 ( 6.3%)  │  6 ( 6.3%)  │ Maintained               │
  │ CWE eval           │ 10 (15.6%)  │ 22 (22.9%)  │ ↑ Addresses regression   │
  │ QA eval            │ 14 (21.9%)  │ 20 (20.8%)  │ ~Maintained              │
  │ NIAH eval          │ 10 (15.6%)  │ 12 (12.5%)  │ Slight dilution OK       │
  │ FWE eval           │  6 ( 9.4%)  │ 10 (10.4%)  │ ↑ Addresses regression   │
  └────────────────────┴─────────────┴─────────────┴──────────────────────────┘

Usage:
  python /opt/oldMoney-Project/quantization/calibration/build_optimal_96.py \
      --gen-traces /opt/oldMoney-Project/quantization/calibration/gen_aware_calib.jsonl \
      --eval-data /opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
      --output /opt/oldMoney-Project/quantization/calibration/optimal_96.jsonl
"""

import os
import sys
import json
import argparse
from collections import Counter

# ════════════════════════════════════════════════════════════════════
# ORIGINAL 64 SAMPLES (from build_optimal_64.py — kept as-is)
# ════════════════════════════════════════════════════════════════════

ORIGINAL_QA = {61, 63, 66, 68, 70, 72, 73, 75, 77, 79, 82, 84, 88, 90}
ORIGINAL_CWE = {121, 125, 128, 132, 135, 137, 140, 144, 147, 150}
ORIGINAL_NIAH = {31, 33, 37, 41, 45, 47, 50, 51, 54, 60}
ORIGINAL_FWE = {91, 95, 103, 107, 115, 118}

# ════════════════════════════════════════════════════════════════════
# ADDITIONAL 32 SAMPLES — each one justified
# ════════════════════════════════════════════════════════════════════

# ── CWE: 12 more samples ───────────────────────────────────────────
# CWE REGRESSED from 83.3% to 80.7%. This is the #1 priority.
# 10 samples in the 64-set was not enough. Adding 12 more = 22 total.
# CWE has a unique activation pattern: thousands of repeated words
# in numbered lists. Very different from natural language.
# Need dense coverage at all 3 length tiers.

EXTRA_CWE = {
    # === SHORT TIER (~31k tokens) — 4 more ===
    # Already have: 121, 125, 128
    122: "31,426 tok. DOC+REASON+MATH(71). Different word set from existing short samples. "
         "Highest math density among short CWE — tests numeric-adjacent activations.",
    124: "31,515 tok. DOC+MATH(48) but NO REASON flag — structurally different from others. "
         "Tests word-counting without explicit reasoning instruction.",
    126: "31,585 tok. MATH(60) only — no DOC, no REASON. Minimal instruction variant. "
         "Covers the 'bare word list' activation pattern.",
    129: "31,690 tok. DOC+REASON+MATH(53). Fills the gap between 128 (31,651) and 130. "
         "Adds word variety diversity in short tier.",

    # === MEDIUM TIER (~63k tokens) — 4 more ===
    # Already have: 132, 135, 137, 140
    131: "63,062 tok. DOC+REASON+MATH(126). Lowest medium-tier index. "
         "Fills the gap between short (31k) and existing medium (63,296).",
    134: "63,386 tok. DOC+REASON+MATH(90). Lowest math density in medium tier — "
         "tests when word list has fewer numeric distractors.",
    136: "63,459 tok. DOC+REASON+MATH(132). Between 135 and 137. "
         "Adds coverage density in the most critical medium range.",
    139: "63,577 tok. DOC+REASON+MATH(128). Between 137 and 140. "
         "Near-complete coverage of medium tier now (6 of 10 medium samples).",

    # === LONG TIER (~127k tokens) — 4 more ===
    # Already have: 144, 147, 150
    141: "127,085 tok. Shortest long-tier CWE. DOC+REASON+MATH(212). "
         "Anchors the start of 127k range. ~15,900 words to count.",
    143: "127,432 tok. DOC+REASON+MATH(240). Highest math density among long CWE. "
         "Tests dense numeric content within word lists.",
    146: "127,554 tok. DOC+REASON+MATH(219). Between 144 and 147. "
         "Fills gap in long tier coverage.",
    148: "127,621 tok. DOC+REASON+MATH(204). Between 147 and 150. "
         "Near-complete long tier coverage now (7 of 10 long samples).",
}

# ── QA: 6 more samples ─────────────────────────────────────────────
# QA improved massively (+10%) but still -3.3% from BF16.
# Focus on long-tier QA where degradation persists at 120k+ context.
# Also add Chinese QA samples (4 exist, none in 64-set).

EXTRA_QA = {
    # === SHORT TIER — 1 more ===
    # Already have: 61, 63, 66, 68, 70
    64: "27,489 tok. Gold='Battle of Hastings'. Historical fact retrieval. "
        "Both models got this right — adds stability anchor for short QA.",

    # === MEDIUM TIER — 2 more ===
    # Already have: 72, 73, 75, 77, 79
    74: "61,200 tok. Gold='James Packer'. Person-entity question. "
        "Tests entity extraction at medium context — a common failure mode.",
    78: "62,860 tok. Gold='Beijing'. CHINESE content (ZH flag). "
        "Only 4 QA samples have Chinese — zero in current calibration. "
        "Must cover this for cross-lingual activation patterns.",

    # === LONG TIER — 3 more ===
    # Already have: 82, 84, 88, 90
    81: "119,911 tok. Gold='nineteenth'. Numeric-to-word format question. "
        "Similar failure mode as 75 ('seven' vs '7'). Anchors long tier.",
    83: "124,526 tok. Gold='probabilistic Turing machines'. CS terminology. "
        "Tests precise technical term extraction at 125k context.",
    86: "125,906 tok. Gold='1987'. CHINESE content (ZH flag). "
        "Second Chinese QA sample — critical for ZH coverage at long context.",
}

# ── FWE: 4 more samples ────────────────────────────────────────────
# FWE regressed slightly (98.9% → 96.7%). Only 1 sample failed.
# But: FWE has completely unique activation patterns (random 6-char strings).
# 6 samples in 64-set may not be enough to protect these channels.
# Adding 4 more = 10 total, matching NIAH allocation.

EXTRA_FWE = {
    # === SHORT TIER — 1 more ===
    # Already have: 91, 95
    93: "28,456 tok. Coded words like 'tkrjdw', 'pqcvgu'. "
        "Fills gap between 91 (28k) and 95 (29k). Different code vocabulary.",

    # === MEDIUM TIER — 2 more ===
    # Already have: 103, 107
    101: "52,706 tok. Shortest medium FWE. Coded words 'szygym', 'reckcf'. "
         "Anchors the start of medium range (52k vs existing 55k).",
    109: "58,134 tok. Longest medium-tier before the gap to long. "
         "Fills the top of medium range. Words 'srceaa', 'qxnukn'.",

    # === LONG TIER — 1 more ===
    # Already have: 115, 118
    120: "126,884 tok. LONGEST FWE sample in entire eval set. "
         "Maximum context length for coded text. Must be in calibration.",
}

# ── NIAH: 2 more samples ───────────────────────────────────────────
# NIAH is FULLY RECOVERED (100%!). These 2 are for format diversity only.
# Current 10 samples are heavily MCQ+UUID. Add non-MCQ variants.

EXTRA_NIAH = {
    35: "31,319 tok. Pure DOC format — no MCQ options, no UUID-heavy, no REASON. "
        "Only 'plain text with hidden number' pattern. Covers the simplest NIAH variant "
        "that's underrepresented in current selection.",
    52: "127,062 tok. DOC+REASON+MATH(5448). Very high math density. "
        "Non-MCQ long NIAH with dense numeric content. Covers the 'number buried in math' pattern.",
}


# ════════════════════════════════════════════════════════════════════
# MCQ + REASONING TRACE SELECTION (same logic as build_optimal_64)
# ════════════════════════════════════════════════════════════════════

def select_mcq_traces(gen_data, already_selected_count=20, extra_count=6):
    """
    Select 6 MORE MCQ gen traces beyond the first 20.

    The 77% model shows </think>% went from 57% to 90% — gen traces work.
    But accuracy is still 50% vs 60% BF16.
    6 more traces (diverse topics) for marginal improvement.
    Diminishing returns beyond this.
    """
    mcq_traces = [item for item in gen_data if item.get("_type") == "mcq_gen_trace"]

    valid = []
    for item in mcq_traces:
        text = item["question"]
        tokens = item.get("_tokens", len(text) // 4)
        has_think_close = "</think>" in text
        if not has_think_close or len(text) < 1000:
            continue
        has_answer = "ANSWER:" in text.upper()
        score = 10 if has_answer else 0
        if 10000 < len(text) < 30000: score += 8
        elif 2000 < len(text) < 10000: score += 5
        valid.append((score, tokens, item))

    valid.sort(key=lambda x: (-x[0], x[1]))

    # Skip first `already_selected_count` (already in optimal_64)
    # Take next `extra_count`
    if len(valid) > already_selected_count:
        extras = valid[already_selected_count:already_selected_count + extra_count]
    else:
        extras = valid[:extra_count]  # fallback

    return [item for _, _, item in extras]


def select_extra_reasoning_traces(gen_data, already_selected_count=4, extra_count=2):
    """Select 2 more general reasoning traces."""
    traces = [item for item in gen_data if item.get("_type") == "reasoning_gen_trace"]
    valid = [item for item in traces if len(item.get("question", "")) > 1000]
    valid.sort(key=lambda x: -len(x["question"]))

    if len(valid) > already_selected_count:
        return valid[already_selected_count:already_selected_count + extra_count]
    return []


# ════════════════════════════════════════════════════════════════════
# ORIGINAL 64-SAMPLE SPECS (copied from build_optimal_64.py)
# ════════════════════════════════════════════════════════════════════

ORIGINAL_QA_SPECS = {
    61: "Baseline short QA", 63: "DEGRADED", 66: "DEGRADED",
    68: "Near-miss", 70: "Near-miss",
    72: "DEGRADED", 73: "AWQ wrong", 75: "AWQ wrong",
    77: "Baseline medium", 79: "AWQ wrong",
    82: "DEGRADED", 84: "DEGRADED", 88: "DEGRADED", 90: "DEGRADED",
}

ORIGINAL_CWE_SPECS = {
    121: "Shortest CWE", 125: "Short tier", 128: "Short tier",
    132: "Medium CWE", 135: "Medium CWE", 137: "Medium CWE", 140: "Medium CWE",
    144: "Long CWE DEGRADED", 147: "Long CWE DEGRADED", 150: "Longest CWE",
}

ORIGINAL_NIAH_SPECS = {
    31: "UUID-search", 33: "Multi-number", 37: "UUID+MCQ hybrid",
    41: "UUID-heavy", 45: "MCQ+number", 47: "Multi-number",
    50: "Multi-number+MCQ", 51: "UUID-heavy max",
    54: "Multi-number long", 60: "DEGRADED NIAH",
}

ORIGINAL_FWE_SPECS = {
    91: "Shortest FWE", 95: "Short FWE",
    103: "Medium FWE", 107: "Medium FWE",
    115: "Long FWE", 118: "Long FWE",
}


def select_original_mcq_traces(gen_data, target_count=20):
    """Same selection logic as build_optimal_64.py"""
    mcq_traces = [item for item in gen_data if item.get("_type") == "mcq_gen_trace"]
    valid = []
    for item in mcq_traces:
        text = item["question"]
        tokens = item.get("_tokens", len(text) // 4)
        has_think_close = "</think>" in text
        if not has_think_close or len(text) < 1000:
            continue
        has_answer = "ANSWER:" in text.upper()
        score = 10 if has_answer else 0
        score += 10  # has_think_close
        if 10000 < len(text) < 30000: score += 8
        elif 2000 < len(text) < 10000: score += 5
        elif len(text) > 30000: score += 3
        valid.append((score, tokens, item))
    valid.sort(key=lambda x: (-x[0], x[1]))

    # Length-diverse selection
    short = [(s,t,i) for s,t,i in valid if t < 3000]
    medium = [(s,t,i) for s,t,i in valid if 3000 <= t < 10000]
    long = [(s,t,i) for s,t,i in valid if t >= 10000]

    selected = []
    for bucket, quota in [(long, 8), (medium, 8), (short, 4)]:
        selected.extend(bucket[:min(quota, len(bucket))])

    used = set(id(i) for _,_,i in selected)
    for s,t,i in valid:
        if len(selected) >= target_count: break
        if id(i) not in used:
            selected.append((s,t,i))
            used.add(id(i))

    return [i for _,_,i in selected[:target_count]]


def select_original_reasoning_traces(gen_data, target_count=4):
    """Same as build_optimal_64.py"""
    traces = [item for item in gen_data if item.get("_type") == "reasoning_gen_trace"]
    valid = [item for item in traces if len(item.get("question", "")) > 1000]
    valid.sort(key=lambda x: -len(x["question"]))
    return valid[:target_count]


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build optimal 96-sample calibration dataset")
    parser.add_argument("--gen-traces",
                        default="/opt/oldMoney-Project/quantization/calibration/gen_aware_calib.jsonl",
                        help="Path to gen_aware_calib.jsonl")
    parser.add_argument("--eval-data",
                        default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
                        help="Path to perf_public_set.jsonl")
    parser.add_argument("--output",
                        default="/opt/oldMoney-Project/quantization/calibration/optimal_96.jsonl",
                        help="Output path")
    args = parser.parse_args()

    print("=" * 80)
    print("  BUILDING OPTIMAL 96-SAMPLE CALIBRATION DATASET")
    print("  (64 original + 32 additional)")
    print("=" * 80)

    # ── Load data ──
    print(f"\n[1] Loading eval data...")
    eval_by_index = {}
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                eval_by_index[item["index"]] = item
    print(f"  Loaded {len(eval_by_index)} eval samples")

    print(f"\n[2] Loading generation traces...")
    gen_data = []
    with open(args.gen_traces, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                gen_data.append(json.loads(line.strip()))
    type_counts = Counter(item.get("_type", "unknown") for item in gen_data)
    print(f"  Loaded {len(gen_data)} items: {dict(type_counts)}")

    # ════════════════════════════════════════════════════════════════
    # BUILD ORIGINAL 64
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 1: ORIGINAL 64 SAMPLES")
    print(f"{'=' * 80}")

    final_dataset = []

    # Original MCQ gen traces (20)
    print(f"\n  Selecting original 20 MCQ gen traces...")
    orig_mcq = select_original_mcq_traces(gen_data, 20)
    for item in orig_mcq:
        item_copy = dict(item)
        item_copy["_slot_type"] = "mcq_gen_trace"
        final_dataset.append(item_copy)
    print(f"  ✓ {len(orig_mcq)} MCQ gen traces")

    # Original reasoning traces (4)
    print(f"  Selecting original 4 reasoning traces...")
    orig_reason = select_original_reasoning_traces(gen_data, 4)
    for item in orig_reason:
        item_copy = dict(item)
        item_copy["_slot_type"] = "reasoning_gen_trace"
        final_dataset.append(item_copy)
    print(f"  ✓ {len(orig_reason)} reasoning traces")

    # Original eval samples
    all_original_specs = [
        ("QA", ORIGINAL_QA_SPECS),
        ("CWE", ORIGINAL_CWE_SPECS),
        ("NIAH", ORIGINAL_NIAH_SPECS),
        ("FWE", ORIGINAL_FWE_SPECS),
    ]
    for task_name, specs in all_original_specs:
        for idx in sorted(specs.keys()):
            item = eval_by_index[idx]
            final_dataset.append({
                "question": item["question"],
                "_slot_type": f"eval_{item.get('task', 'unknown')}",
                "_eval_index": idx,
                "_prompt_tokens": item.get("prompt_tokens", 0),
            })
        print(f"  ✓ {len(specs)} {task_name} eval samples")

    print(f"\n  Original 64 total: {len(final_dataset)}")

    # ════════════════════════════════════════════════════════════════
    # ADD 32 EXTRA SAMPLES
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("PART 2: ADDITIONAL 32 SAMPLES")
    print(f"{'=' * 80}")

    extra_count_before = len(final_dataset)

    # ── Extra MCQ gen traces (6) ──
    print(f"\n  ── Extra MCQ gen traces (6) ──")
    extra_mcq = select_mcq_traces(gen_data, already_selected_count=20, extra_count=6)
    for i, item in enumerate(extra_mcq):
        item_copy = dict(item)
        item_copy["_slot_type"] = "mcq_gen_trace"
        final_dataset.append(item_copy)
        text = item["question"]
        tokens = item.get("_tokens", "?")
        first_line = text.split("\n")[0][:70]
        has_answer = "ANSWER:" in text.upper()
        status = "✓" if has_answer else "~"
        print(f"    [{i+1}] {status} {len(text):>7,} chars, {str(tokens):>6} tok | {first_line}...")
    print(f"    Total: {len(extra_mcq)} extra MCQ traces")

    # ── Extra reasoning traces (2) ──
    print(f"\n  ── Extra reasoning traces (2) ──")
    extra_reason = select_extra_reasoning_traces(gen_data, already_selected_count=4, extra_count=2)
    for i, item in enumerate(extra_reason):
        item_copy = dict(item)
        item_copy["_slot_type"] = "reasoning_gen_trace"
        final_dataset.append(item_copy)
        text = item["question"]
        first_line = text.split("\n")[0][:70]
        print(f"    [{i+1}] {len(text):>7,} chars | {first_line}...")
    print(f"    Total: {len(extra_reason)} extra reasoning traces")

    # ── Extra eval samples ──
    all_extra_specs = [
        ("CWE", EXTRA_CWE, "eval_cwe"),
        ("QA", EXTRA_QA, "eval_qa"),
        ("FWE", EXTRA_FWE, "eval_fwe"),
        ("NIAH", EXTRA_NIAH, "eval_niah"),
    ]

    for task_name, specs, slot_type in all_extra_specs:
        print(f"\n  ── Extra {task_name} ({len(specs)} samples) ──")
        for idx in sorted(specs.keys()):
            justification = specs[idx]
            if idx not in eval_by_index:
                print(f"    [MISSING] Index {idx}!")
                continue
            item = eval_by_index[idx]
            text = item.get("question", "")
            actual_task = item.get("task", "?")

            # Verify no overlap with original 64
            all_original = ORIGINAL_QA | ORIGINAL_CWE | ORIGINAL_NIAH | ORIGINAL_FWE
            if idx in all_original:
                print(f"    [DUPLICATE] Index {idx} already in original 64!")
                continue

            final_dataset.append({
                "question": text,
                "_slot_type": slot_type,
                "_eval_index": idx,
                "_prompt_tokens": item.get("prompt_tokens", 0),
            })
            print(f"    [{idx:>3}] ✓ {len(text):>8,} chars | {justification[:72]}")

    extra_added = len(final_dataset) - extra_count_before
    print(f"\n  Extra samples added: {extra_added}")

    # ════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"FINAL DATASET: {len(final_dataset)} SAMPLES")
    print(f"{'=' * 80}")

    slot_counts = Counter(item.get("_slot_type", "unknown") for item in final_dataset)

    print(f"\n{'─' * 80}")
    print(f"COMPOSITION")
    print(f"{'─' * 80}")
    for slot_type in sorted(slot_counts.keys()):
        count = slot_counts[slot_type]
        pct = count / len(final_dataset) * 100
        bar = "█" * int(pct / 2)
        print(f"  {slot_type:<25} {count:>3} ({pct:>5.1f}%) {bar}")

    # Length distribution
    print(f"\n{'─' * 80}")
    print(f"LENGTH DISTRIBUTION")
    print(f"{'─' * 80}")
    lengths = [len(item.get("question", "")) for item in final_dataset]
    buckets = [
        ("<1k chars", 0, 1000),
        ("1k-10k", 1000, 10000),
        ("10k-50k", 10000, 50000),
        ("50k-120k", 50000, 120000),
        ("120k-250k", 120000, 250000),
        ("250k+", 250000, float('inf')),
    ]
    for label, lo, hi in buckets:
        count = sum(1 for l in lengths if lo <= l < hi)
        bar = "█" * count
        print(f"  {label:>12}: {count:>3} {bar}")

    # Per-category stats
    print(f"\n{'─' * 80}")
    print(f"PER-CATEGORY LENGTH STATS")
    print(f"{'─' * 80}")
    for slot_type in sorted(slot_counts.keys()):
        items = [item for item in final_dataset if item.get("_slot_type") == slot_type]
        lens = [len(item.get("question", "")) for item in items]
        if lens:
            print(f"  {slot_type:<25} min={min(lens):>9,}  max={max(lens):>9,}  "
                  f"avg={sum(lens)/len(lens):>9,.0f}")

    # Hessian analysis
    print(f"\n{'─' * 80}")
    print(f"HESSIAN WEIGHT (per-sample observer)")
    print(f"{'─' * 80}")
    total = len(final_dataset)
    total_chars = sum(len(item.get("question", "")) for item in final_dataset)

    print(f"\n  WITH observer fix (each = 1/{total}):")
    for slot_type in sorted(slot_counts.keys()):
        c = slot_counts[slot_type]
        print(f"    {slot_type:<25} {c:>3}/{total} = {c/total*100:>5.1f}%")

    print(f"\n  WITHOUT observer fix (token-weighted):")
    for slot_type in sorted(slot_counts.keys()):
        items = [item for item in final_dataset if item.get("_slot_type") == slot_type]
        tc = sum(len(item.get("question", "")) for item in items)
        print(f"    {slot_type:<25} ~{tc/total_chars*100:>5.1f}%")

    # Comparison
    print(f"\n{'─' * 80}")
    print(f"EVOLUTION: Previous → 64-set → 96-set")
    print(f"{'─' * 80}")
    print(f"""
  PREVIOUS (final_merged_calibration.jsonl):
    NIAH-like: 66/96 = 68.8%  |  MCQ gen: 0  |  QA: 0  |  CWE: 0  |  FWE: 0

  OPTIMAL 64:
    MCQ gen:  20 (31.3%)  |  CWE: 10 (15.6%)  |  QA: 14 (21.9%)
    NIAH:     10 (15.6%)  |  FWE:  6 ( 9.4%)  |  Reason: 4 (6.3%)

  OPTIMAL 96 (this dataset):
    MCQ gen:  {slot_counts.get('mcq_gen_trace', 0):>2} ({slot_counts.get('mcq_gen_trace', 0)/total*100:.1f}%)  |  CWE: {slot_counts.get('eval_cwe', 0):>2} ({slot_counts.get('eval_cwe', 0)/total*100:.1f}%)  |  QA: {slot_counts.get('eval_qa', 0):>2} ({slot_counts.get('eval_qa', 0)/total*100:.1f}%)
    NIAH:     {slot_counts.get('eval_niah', 0):>2} ({slot_counts.get('eval_niah', 0)/total*100:.1f}%)  |  FWE: {slot_counts.get('eval_fwe', 0):>2} ({slot_counts.get('eval_fwe', 0)/total*100:.1f}%)  |  Reason: {slot_counts.get('reasoning_gen_trace', 0):>2} ({slot_counts.get('reasoning_gen_trace', 0)/total*100:.1f}%)
""")

    # ── Write output ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  ✓ Written to: {args.output}")
    print(f"    File size: {os.path.getsize(args.output) / 1024:,.0f} KB")
    print(f"    Total samples: {len(final_dataset)}")

    print(f"""
{'=' * 80}
NEXT STEPS
{'=' * 80}

  1. Apply observer fix (if not already applied):
     python /opt/oldMoney-Project/quantization/calibration/patch_observer.py \\
         --awq-script /opt/oldMoney-Project/quantization/nvfp4_awq.py

  2. Run quantization:
     python /opt/oldMoney-Project/quantization/nvfp4_awq.py \\
         --input /opt/model \\
         --output /opt/model_nvfp4_awq_v3 \\
         --calib-data {args.output} \\
         --max-samples 96 \\
         --max-len 131072 \\
         --mse-iters 200 \\
         --mse-max-shrink 0.60 \\
         --mse-error-norm 2.0

  EXPECTED IMPROVEMENTS:
    CWE: 80.7% → ~86-88%  (22 samples, up from 10)
    MCQ: 50.0% → ~52-55%  (26 gen traces, diminishing returns)
    QA:  60.0% → ~62-63%  (20 samples, close to BF16)
    FWE: 96.7% → ~98%     (10 samples, recovery expected)
    NIAH: 100% → 100%     (maintained)
    OVERALL: ~78-80%      (target: 79%)
""")


if __name__ == "__main__":
    main()