#!/usr/bin/env python3
"""
Deep analysis of eval dataset: token lengths, prompt structure, domain coverage.
Uses the actual model tokenizer for accurate token counts.

Run on your server:
  python /opt/oldMoney-Project/quantization/calibration/ANALYZE_DATA.py \
      --eval-data /opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
      --tokenizer /opt/model \
      --predictions-bf16 /opt/SOAR-Toolkit/outputs/20260322_020604/predictions.jsonl \
      --predictions-awq /opt/SOAR-Toolkit/outputs/20260321_163020/predictions.jsonl
"""

import os
import re
import sys
import json
import argparse
import numpy as np
from collections import defaultdict, Counter

def load_tokenizer(path):
    """Load tokenizer from model directory."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tokenizer

def count_tokens(tokenizer, text):
    """Count tokens for a given text."""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except:
        # Fallback: estimate from chars
        return len(text) // 3

def extract_prompt_and_expected(sample):
    """Extract the user prompt and expected answer structure from an eval sample."""
    # Try common field names
    prompt = ""
    for key in ["prompt", "question", "input", "text", "content", "messages"]:
        if key in sample:
            val = sample[key]
            if isinstance(val, str):
                prompt = val
                break
            elif isinstance(val, list):
                # messages format
                for msg in val:
                    if isinstance(msg, dict):
                        if msg.get("role") == "user":
                            prompt = msg.get("content", "")
                            break
                if not prompt:
                    prompt = json.dumps(val)
                break
    
    # Try to get the gold answer
    gold = sample.get("gold", sample.get("answer", sample.get("expected", sample.get("label", None))))
    task = sample.get("task", sample.get("type", sample.get("category", "unknown")))
    
    return prompt, gold, task

def analyze_prompt_structure(prompt, task):
    """Analyze what's inside a prompt - MCQ options, document context, etc."""
    info = {}
    
    # Check for MCQ options
    has_options = bool(re.search(r'\b[A-D]\)', prompt) or re.search(r'\b[A-D][\.\:]', prompt))
    info["has_mcq_options"] = has_options
    
    # Check for document/context markers
    info["has_document_context"] = any(marker in prompt.lower() for marker in 
        ["document", "passage", "context", "article", "text:", "below:", "following"])
    
    # Check for think/reasoning instructions
    info["asks_for_reasoning"] = any(marker in prompt.lower() for marker in 
        ["step by step", "think", "reason", "explain", "derive", "prove", "calculate"])
    
    # Check for specific answer format instructions
    info["has_format_instruction"] = any(marker in prompt for marker in
        ["ANSWER:", "answer:", "Answer:", "format", "Format"])
    
    # Count lines
    info["num_lines"] = prompt.count('\n') + 1
    
    # Check if it contains code
    info["has_code"] = bool(re.search(r'```|def |class |import |function ', prompt))
    
    # Check for numbers/math
    math_patterns = len(re.findall(r'\d+\.\d+|\d+/\d+|[=+\-*/^]|\bsin\b|\bcos\b|\blog\b|\bexp\b', prompt))
    info["math_density"] = math_patterns
    
    # Look for embedded long text (UUID patterns, random words, etc.)
    uuid_count = len(re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}', prompt))
    info["uuid_count"] = uuid_count
    
    # Detect language
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', prompt))
    info["chinese_char_count"] = chinese_chars
    info["is_chinese"] = chinese_chars > 50
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Deep eval dataset analysis")
    parser.add_argument("--eval-data", required=True, help="Path to perf_public_set.jsonl")
    parser.add_argument("--tokenizer", required=True, help="Path to model directory (for tokenizer)")
    parser.add_argument("--predictions-bf16", default=None, help="BF16 predictions.jsonl (optional)")
    parser.add_argument("--predictions-awq", default=None, help="AWQ predictions.jsonl (optional)")
    parser.add_argument("--output", default=None, help="Save analysis to JSON file")
    args = parser.parse_args()

    # ── Load tokenizer ──
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")
    
    # Check special tokens
    print(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    # Check think tokens
    think_tokens = {}
    for tok_str in ["<think>", "</think>", "ANSWER:", "ANSWER"]:
        encoded = tokenizer.encode(tok_str, add_special_tokens=False)
        think_tokens[tok_str] = encoded
        print(f"  '{tok_str}' → token ids: {encoded}")

    # ── Load eval data ──
    print(f"\nLoading eval data from {args.eval_data}...")
    samples = []
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    print(f"  Total samples: {len(samples)}")
    
    # ── Print raw field names from first sample ──
    print(f"\n  Fields in first sample: {list(samples[0].keys())}")
    # Print first sample (truncated) for structure understanding
    first = samples[0]
    print(f"  First sample preview:")
    for k, v in first.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:100] + f"...({len(v_str)} chars)..." + v_str[-50:]
        print(f"    {k}: {v_str}")

    # ── Tokenize and analyze every sample ──
    print(f"\nTokenizing all {len(samples)} samples (this may take a few minutes for long contexts)...")
    
    task_data = defaultdict(list)  # task -> list of sample info dicts
    all_data = []
    
    for i, sample in enumerate(samples):
        prompt, gold, task = extract_prompt_and_expected(sample)
        task = str(task).lower().strip()
        
        # Tokenize prompt
        prompt_tokens = count_tokens(tokenizer, prompt)
        prompt_chars = len(prompt)
        
        # Analyze structure
        structure = analyze_prompt_structure(prompt, task)
        
        # Gold answer analysis
        if isinstance(gold, list):
            gold_str = " | ".join(str(g) for g in gold)
        else:
            gold_str = str(gold) if gold else ""
        gold_tokens = count_tokens(tokenizer, gold_str)
        
        info = {
            "index": i,
            "task": task,
            "prompt_tokens": prompt_tokens,
            "prompt_chars": prompt_chars,
            "gold_str": gold_str[:200],
            "gold_tokens": gold_tokens,
            **structure,
        }
        
        task_data[task].append(info)
        all_data.append(info)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)}...", end='\r')
    
    print(f"  Processed {len(samples)}/{len(samples)} ✓" + " " * 20)

    # ══════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════
    
    print(f"\n{'=' * 90}")
    print(f"{'EVAL DATASET ANALYSIS':^90}")
    print(f"{'=' * 90}")
    
    # ── Per-task summary ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 1: PER-TASK TOKEN LENGTH DISTRIBUTION':^90}")
    print(f"{'─' * 90}")
    
    print(f"\n  {'Task':<8} {'N':>4} {'Min Tok':>10} {'P25 Tok':>10} {'Median':>10} "
          f"{'P75 Tok':>10} {'Max Tok':>10} {'Mean Tok':>10} {'Std Tok':>10}")
    print(f"  {'─' * 82}")
    
    for task in sorted(task_data.keys()):
        items = task_data[task]
        toks = [it["prompt_tokens"] for it in items]
        arr = np.array(toks)
        print(f"  {task.upper():<8} {len(items):>4} {arr.min():>10,} {int(np.percentile(arr, 25)):>10,} "
              f"{int(np.median(arr)):>10,} {int(np.percentile(arr, 75)):>10,} {arr.max():>10,} "
              f"{arr.mean():>10,.0f} {arr.std():>10,.0f}")
    
    # Overall
    all_toks = [it["prompt_tokens"] for it in all_data]
    arr = np.array(all_toks)
    print(f"  {'ALL':<8} {len(all_data):>4} {arr.min():>10,} {int(np.percentile(arr, 25)):>10,} "
          f"{int(np.median(arr)):>10,} {int(np.percentile(arr, 75)):>10,} {arr.max():>10,} "
          f"{arr.mean():>10,.0f} {arr.std():>10,.0f}")

    # ── Gold answer analysis ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 2: GOLD ANSWER / EXPECTED OUTPUT ANALYSIS':^90}")
    print(f"{'─' * 90}")
    
    print(f"\n  {'Task':<8} {'N':>4} {'Gold Example':<50} {'Gold Tok':>10}")
    print(f"  {'─' * 75}")
    for task in sorted(task_data.keys()):
        items = task_data[task]
        gold_toks = [it["gold_tokens"] for it in items]
        # Show a few examples
        examples = list(set(it["gold_str"][:40] for it in items[:5]))
        for j, ex in enumerate(examples[:3]):
            if j == 0:
                print(f"  {task.upper():<8} {len(items):>4} {ex:<50} {np.mean(gold_toks):>10.1f}")
            else:
                print(f"  {'':8} {'':>4} {ex:<50}")

    # ── Structural features ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 3: PROMPT STRUCTURAL FEATURES':^90}")
    print(f"{'─' * 90}")
    
    print(f"\n  {'Task':<8} {'MCQ Opts':>10} {'Has Doc':>10} {'Reasoning':>10} "
          f"{'Fmt Instr':>10} {'Has Code':>10} {'Math Den':>10} {'UUIDs':>8} {'Chinese':>8}")
    print(f"  {'─' * 84}")
    for task in sorted(task_data.keys()):
        items = task_data[task]
        n = len(items)
        mcq = sum(1 for it in items if it["has_mcq_options"]) / n * 100
        doc = sum(1 for it in items if it["has_document_context"]) / n * 100
        reas = sum(1 for it in items if it["asks_for_reasoning"]) / n * 100
        fmt = sum(1 for it in items if it["has_format_instruction"]) / n * 100
        code = sum(1 for it in items if it["has_code"]) / n * 100
        math_d = np.mean([it["math_density"] for it in items])
        uuids = np.mean([it["uuid_count"] for it in items])
        cn = sum(1 for it in items if it["is_chinese"]) / n * 100
        print(f"  {task.upper():<8} {mcq:>9.0f}% {doc:>9.0f}% {reas:>9.0f}% "
              f"{fmt:>9.0f}% {code:>9.0f}% {math_d:>10.1f} {uuids:>8.1f} {cn:>7.0f}%")

    # ── Per-sample token length listing ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 4: INDIVIDUAL SAMPLE TOKEN LENGTHS (sorted by task)':^90}")
    print(f"{'─' * 90}")
    
    for task in sorted(task_data.keys()):
        items = sorted(task_data[task], key=lambda x: x["prompt_tokens"])
        print(f"\n  {task.upper()} ({len(items)} samples):")
        print(f"    {'#':>4} {'Tokens':>10} {'Chars':>10} {'Gold':>30} {'Features'}")
        print(f"    {'─' * 75}")
        for it in items:
            features = []
            if it["has_mcq_options"]: features.append("MCQ")
            if it["has_document_context"]: features.append("DOC")
            if it["asks_for_reasoning"]: features.append("REASON")
            if it["is_chinese"]: features.append("ZH")
            if it["uuid_count"] > 0: features.append(f"UUID×{it['uuid_count']}")
            if it["math_density"] > 10: features.append(f"MATH({it['math_density']})")
            feat_str = " ".join(features) if features else "-"
            gold_short = it["gold_str"][:28]
            print(f"    {it['index']+1:>4} {it['prompt_tokens']:>10,} {it['prompt_chars']:>10,} "
                  f"{gold_short:>30} {feat_str}")

    # ── Token length histogram (text-based) ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 5: TOKEN LENGTH HISTOGRAM':^90}")
    print(f"{'─' * 90}")
    
    for task in sorted(task_data.keys()):
        items = task_data[task]
        toks = [it["prompt_tokens"] for it in items]
        
        # Define bins
        bins = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
        counts, _ = np.histogram(toks, bins=bins)
        
        print(f"\n  {task.upper()}:")
        max_count = max(counts) if max(counts) > 0 else 1
        for j in range(len(counts)):
            bar = "█" * int(counts[j] / max_count * 40) if counts[j] > 0 else ""
            label = f"  {bins[j]:>7,}-{bins[j+1]:>7,}"
            print(f"  {label} │ {bar} {counts[j]}")

    # ── Prompt content deep-dive: first/last 200 chars ──
    print(f"\n{'─' * 90}")
    print(f"{'SECTION 6: PROMPT CONTENT SAMPLES (first & last 150 chars per task)':^90}")
    print(f"{'─' * 90}")
    
    for task in sorted(task_data.keys()):
        items = task_data[task]
        # Pick shortest, median, longest
        sorted_items = sorted(items, key=lambda x: x["prompt_tokens"])
        picks = [sorted_items[0], sorted_items[len(sorted_items)//2], sorted_items[-1]]
        labels = ["SHORTEST", "MEDIAN", "LONGEST"]
        
        print(f"\n  ── {task.upper()} ──")
        for label, it in zip(labels, picks):
            idx = it["index"]
            prompt = extract_prompt_and_expected(samples[idx])[0]
            first = prompt[:150].replace('\n', '↵')
            last = prompt[-150:].replace('\n', '↵')
            print(f"\n    [{label}] Sample #{idx+1} ({it['prompt_tokens']:,} tokens)")
            print(f"    FIRST: {first}")
            print(f"    LAST:  {last}")

    # ══════════════════════════════════════════════════
    # PREDICTIONS ANALYSIS (if provided)
    # ══════════════════════════════════════════════════
    
    if args.predictions_bf16 or args.predictions_awq:
        print(f"\n{'=' * 90}")
        print(f"{'PREDICTIONS OUTPUT TOKEN ANALYSIS':^90}")
        print(f"{'=' * 90}")
        
        for label, path in [("BF16", args.predictions_bf16), ("AWQ", args.predictions_awq)]:
            if not path or not os.path.exists(path):
                continue
            
            print(f"\n{'─' * 90}")
            print(f"  {label} Model Outputs")
            print(f"{'─' * 90}")
            
            preds = []
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line.strip()))
            
            pred_task_data = defaultdict(list)
            for p in preds:
                task = str(p.get("task", "unknown")).lower()
                output = p.get("prediction", "")
                output_tokens = count_tokens(tokenizer, output)
                score = p.get("score", 0)
                pred_task_data[task].append({
                    "output_tokens": output_tokens,
                    "output_chars": len(output),
                    "score": score,
                    "has_think_close": "</think>" in output,
                    "has_answer_line": bool(re.search(r'ANSWER:\s*[A-D]', output)),
                })
            
            print(f"\n  {'Task':<8} {'N':>4} {'Min Out':>10} {'Med Out':>10} {'Max Out':>10} "
                  f"{'Mean Out':>10} {'</think>%':>10} {'Score':>8}")
            print(f"  {'─' * 72}")
            
            for task in sorted(pred_task_data.keys()):
                items = pred_task_data[task]
                out_toks = [it["output_tokens"] for it in items]
                arr = np.array(out_toks)
                think_pct = sum(1 for it in items if it["has_think_close"]) / len(items) * 100
                avg_score = np.mean([it["score"] for it in items])
                print(f"  {task.upper():<8} {len(items):>4} {arr.min():>10,} {int(np.median(arr)):>10,} "
                      f"{arr.max():>10,} {arr.mean():>10,.0f} {think_pct:>9.0f}% {avg_score:>7.1%}")
            
            # Per-sample output token listing for MCQ
            mcq_items = pred_task_data.get("mcq", [])
            if mcq_items:
                print(f"\n  MCQ Output Token Distribution ({label}):")
                print(f"    {'#':>4} {'Out Tok':>10} {'Score':>8} {'</think>':>10} {'ANSWER':>10}")
                print(f"    {'─' * 45}")
                for j, it in enumerate(mcq_items):
                    think = "✓" if it["has_think_close"] else "✗"
                    answer = "✓" if it["has_answer_line"] else "✗"
                    score_mark = "✓" if it["score"] > 0 else "✗"
                    print(f"    {j+1:>4} {it['output_tokens']:>10,} {score_mark:>8} {think:>10} {answer:>10}")

    # ══════════════════════════════════════════════════
    # CALIBRATION RECOMMENDATIONS
    # ══════════════════════════════════════════════════
    
    print(f"\n{'=' * 90}")
    print(f"{'CALIBRATION DATA RECOMMENDATIONS':^90}")
    print(f"{'=' * 90}")
    
    print(f"""
  Based on the analysis above, your calibration data should cover:
  
  PROMPT TOKEN LENGTH BUCKETS (from eval distribution):""")
    
    for task in sorted(task_data.keys()):
        items = task_data[task]
        toks = [it["prompt_tokens"] for it in items]
        arr = np.array(toks)
        print(f"    {task.upper():<8}: {arr.min():>7,} - {arr.max():>7,} tokens "
              f"(median {int(np.median(arr)):>7,})")
    
    print(f"""
  KEY OBSERVATIONS FOR CALIBRATION:""")
    
    # Check if there are very short and very long samples
    short_tasks = [t for t in task_data if np.median([it["prompt_tokens"] for it in task_data[t]]) < 1000]
    long_tasks = [t for t in task_data if np.median([it["prompt_tokens"] for it in task_data[t]]) > 50000]
    
    if short_tasks:
        print(f"    - Short-prompt tasks ({', '.join(t.upper() for t in short_tasks)}): "
              f"Need generation-aware traces (prompt+reasoning)")
    if long_tasks:
        print(f"    - Long-prompt tasks ({', '.join(t.upper() for t in long_tasks)}): "
              f"Need long-context calibration samples matching these lengths")
    
    # Check for structural patterns
    mcq_tasks = [t for t in task_data if sum(it["has_mcq_options"] for it in task_data[t]) > len(task_data[t]) * 0.5]
    if mcq_tasks:
        print(f"    - MCQ-style tasks ({', '.join(t.upper() for t in mcq_tasks)}): "
              f"Need MCQ prompts with options A/B/C/D")
    
    uuid_tasks = [t for t in task_data if np.mean([it["uuid_count"] for it in task_data[t]]) > 5]
    if uuid_tasks:
        print(f"    - UUID-heavy tasks ({', '.join(t.upper() for t in uuid_tasks)}): "
              f"Contains random identifiers (low semantic content)")

    # ── Save analysis to JSON ──
    if args.output:
        analysis = {
            "total_samples": len(samples),
            "tasks": {},
        }
        for task in sorted(task_data.keys()):
            items = task_data[task]
            toks = [it["prompt_tokens"] for it in items]
            analysis["tasks"][task] = {
                "count": len(items),
                "token_stats": {
                    "min": int(np.min(toks)),
                    "p25": int(np.percentile(toks, 25)),
                    "median": int(np.median(toks)),
                    "p75": int(np.percentile(toks, 75)),
                    "max": int(np.max(toks)),
                    "mean": float(np.mean(toks)),
                    "std": float(np.std(toks)),
                },
                "per_sample_tokens": toks,
                "features": {
                    "mcq_options_pct": sum(1 for it in items if it["has_mcq_options"]) / len(items),
                    "document_context_pct": sum(1 for it in items if it["has_document_context"]) / len(items),
                    "reasoning_pct": sum(1 for it in items if it["asks_for_reasoning"]) / len(items),
                    "chinese_pct": sum(1 for it in items if it["is_chinese"]) / len(items),
                    "avg_uuid_count": float(np.mean([it["uuid_count"] for it in items])),
                    "avg_math_density": float(np.mean([it["math_density"] for it in items])),
                },
            }
        
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n  Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()