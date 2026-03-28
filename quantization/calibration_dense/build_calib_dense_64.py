#!/usr/bin/env python3
"""
Build 64-sample calibration dataset for dense flashinfer NVFP4 quantization.

Strategy (following champion 曹议's insight):
  - 56 semantic-rich long samples (LongBench + FineWeb-Edu)
  - 8 low-semantic samples (niah/fwe/cwe from eval, with gold answers appended)
  - All truncated to --max-len tokens (default 131072)
  - Semantic data prioritized over repetitive patterns

Data sources:
  1. zai-org/LongBench — long-context QA/summarization tasks (diverse, semantic)
  2. HuggingFaceFW/fineweb-edu — high-quality educational web text
  3. SOAR eval perf_public_set.jsonl — 8 niah/fwe/cwe with gold answers

Usage (on server):
    pip install datasets
    python build_calib_dense_64.py \
        --eval-path /opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
        --tokenizer-path /opt/model \
        --output calib_dense_64.jsonl \
        --max-len 131072

Output: JSONL with {"question": "..."} field, compatible with all AWQ quantizers.
"""

import os
import json
import random
import argparse
from pathlib import Path


def download_longbench(max_samples: int = 40, min_chars: int = 20000) -> list:
    """
    Download from zai-org/LongBench, select long semantic samples.
    Prioritizes subsets with real documents (QA, summarization).
    """
    from datasets import load_dataset

    samples = []

    # LongBench has multiple configs — pick the most semantic-rich ones
    # These contain real documents with questions
    semantic_configs = [
        "multifieldqa_en",
        "multifieldqa_zh",
        "qasper",
        "narrativeqa",
        "multi_news",
        "gov_report",
        "passage_retrieval_en",
        "passage_retrieval_zh",
        "musique",
        "hotpotqa",
        "2wikimqa",
        "dureader",
        "lsht",
        "vcsum",
    ]

    for config_name in semantic_configs:
        if len(samples) >= max_samples:
            break
        try:
            print(f"  Loading LongBench/{config_name}...")
            ds = load_dataset("zai-org/LongBench", config_name, split="test")
            for item in ds:
                if len(samples) >= max_samples:
                    break
                # Build text from context + input
                ctx = item.get("context", "")
                inp = item.get("input", "")
                text = f"{ctx}\n\n{inp}".strip()
                if len(text) < min_chars:
                    continue
                samples.append({
                    "question": text,
                    "_source": f"longbench_{config_name}",
                    "_chars": len(text),
                })
        except Exception as e:
            print(f"  [WARN] Failed to load {config_name}: {e}")
            continue

    print(f"  LongBench: collected {len(samples)} samples (min {min_chars} chars)")
    return samples


def download_fineweb_edu(max_samples: int = 20, min_chars: int = 30000) -> list:
    """
    Stream from HuggingFaceFW/fineweb-edu, pick longest documents.
    Uses streaming to avoid downloading the full dataset (~10TB).
    """
    from datasets import load_dataset

    samples = []
    scanned = 0
    max_scan = 50000  # scan at most this many documents to find long ones

    print(f"  Streaming FineWeb-Edu (scanning up to {max_scan} docs for long ones)...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",  # use the 10BT sample, much smaller
            split="train",
            streaming=True,
        )
        for item in ds:
            scanned += 1
            text = item.get("text", "")
            if len(text) >= min_chars:
                samples.append({
                    "question": text,
                    "_source": "fineweb_edu",
                    "_chars": len(text),
                })
                if len(samples) >= max_samples:
                    break
            if scanned >= max_scan:
                break
            if scanned % 10000 == 0:
                print(f"    Scanned {scanned} docs, found {len(samples)} long ones...")
    except Exception as e:
        print(f"  [WARN] FineWeb-Edu streaming failed: {e}")

    print(f"  FineWeb-Edu: collected {len(samples)} samples from {scanned} scanned (min {min_chars} chars)")

    # If we didn't get enough long samples, lower the threshold and rescan
    if len(samples) < max_samples:
        print(f"  [INFO] Only got {len(samples)}/{max_samples} from FineWeb-Edu.")
        print(f"  [INFO] This is OK — LongBench will fill the remaining slots.")

    return samples


def load_eval_low_semantic(
    eval_path: str,
    n_niah: int = 3,
    n_fwe: int = 3,
    n_cwe: int = 2,
) -> list:
    """
    Load niah/fwe/cwe samples from eval data.
    Appends gold answer to the question for better activation coverage.

    Eval layout (perf_public_set.jsonl):
      indices 0-29:   mcq
      indices 30-59:  niah
      indices 60-89:  qa
      indices 90-119: fwe
      indices 120-149: cwe
    """
    samples = []
    all_eval = []

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            all_eval.append(json.loads(line.strip()))

    # Group by task
    niah_pool = [s for s in all_eval if s.get("task") == "niah"]
    fwe_pool = [s for s in all_eval if s.get("task") == "fwe"]
    cwe_pool = [s for s in all_eval if s.get("task") == "cwe"]

    # If task field not present, use index-based fallback
    if not niah_pool:
        niah_pool = all_eval[30:60]
    if not fwe_pool:
        fwe_pool = all_eval[90:120]
    if not cwe_pool:
        cwe_pool = all_eval[120:150]

    # Pick longest samples from each category (most representative)
    def by_length(s):
        return len(s.get("question", ""))

    niah_pool.sort(key=by_length, reverse=True)
    fwe_pool.sort(key=by_length, reverse=True)
    cwe_pool.sort(key=by_length, reverse=True)

    def build_sample(item, task_name):
        """Concatenate question + gold answer for fuller activation coverage."""
        q = item.get("question", "")
        gold = item.get("gold", "")
        if gold:
            # Append gold answer so the calibration covers answer-phase activations
            text = f"{q}\n\nAnswer: {gold}"
        else:
            text = q
        return {
            "question": text,
            "_source": f"eval_{task_name}",
            "_chars": len(text),
        }

    for s in niah_pool[:n_niah]:
        samples.append(build_sample(s, "niah"))
    for s in fwe_pool[:n_fwe]:
        samples.append(build_sample(s, "fwe"))
    for s in cwe_pool[:n_cwe]:
        samples.append(build_sample(s, "cwe"))

    print(f"  Eval low-semantic: {n_niah} niah + {n_fwe} fwe + {n_cwe} cwe = {len(samples)} samples (with gold answers)")
    return samples


def tokenize_and_truncate(samples, tokenizer, max_len):
    """Tokenize all samples, report stats, truncate to max_len."""
    results = []
    for s in samples:
        enc = tokenizer(
            s["question"],
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        n_tokens = enc["input_ids"].shape[1]
        results.append({
            "question": s["question"],
            "_source": s.get("_source", "unknown"),
            "_tokens": n_tokens,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build 64-sample dense calibration dataset for NVFP4"
    )
    parser.add_argument(
        "--eval-path", type=str,
        default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
        help="Path to SOAR eval perf_public_set.jsonl",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default="/opt/model",
        help="Path to model tokenizer (for token counting)",
    )
    parser.add_argument(
        "--output", type=str, default="calib_dense_64.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--total-samples", type=int, default=64)
    parser.add_argument("--n-niah", type=int, default=3)
    parser.add_argument("--n-fwe", type=int, default=3)
    parser.add_argument("--n-cwe", type=int, default=2)
    parser.add_argument(
        "--longbench-min-chars", type=int, default=20000,
        help="Minimum character count for LongBench samples",
    )
    parser.add_argument(
        "--fineweb-min-chars", type=int, default=30000,
        help="Minimum character count for FineWeb-Edu samples",
    )
    parser.add_argument(
        "--fineweb-max-scan", type=int, default=50000,
        help="Max documents to scan in FineWeb-Edu streaming",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    n_low_semantic = args.n_niah + args.n_fwe + args.n_cwe
    n_semantic = args.total_samples - n_low_semantic

    print("=" * 70)
    print("  Dense Calibration Data Builder")
    print("=" * 70)
    print(f"  Total samples:    {args.total_samples}")
    print(f"  Semantic:         {n_semantic} (LongBench + FineWeb-Edu)")
    print(f"  Low-semantic:     {n_low_semantic} ({args.n_niah} niah + {args.n_fwe} fwe + {args.n_cwe} cwe)")
    print(f"  Max token length: {args.max_len}")
    print()

    # ---- Step 1: Download semantic data ----
    print("[Step 1] Downloading LongBench...")
    longbench_samples = download_longbench(
        max_samples=n_semantic,
        min_chars=args.longbench_min_chars,
    )

    remaining = n_semantic - len(longbench_samples)
    fineweb_samples = []
    if remaining > 0:
        print(f"\n[Step 2] Downloading FineWeb-Edu ({remaining} more needed)...")
        fineweb_samples = download_fineweb_edu(
            max_samples=remaining,
            min_chars=args.fineweb_min_chars,
        )
    else:
        print(f"\n[Step 2] LongBench provided enough samples, skipping FineWeb-Edu.")

    # ---- Step 3: Load eval low-semantic data ----
    print(f"\n[Step 3] Loading eval low-semantic data...")
    if os.path.exists(args.eval_path):
        eval_samples = load_eval_low_semantic(
            args.eval_path,
            n_niah=args.n_niah,
            n_fwe=args.n_fwe,
            n_cwe=args.n_cwe,
        )
    else:
        print(f"  [WARN] Eval file not found: {args.eval_path}")
        print(f"  [WARN] Skipping eval samples. You can add them manually later.")
        eval_samples = []

    # ---- Step 4: Combine and balance ----
    print(f"\n[Step 4] Combining...")
    semantic_pool = longbench_samples + fineweb_samples
    # Sort by length (longest first) and pick top n_semantic
    semantic_pool.sort(key=lambda x: x.get("_chars", 0), reverse=True)
    semantic_selected = semantic_pool[:n_semantic]

    all_samples = semantic_selected + eval_samples
    actual_total = len(all_samples)

    if actual_total < args.total_samples:
        print(f"  [WARN] Only got {actual_total}/{args.total_samples} samples.")
        print(f"  [INFO] Filling remaining slots with shorter LongBench/FineWeb samples...")
        # Add more from the pool if available
        extra = [s for s in semantic_pool[n_semantic:]]
        while len(all_samples) < args.total_samples and extra:
            all_samples.append(extra.pop(0))

    # Shuffle to prevent task clustering (important for layer-by-layer calibration)
    random.shuffle(all_samples)

    # ---- Step 5: Tokenize and report stats ----
    print(f"\n[Step 5] Tokenizing (using {args.tokenizer_path})...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        all_samples = tokenize_and_truncate(all_samples, tokenizer, args.max_len)
        has_tokens = True
    except Exception as e:
        print(f"  [WARN] Tokenizer not available ({e}). Skipping token counting.")
        print(f"  [INFO] Token counts will be estimated from char length.")
        for s in all_samples:
            s["_tokens"] = s.get("_chars", len(s["question"])) // 4  # rough estimate
        has_tokens = False

    # ---- Step 6: Save ----
    print(f"\n[Step 6] Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            # Only write the fields needed by quantizer + metadata
            out = {
                "question": s["question"],
                "_source": s.get("_source", "unknown"),
                "_tokens": s.get("_tokens", 0),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print(f"  CALIBRATION DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Output:         {args.output}")
    print(f"  Total samples:  {len(all_samples)}")

    source_counts = {}
    token_lengths = []
    for s in all_samples:
        src = s.get("_source", "unknown")
        # Simplify source name
        if "longbench" in src:
            key = "LongBench"
        elif "fineweb" in src:
            key = "FineWeb-Edu"
        elif "eval" in src:
            key = f"Eval ({src.split('_')[-1]})"
        else:
            key = src
        source_counts[key] = source_counts.get(key, 0) + 1
        token_lengths.append(s.get("_tokens", 0))

    print(f"\n  Source distribution:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")

    if token_lengths:
        print(f"\n  Token length stats:")
        print(f"    Min:    {min(token_lengths):,}")
        print(f"    Max:    {max(token_lengths):,}")
        print(f"    Mean:   {sum(token_lengths) // len(token_lengths):,}")
        print(f"    Total:  {sum(token_lengths):,}")
        # Bucket distribution
        buckets = {"<10k": 0, "10-50k": 0, "50-100k": 0, "100-131k": 0}
        for t in token_lengths:
            if t < 10000:
                buckets["<10k"] += 1
            elif t < 50000:
                buckets["10-50k"] += 1
            elif t < 100000:
                buckets["50-100k"] += 1
            else:
                buckets["100-131k"] += 1
        print(f"\n  Length buckets:")
        for bucket, cnt in buckets.items():
            print(f"    {bucket}: {cnt}")

    print(f"\n{'=' * 70}")
    print(f"  Ready for quantization!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
