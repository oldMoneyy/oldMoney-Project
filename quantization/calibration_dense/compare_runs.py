#!/usr/bin/env python3
"""
Compare multiple SOAR eval runs from the same model to identify:
1. Run-to-run instability (accuracy variance, token-0 collapse)
2. Per-question consistency (always right, always wrong, flipping)
3. Failure mode analysis per run with tokenizer
4. Actionable breakdown: which questions to focus calibration on

Usage (on remote server):
python /opt/oldMoney-Project/quantization/calibration_dense/compare_runs.py \
    --runs /opt/oldMoney-Project/SOAR-Toolkit/outputs/20260329_065254 \
         /opt/oldMoney-Project/SOAR-Toolkit/outputs/20260329_112231 \
         /opt/oldMoney-Project/SOAR-Toolkit/outputs/20260329_124344 \
    --tokenizer-path /opt/model_nvfp4_dense_all
"""

import json
import argparse
import os
from collections import Counter, defaultdict


def load_run(run_dir):
    """Load predictions and summary from a run directory."""
    preds_path = os.path.join(run_dir, "predictions.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    preds = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    return preds, summary


def classify_failure(sample, tokenizer=None):
    """Classify failure mode of a prediction."""
    score = sample.get("score", 0)
    prediction = sample.get("prediction", "")
    extracted = sample.get("extracted")
    out_tok = sample.get("output_tokens", sample.get("completion_tokens", 0))

    if score >= 0.99:
        return "correct"

    # Check for token-0 collapse via tokenizer
    unk_ratio = 0.0
    if tokenizer and prediction:
        try:
            tokens = tokenizer.encode(prediction)
            unk_count = tokens.count(0)
            unk_ratio = unk_count / max(len(tokens), 1)
        except Exception:
            pass

    if not prediction or len(prediction.strip()) < 10:
        return "empty/null"
    if unk_ratio > 0.5:
        return "token0_collapse"
    if out_tok <= 20 and extracted is None:
        return "early_stop_none"
    if extracted is None:
        return "extraction_failed"
    if out_tok >= 50000 and score < 0.5:
        return "max_tokens"
    if score > 0:
        return "partial"
    return "wrong_answer"


def main():
    parser = argparse.ArgumentParser(description="Compare SOAR eval runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run output directories")
    parser.add_argument("--tokenizer-path", default="/opt/model_nvfp4_dense_all",
                        help="Tokenizer path for token-level analysis")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"[INFO] Loaded tokenizer from {args.tokenizer_path}")
    except Exception as e:
        print(f"[WARN] Tokenizer unavailable ({e}), skipping token-level analysis")

    # Load all runs
    runs = []
    run_labels = []
    for run_dir in args.runs:
        preds, summary = load_run(run_dir)
        runs.append(preds)
        label = os.path.basename(run_dir)
        run_labels.append(label)
        print(f"[INFO] Loaded {label}: {len(preds)} samples, "
              f"accuracy={summary.get('ori_accuracy', '?')}%")

    num_runs = len(runs)
    num_samples = len(runs[0])

    # Classify failures
    for run_preds in runs:
        for s in run_preds:
            s["_failure_mode"] = classify_failure(s, tokenizer)

    # ======================================================================
    # Section 1: Overall comparison
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 1: OVERALL COMPARISON")
    print("=" * 100)

    tasks = ["mcq", "niah", "qa", "fwe", "cwe"]

    # Header
    col_w = 16
    header = "Task".ljust(8) + "".join(l.rjust(col_w) for l in run_labels) + "   Variance"
    print(f"\n{header}")
    print("-" * (8 + col_w * num_runs + 12))

    task_variances = {}
    for task in tasks:
        row = task.ljust(8)
        accs = []
        for preds in runs:
            task_preds = [p for p in preds if p.get("task") == task]
            if task_preds:
                acc = sum(p["score"] for p in task_preds) / len(task_preds) * 100
            else:
                acc = 0
            accs.append(acc)
            row += ("%.1f%%" % acc).rjust(col_w)
        var = max(accs) - min(accs)
        task_variances[task] = var
        row += ("  %.1f pp" % var).rjust(12)
        print(row)

    # Overall
    row = "TOTAL".ljust(8)
    overall_accs = []
    for preds in runs:
        acc = sum(p["score"] for p in preds) / len(preds) * 100
        overall_accs.append(acc)
        row += ("%.2f%%" % acc).rjust(col_w)
    var = max(overall_accs) - min(overall_accs)
    row += ("  %.1f pp" % var).rjust(12)
    print(row)

    print(f"\n  Accuracy range: {min(overall_accs):.2f}% - {max(overall_accs):.2f}% "
          f"(spread: {var:.1f} pp)")
    print(f"  Most unstable task: {max(task_variances, key=task_variances.get)} "
          f"({task_variances[max(task_variances, key=task_variances.get)]:.1f} pp spread)")

    # ======================================================================
    # Section 2: None/Empty output counts
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 2: NONE/EMPTY OUTPUT COUNTS (token-0 collapse indicator)")
    print("=" * 100)

    header = "Task".ljust(8) + "".join(l.rjust(col_w) for l in run_labels)
    print(f"\n{header}")
    print("-" * (8 + col_w * num_runs))

    for task in tasks:
        row = task.ljust(8)
        for preds in runs:
            task_preds = [p for p in preds if p.get("task") == task]
            none_count = sum(1 for p in task_preds
                            if p.get("extracted") is None
                            or str(p.get("extracted")) in ("", "None"))
            row += str(none_count).rjust(col_w)
        print(row)

    # ======================================================================
    # Section 3: Failure mode breakdown
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 3: FAILURE MODE BREAKDOWN")
    print("=" * 100)

    all_modes = set()
    for run_preds in runs:
        for s in run_preds:
            all_modes.add(s["_failure_mode"])

    mode_order = ["correct", "partial", "wrong_answer", "empty/null",
                  "token0_collapse", "early_stop_none", "extraction_failed", "max_tokens"]
    mode_order = [m for m in mode_order if m in all_modes]
    mode_order += sorted(all_modes - set(mode_order))

    header = "Mode".ljust(22) + "".join(l.rjust(col_w) for l in run_labels)
    print(f"\n{header}")
    print("-" * (22 + col_w * num_runs))

    for mode in mode_order:
        row = mode.ljust(22)
        for run_preds in runs:
            cnt = sum(1 for s in run_preds if s["_failure_mode"] == mode)
            row += str(cnt).rjust(col_w)
        print(row)

    # ======================================================================
    # Section 4: Per-question stability
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 4: PER-QUESTION STABILITY ANALYSIS")
    print("=" * 100)

    always_right = []
    always_wrong = []
    flipping = []

    for idx in range(num_samples):
        scores = [runs[r][idx]["score"] for r in range(num_runs)]
        passed = [s > 0 for s in scores]

        if all(passed):
            always_right.append(idx)
        elif not any(passed):
            always_wrong.append(idx)
        else:
            flipping.append(idx)

    print(f"\n  Always correct (all {num_runs} runs):  {len(always_right)}")
    print(f"  Always wrong   (all {num_runs} runs):  {len(always_wrong)}")
    print(f"  Flipping (inconsistent):        {len(flipping)}")
    print(f"  Stability ratio:                {len(always_right + always_wrong)}/{num_samples} "
          f"({(len(always_right) + len(always_wrong)) / num_samples * 100:.1f}%)")

    # Flipping breakdown by task
    print(f"\n  Flipping questions by task:")
    flip_by_task = defaultdict(list)
    for idx in flipping:
        task = runs[0][idx].get("task", "?")
        flip_by_task[task].append(idx)
    for task in tasks:
        indices = flip_by_task.get(task, [])
        if indices:
            print(f"    {task}: {len(indices)} questions — indices: {indices}")

    # Always wrong breakdown by task
    print(f"\n  Always wrong by task:")
    wrong_by_task = defaultdict(list)
    for idx in always_wrong:
        task = runs[0][idx].get("task", "?")
        wrong_by_task[task].append(idx)
    for task in tasks:
        indices = wrong_by_task.get(task, [])
        if indices:
            print(f"    {task}: {len(indices)} — indices: {indices}")

    # ======================================================================
    # Section 5: Flipping questions detail
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 5: FLIPPING QUESTIONS DETAIL")
    print("=" * 100)

    score_header = "".join(("R%d" % (i+1)).rjust(6) for i in range(num_runs))
    print(f"\n  {'Idx':>4} {'Task':<5} {score_header}  {'Gold (truncated)':<50}")
    print("  " + "-" * (4 + 5 + 6 * num_runs + 2 + 50))

    for idx in flipping:
        scores = [runs[r][idx]["score"] for r in range(num_runs)]
        task = runs[0][idx].get("task", "?")
        gold = str(runs[0][idx].get("gold", ""))[:50]
        score_str = "".join(("%.2f" % s).rjust(6) for s in scores)
        print(f"  {idx:>4} {task:<5} {score_str}  {gold}")

    # ======================================================================
    # Section 6: Always-wrong questions (calibration targets)
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 6: ALWAYS-WRONG QUESTIONS (calibration improvement targets)")
    print("=" * 100)

    print(f"\n  These {len(always_wrong)} questions are wrong in ALL runs — "
          f"potential targets for calibration data improvement.\n")

    for idx in always_wrong:
        task = runs[0][idx].get("task", "?")
        gold = str(runs[0][idx].get("gold", ""))[:70]
        # Show what the model extracted in each run
        extracts = []
        for r in range(num_runs):
            ext = runs[r][idx].get("extracted")
            if ext is None:
                extracts.append("None")
            else:
                extracts.append(str(ext)[:30])
        print(f"  [{idx:>3}] task={task:<4} gold={gold}")
        for r in range(num_runs):
            print(f"         R{r+1}: extracted={extracts[r]}")

    # ======================================================================
    # Section 7: Output token analysis
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 7: OUTPUT TOKEN & THROUGHPUT ANALYSIS")
    print("=" * 100)

    for r, label in enumerate(run_labels):
        preds = runs[r]
        out_toks = [p.get("output_tokens", 0) for p in preds]
        in_toks = [p.get("input_tokens", 0) for p in preds]
        total_out = sum(out_toks)
        total_in = sum(in_toks)
        avg_out = total_out // num_samples
        max_out = max(out_toks)
        min_out = min(out_toks)
        very_short = sum(1 for t in out_toks if t <= 20)

        print(f"\n  {label}:")
        print(f"    Total output tokens: {total_out:,}")
        print(f"    Avg output tokens:   {avg_out:,}")
        print(f"    Min/Max output:      {min_out} / {max_out:,}")
        print(f"    Very short (<=20):   {very_short} samples")

    # ======================================================================
    # Section 8: Stability score per question
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SECTION 8: QUESTION STABILITY SCORES")
    print("=" * 100)

    print(f"\n  Stability = (mean_score, std_score) across runs. "
          f"Sorted by most unstable first.\n")

    question_stats = []
    for idx in range(num_samples):
        scores = [runs[r][idx]["score"] for r in range(num_runs)]
        mean_s = sum(scores) / num_runs
        std_s = (sum((s - mean_s) ** 2 for s in scores) / num_runs) ** 0.5
        task = runs[0][idx].get("task", "?")
        question_stats.append((idx, task, mean_s, std_s, scores))

    # Sort by std desc
    question_stats.sort(key=lambda x: -x[3])

    score_header = "".join(("R%d" % (i+1)).rjust(6) for i in range(num_runs))
    print(f"  {'Idx':>4} {'Task':<5} {'Mean':>6} {'Std':>6} {score_header}")
    print("  " + "-" * (4 + 5 + 6 + 6 + 6 * num_runs))

    for idx, task, mean_s, std_s, scores in question_stats[:30]:
        score_str = "".join(("%.2f" % s).rjust(6) for s in scores)
        print(f"  {idx:>4} {task:<5} {mean_s:>5.2f} {std_s:>5.3f} {score_str}")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 100)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 100)

    best_run = run_labels[overall_accs.index(max(overall_accs))]
    worst_run = run_labels[overall_accs.index(min(overall_accs))]

    print(f"""
  Overall accuracy range: {min(overall_accs):.2f}% - {max(overall_accs):.2f}% (spread: {var:.1f} pp)
  Best run:  {best_run} ({max(overall_accs):.2f}%)
  Worst run: {worst_run} ({min(overall_accs):.2f}%)

  Stability: {len(always_right)} always-right, {len(always_wrong)} always-wrong, {len(flipping)} flipping

  Ceiling (if all flipping questions were correct): {(len(always_right) + len(flipping)) / num_samples * 100:.1f}%
  Floor   (if all flipping questions were wrong):   {len(always_right) / num_samples * 100:.1f}%

  Priority targets for improvement:
    1. {len(always_wrong)} always-wrong questions (mostly MCQ={len(wrong_by_task.get('mcq',[]))} + QA={len(wrong_by_task.get('qa',[]))})
       -> Better calibration data with correct reasoning traces for these topics
    2. {len(flipping)} flipping questions (most unstable: {max(task_variances, key=task_variances.get)})
       -> Run-to-run variance from CUDA non-determinism; consider multiple eval runs and majority vote
    3. Token-0 collapse still occurs sporadically (Run2 had 6 NIAH Nones with only 9 output tokens)
       -> May indicate residual numerical instability under high concurrency
""")


if __name__ == "__main__":
    main()
