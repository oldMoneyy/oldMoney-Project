#!/usr/bin/env python3
"""
Inspect the actual model outputs for failed samples.
This will tell us WHY extraction fails - is the model:
  (a) generating garbage/repetition?
  (b) generating correct answers in wrong format?
  (c) generating nothing?
  (d) generating in wrong language?

Usage:
  python inspect_failures.py --predictions /opt/SOAR-Toolkit/outputs/20260321_163020/predictions.jsonl
"""
import json
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--eval-data", default="/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl")
    args = parser.parse_args()

    preds = []
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line.strip()))

    # Also try to load eval data to see the prompts
    eval_data = []
    try:
        with open(args.eval_data) as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line.strip()))
    except:
        pass

    print("=" * 80)
    print("RAW OUTPUT INSPECTION FOR FAILED SAMPLES")
    print("=" * 80)

    # First: what fields exist in predictions?
    if preds:
        print(f"\n[DEBUG] Fields in predictions.jsonl: {list(preds[0].keys())}")
        print(f"[DEBUG] Sample 0 preview:")
        for k, v in preds[0].items():
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "..."
            print(f"  {k}: {v_str}")

    # Inspect MCQ failures (the worst: 43.3%)
    print("\n" + "=" * 80)
    print("MCQ FAILURES (43.3% accuracy - 17 failures)")
    print("=" * 80)

    mcq_count = 0
    for i, p in enumerate(preds):
        task = p.get("task", p.get("Task", ""))
        score = p.get("score", p.get("Score", 1))
        if task.lower() != "mcq" or score >= 0.5:
            continue
        mcq_count += 1
        if mcq_count > 8:  # Show first 8 failures
            continue

        print(f"\n--- MCQ Sample {i+1} (score={score}) ---")
        
        # Show gold answer
        gold = p.get("gold", p.get("Gold", "?"))
        extracted = p.get("extracted", p.get("Extracted", p.get("prediction", "?")))
        print(f"  Gold: {gold}")
        print(f"  Extracted: {extracted}")
        
        # Show the actual model output (try various field names)
        output = None
        for key in ["output", "response", "generated", "completion", "model_output", 
                     "pred", "prediction", "answer", "text", "content", "raw_output"]:
            if key in p:
                output = p[key]
                break
        
        if output is not None:
            out_str = str(output)
            if len(out_str) > 1000:
                print(f"  Output (first 500 chars): {out_str[:500]}")
                print(f"  Output (last 500 chars):  {out_str[-500:]}")
                print(f"  Output length: {len(out_str)} chars")
            else:
                print(f"  Output: {out_str}")
        else:
            print(f"  [NO OUTPUT FIELD FOUND - available fields: {list(p.keys())}]")

        # Show the prompt if available
        prompt = None
        for key in ["prompt", "input", "question", "query"]:
            if key in p:
                prompt = p[key]
                break
        if prompt:
            prompt_str = str(prompt)
            if len(prompt_str) > 500:
                print(f"  Prompt (first 300): {prompt_str[:300]}...")
            else:
                print(f"  Prompt: {prompt_str}")

    # Inspect QA failures (50% accuracy - 15 failures)
    print("\n" + "=" * 80)
    print("QA FAILURES (50% accuracy - 15 failures)")
    print("=" * 80)

    qa_count = 0
    for i, p in enumerate(preds):
        task = p.get("task", p.get("Task", ""))
        score = p.get("score", p.get("Score", 1))
        if task.lower() != "qa" or score >= 0.5:
            continue
        qa_count += 1
        if qa_count > 5:  # Show first 5
            continue

        print(f"\n--- QA Sample {i+1} (score={score}) ---")
        gold = p.get("gold", p.get("Gold", "?"))
        extracted = p.get("extracted", p.get("Extracted", p.get("prediction", "?")))
        print(f"  Gold: {str(gold)[:200]}")
        print(f"  Extracted: {str(extracted)[:200]}")

        output = None
        for key in ["output", "response", "generated", "completion", "model_output",
                     "pred", "prediction", "answer", "text", "content", "raw_output"]:
            if key in p:
                output = p[key]
                break
        if output is not None:
            out_str = str(output)
            if len(out_str) > 1000:
                print(f"  Output (first 500): {out_str[:500]}")
                print(f"  Output (last 500):  {out_str[-500:]}")
            else:
                print(f"  Output: {out_str}")
        else:
            print(f"  [NO OUTPUT FIELD FOUND - fields: {list(p.keys())}]")

    # Also inspect SUCCESSFUL samples to see what the expected format looks like
    print("\n" + "=" * 80)
    print("SUCCESSFUL MCQ SAMPLES (for format comparison)")
    print("=" * 80)

    success_count = 0
    for i, p in enumerate(preds):
        task = p.get("task", p.get("Task", ""))
        score = p.get("score", p.get("Score", 0))
        if task.lower() != "mcq" or score < 0.5:
            continue
        success_count += 1
        if success_count > 3:
            continue

        print(f"\n--- MCQ Sample {i+1} (score={score}) ---")
        gold = p.get("gold", p.get("Gold", "?"))
        extracted = p.get("extracted", p.get("Extracted", "?"))
        print(f"  Gold: {gold}")
        print(f"  Extracted: {extracted}")

        output = None
        for key in ["output", "response", "generated", "completion", "model_output",
                     "pred", "prediction", "answer", "text", "content", "raw_output"]:
            if key in p:
                output = p[key]
                break
        if output is not None:
            out_str = str(output)
            print(f"  Output (first 500): {out_str[:500]}")

    # TOKEN ANALYSIS: Check if MCQ outputs are abnormally long/short
    print("\n" + "=" * 80)
    print("OUTPUT LENGTH ANALYSIS BY TASK AND SUCCESS")
    print("=" * 80)

    for task_name in ["mcq", "qa", "niah", "fwe", "cwe"]:
        success_lens = []
        fail_lens = []
        for p in preds:
            task = p.get("task", p.get("Task", "")).lower()
            if task != task_name:
                continue
            score = p.get("score", p.get("Score", 0))
            
            # Try to get output length
            output = None
            for key in ["output", "response", "generated", "completion", "model_output",
                         "pred", "prediction", "text", "content", "raw_output"]:
                if key in p:
                    output = str(p[key])
                    break
            
            out_len = len(output) if output else 0
            
            # Also check tokens_in/tokens_out fields
            tokens_out = p.get("tokens_out", p.get("Tokens_out", p.get("out_tokens", 0)))
            tokens_in = p.get("tokens_in", p.get("Tokens_in", p.get("in_tokens", 0)))
            
            entry = {"chars": out_len, "tokens_out": tokens_out, "tokens_in": tokens_in}
            
            if score >= 0.5:
                success_lens.append(entry)
            else:
                fail_lens.append(entry)
        
        if success_lens or fail_lens:
            print(f"\n  {task_name.upper()}:")
            if success_lens:
                avg_chars = sum(e["chars"] for e in success_lens) / len(success_lens)
                avg_tout = sum(e["tokens_out"] for e in success_lens) / len(success_lens)
                avg_tin = sum(e["tokens_in"] for e in success_lens) / len(success_lens)
                print(f"    Success ({len(success_lens)}): avg {avg_chars:.0f} chars, {avg_tout:.0f} out_tokens, {avg_tin:.0f} in_tokens")
            if fail_lens:
                avg_chars = sum(e["chars"] for e in fail_lens) / len(fail_lens)
                avg_tout = sum(e["tokens_out"] for e in fail_lens) / len(fail_lens)
                avg_tin = sum(e["tokens_in"] for e in fail_lens) / len(fail_lens)
                print(f"    Fail ({len(fail_lens)}):    avg {avg_chars:.0f} chars, {avg_tout:.0f} out_tokens, {avg_tin:.0f} in_tokens")

    # COMPARISON WITH BF16 BASELINE
    print("\n" + "=" * 80)
    print("KEY QUESTION: Does the BF16 model get MCQ right?")
    print("=" * 80)
    print("If BF16 MCQ accuracy is also low (~50%), then MCQ damage is NOT from quantization.")
    print("If BF16 MCQ accuracy is high (~80%), then quantization damaged instruction following.")
    print("\nPlease check your BF16 baseline results for MCQ specifically.")

if __name__ == "__main__":
    main()
