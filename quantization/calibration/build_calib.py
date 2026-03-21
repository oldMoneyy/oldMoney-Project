import os
import json
import random
from datasets import load_dataset

def build_cao_yi_dataset():
    # 1. Setup paths
    out_dir = "/opt/oldMoney-Project/quantization/calibration"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "calib_dataset.jsonl")
    
    print(f"Downloading datasets... (This might take a minute)")
    
    # 2. Download high-quality datasets
    # WikiText for pure, continuous semantic grammar (Replacing NIAH/FWE garbage)
    wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # GSM8K for rigid reasoning, logic, and formatting (Simulating QA/MCQ)
    math_data = load_dataset("gsm8k", "main", split="train")

    samples = []
    
    print("Processing and mixing data...")

    # --- TYPE 1: Long Continuous Semantics (Replaces NIAH / FWE) ---
    # We take 80 samples of VERY long, continuous Wikipedia text.
    # This gives the Hessian matrix rich, long-range language patterns without UUID noise.
    wiki_texts = [text for text in wiki_data['text'] if len(text.strip()) > 100]
    
    for _ in range(80):
        # Join ~40 random high-quality paragraphs to make a long continuous prompt
        long_context = "\n\n".join(random.sample(wiki_texts, 40))
        prompt = f"Please read the following text carefully and summarize the core concepts:\n\n{long_context}\n\nSummary:"
        samples.append({"question": prompt})

    # --- TYPE 2: Structured Reasoning / MCQ (Replaces MCQ / QA) ---
    # We take 48 samples of step-by-step reasoning.
    # This preserves the weights responsible for logic and exact-match formatting.
    math_list = list(math_data)
    for row in random.sample(math_list, 48):
        prompt = f"Question: {row['question']}\n\nPlease think step-by-step and provide the final answer."
        samples.append({"question": prompt})

    # 3. Shuffle so GPTQ gets a balanced diet of long-context and reasoning
    random.shuffle(samples)

    # 4. Save exactly how your nvfp4_quantize_sala.py expects it
    print(f"Saving {len(samples)} samples to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for sample in samples:
            # Your script parses: json.loads(line.strip())["question"]
            f.write(json.dumps({"question": sample["question"]}, ensure_ascii=False) + "\n")
            
    print("Done! Calibration dataset built successfully.")
    print(f"Path: {out_file}")

if __name__ == "__main__":
    build_cao_yi_dataset()