import json
import random

# File paths
ultimate_file = "/opt/oldMoney-Project/quantization/ultimate_64_token_balanced.jsonl"
synth_file = "/opt/oldMoney-Project/quantization/calibration/calib_dataset.jsonl"
out_file = "/opt/oldMoney-Project/quantization/calibration/final_merged_calibration.jsonl"

final_dataset = []

# 1. Load all 64 from the ultimate balanced dataset
try:
    with open(ultimate_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                final_dataset.append(json.loads(line.strip()))
    print(f"[+] Loaded {len(final_dataset)} samples from ultimate_64_token_balanced.jsonl")
except Exception as e:
    print(f"[-] Error loading ultimate dataset: {e}")

# 2. Load and categorize the synthesized dataset
wiki_samples = []
math_samples = []

with open(synth_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line.strip())
        text = data.get("question", "")
        
        # Categorize based on the prompt structure injected by build_calib.py
        if "summarize the core concepts" in text:
            wiki_samples.append(data)
        elif "think step-by-step" in text:
            math_samples.append(data)
        else:
            # Fallback
            wiki_samples.append(data)

print(f"[i] Found {len(wiki_samples)} Wikipedia samples and {len(math_samples)} Math samples in calib_dataset.jsonl")

# 3. Choose the 32 most suitable samples (16 Wiki, 16 Math)
# Setting a fixed seed ensures reproducibility if you run this multiple times
random.seed(42) 

selected_wiki = random.sample(wiki_samples, min(16, len(wiki_samples)))
selected_math = random.sample(math_samples, min(16, len(math_samples)))

print(f"[+] Selected {len(selected_wiki)} Wiki samples (for continuous language modeling)")
print(f"[+] Selected {len(selected_math)} Math samples (for logic and strict formatting)")

# 4. Merge and Shuffle
final_dataset.extend(selected_wiki)
final_dataset.extend(selected_math)

# Shuffle so the quantization observer processes a healthy mix of task types
random.shuffle(final_dataset)

# 5. Save the final merged dataset
with open(out_file, 'w', encoding='utf-8') as f:
    for item in final_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n[SUCCESS] Saved {len(final_dataset)} total samples to: {out_file}")
