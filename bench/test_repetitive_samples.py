"""Test the most repetitive eval samples to see if they trigger token-0 collapse."""
import json
import requests

with open('/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl') as f:
    lines = [json.loads(l) for l in f]

# Most repetitive samples by 0-based index (from analysis)
# idx 58: ratio=0.0006, 127k tok, 82 unique
# idx 41: ratio=0.0013, 63k tok, 82 unique
# idx 50: ratio=0.0017, 127k tok, 213 unique
# idx 34: ratio=0.0026, 31k tok, 81 unique
test_indices = [34, 41, 50, 58]

for idx in test_indices:
    entry = lines[idx]
    question = entry['question']
    prompt_tokens = entry.get('prompt_tokens', '?')
    gold = entry.get('gold', '?')

    print(f"\n{'='*60}")
    print(f"Sample idx={idx} (index={entry['index']}), ~{prompt_tokens} prompt tokens")
    print(f"Gold answer: {gold}")
    print(f"Question preview: {question[:120]}...")
    print(f"{'='*60}")

    r = requests.post('http://localhost:31333/v1/chat/completions',
        json={
            'model': 'MiniCPM-SALA',
            'messages': [{'role': 'user', 'content': question}],
            'max_tokens': 4096,
            'temperature': 0.0
        }, timeout=600)

    data = r.json()
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    finish = data['choices'][0].get('finish_reason')

    if content is None:
        print(f"  RESULT: content=None (TOKEN-0 COLLAPSE)")
    else:
        # Strip think tags
        answer = content.split('</think>')[-1].strip() if '</think>' in content else content.strip()
        print(f"  RESULT: {repr(answer[:200])}")

    print(f"  finish_reason={finish}")
    print(f"  prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}")
    print(f"  MATCH GOLD: {'?' if content is None else (gold.lower() in (content or '').lower())}")
