"""Analyze eval predictions to separate token-0 collapse from real accuracy issues."""
import json
import sys
import os

# Find the latest output dir
base = '/opt/SOAR-Toolkit/outputs'
if len(sys.argv) > 1:
    pred_path = sys.argv[1]
else:
    pred_path = os.path.join(base, '20260327_114100', 'predictions.jsonl')

with open(pred_path) as f:
    preds = [json.loads(l) for l in f]

print(f"Total predictions: {len(preds)}")

# Categorize results
none_content = []
empty_content = []
correct = []
wrong = []
unknown = []

for p in preds:
    idx = p.get('index', '?')
    content = p.get('response', p.get('output', p.get('content', p.get('prediction', ''))))
    gold = p.get('gold', p.get('answer', p.get('expected', '?')))
    score = p.get('score', p.get('correct', None))
    task = p.get('task', '?')

    if content is None or content == '':
        none_content.append(p)
    elif score == 1 or score == True or score == 1.0:
        correct.append(p)
    elif score == 0 or score == False or score == 0.0:
        wrong.append(p)
    else:
        unknown.append(p)

print(f"\n--- Breakdown ---")
print(f"  Correct:        {len(correct)}")
print(f"  Wrong:          {len(wrong)}")
print(f"  None/Empty:     {len(none_content)}")
print(f"  Unknown score:  {len(unknown)}")

if none_content:
    print(f"\n--- None/Empty content samples (TOKEN-0 COLLAPSE?) ---")
    for p in none_content:
        idx = p.get('index', '?')
        task = p.get('task', '?')
        print(f"  idx={idx} task={task}")

# Breakdown by task type
print(f"\n--- Score by task type ---")
tasks = {}
for p in preds:
    task = p.get('task', '?')
    score = p.get('score', p.get('correct', 0))
    if task not in tasks:
        tasks[task] = {'total': 0, 'correct': 0, 'none': 0}
    tasks[task]['total'] += 1
    if score == 1 or score == True or score == 1.0:
        tasks[task]['correct'] += 1
    content = p.get('response', p.get('output', p.get('content', p.get('prediction', ''))))
    if content is None or content == '':
        tasks[task]['none'] += 1

for task, stats in sorted(tasks.items()):
    pct = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"  {task:>25}: {stats['correct']:>3}/{stats['total']:>3} ({pct:5.1f}%) | none={stats['none']}")

# Show wrong answers with short preview
print(f"\n--- Wrong answers preview (first 20) ---")
for p in (wrong + unknown)[:20]:
    idx = p.get('index', '?')
    task = p.get('task', '?')
    gold = p.get('gold', '?')
    content = p.get('response', p.get('output', p.get('content', p.get('prediction', ''))))
    preview = str(content)[-150:].replace('\n', ' ') if content else 'None'
    print(f"  idx={idx} task={task} gold={str(gold)[:50]} | tail: {preview}")
