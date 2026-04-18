#!/usr/bin/env python3
import requests, time

payload = {
    'model': 'openbmb/MiniCPM-SALA',
    'messages': [{'role': 'user', 'content': 'A ' * 200000 + 'What is 2+2?'}],
    'max_tokens': 16,
    'temperature': 0,
}

for name, port in [('Sparse', 31335), ('Dense', 31333)]:
    t0 = time.time()
    r = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=payload, timeout=600)
    elapsed = time.time() - t0
    d = r.json()
    print(f'{name} ({port}): {elapsed:.2f}s, prompt_tokens={d["usage"]["prompt_tokens"]}')
    print(f'  output: {d["choices"][0]["message"]["content"][:80]}')
    print()

