import requests
import time

# Define the tests
tests = []

# Test 1: Short context (Prefill heavy)
text1 = 'The capital of France is Paris. ' * 2500
text1 += 'Based on the text above, what is the capital of France? Answer in one word.'
tests.append(("Test 1 - Capital of France (Prefill Focus)", text1))

# Test 2: Needle-in-haystack (Prefill heavy)
hay = 'The weather in London is often rainy and cloudy during winter months. ' * 5000
needle = 'The secret code is BLUE-TIGER-42. '
text2 = hay[:len(hay)//2] + needle + hay[len(hay)//2:]
text2 += 'What is the secret code mentioned in the text above?'
tests.append(("Test 2 - Needle in Haystack (Prefill Focus)", text2))

# Test 3: Multi-document (Prefill heavy)
parts = [f'Document {i}: The annual rainfall in region {i} averages {100+i}mm per year.' for i in range(8000)]
parts[2000] = 'Document 2000: The president of Greenfield Corp is Alice Zhang.'
parts[5000] = 'Document 5000: The founding year of Greenfield Corp is 1987.'
text3 = ' '.join(parts)
text3 += ' Question: Who is the president of Greenfield Corp and when was it founded? Answer clearly.'
tests.append(("Test 3 - Multi-document (Prefill Focus)", text3))

# ==========================================
# 🌟 NEW: Test 4: Long Generation (Decode heavy)
# Prompt is very short, forcing the model to spend 99% of its time generating tokens.
# ==========================================
text4 = (
    "Write an extremely detailed, comprehensive, and exhaustive article about the history of human space exploration. "
    "You must cover early astronomy, the Space Race, the Apollo missions, the Space Shuttle era, the International Space Station, "
    "and future Mars colonization. "
    "Make sure the article is extremely long, containing at least 30 deep and detailed paragraphs. "
    "Do not summarize; expand on every single detail you can think of. Keep writing until you have covered absolutely everything."
)
tests.append(("Test 4 - Long Generation (Decode Focus)", text4))

results = []  # will store full answers

print("🚀 Starting long-context benchmark with timing...\n")

for test_name, prompt in tests:
    prompt_chars = len(prompt)
    est_tokens = prompt_chars // 4

    print(f"Running {test_name} (~{prompt_chars:,} chars / ~{est_tokens:,} tokens)")

    start_time = time.perf_counter()

    try:
        r = requests.post(
            'http://localhost:31333/v1/chat/completions',
            json={
                'model': 'MiniCPM-SALA',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 4096,  # 允许它生成极长的文本
                'temperature': 0.0
            },
            timeout=None  # ⚠️ 极其重要：改为 None 永不超时，防止 Decode 生成时间超过 5 分钟被强行切断
        )

        duration = time.perf_counter() - start_time

        if r.status_code == 200:
            data = r.json()
            content = data['choices'][0]['message']['content']

            if '</think>' in content:
                answer = content.split('</think>')[-1].strip()
            else:
                answer = content.strip()

            usage = data.get('usage', {})
            prompt_t = usage.get('prompt_tokens', 'N/A')
            comp_t = usage.get('completion_tokens', len(content)//4)
            
            # TPS 计算
            tps = round(comp_t / duration, 1) if duration > 0 else 0

            print(f"   ✅ Completed")
            print(f"      Duration        : {duration:.2f} seconds")
            print(f"      Tokens/sec      : {tps} t/s")
            print(f"      Response length : {len(content):,} chars")
            if prompt_t != 'N/A':
                print(f"      Tokens used     : {prompt_t} prompt + {comp_t} completion")
            print(f"      Preview         : {answer[:300].replace(chr(10), ' ')}{'...' if len(answer) > 300 else ''}\n")

            # Store for full print at the end
            results.append((test_name, answer, duration, tps))
        else:
            print(f"   ❌ ERROR {r.status_code} - {r.text[:200]}\n")
            results.append((test_name, "ERROR", duration, 0))

    except Exception as e:
        print(f"   ❌ EXCEPTION: {str(e)}\n")
        results.append((test_name, f"EXCEPTION: {str(e)}", 0, 0))

print("✅ Benchmark finished!")

# ======================== FULL COMPLETE ANSWERS ========================
print("\n" + "="*90)
print("📋 COMPLETE FULL ANSWERS (no truncation)")
print("="*90 + "\n")

for test_name, full_answer, dur, tps in results:
    print(f"🔹 {test_name}")
    print(f"   ⏱ Duration : {dur:.2f} seconds | {tps} tokens/sec")
    print(f"   📝 Answer   :\n{full_answer}")
    print("-" * 80 + "\n")