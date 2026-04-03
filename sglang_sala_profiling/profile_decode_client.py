"""
Send concurrent requests to profile decode phase.
Uses medium-length inputs with long outputs to maximize decode time.
Run after server is up with --disable-cuda-graph and SGLANG_SALA_PROFILE=1.

Usage: python3 profile_decode_client.py [num_requests] [api_base]
"""
import asyncio
import aiohttp
import json
import sys
import time

API_BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:31333"
NUM_REQUESTS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

INPUT_CONFIGS = [
    (4000, 512),
    (8000, 1024),
    (16000, 512),
    (32000, 256),
    (64000, 128),
    (4000, 2048),
    (8000, 512),
    (16000, 1024),
    (32000, 512),
    (64000, 256),
    (4000, 512),
    (8000, 1024),
    (16000, 512),
    (32000, 256),
    (64000, 128),
    (4000, 2048),
]


def make_prompt(approx_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    repeat_count = max(1, (approx_tokens * 4) // len(base))
    text = base * repeat_count
    return f"Please summarize the following text and provide analysis:\n\n{text}\n\nSummary:"


async def send_request(session, idx, input_tokens, max_output):
    prompt = make_prompt(input_tokens)
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_output,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        async with session.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            result = await resp.json()
            elapsed = time.time() - t0
            out_tokens = result.get("usage", {}).get("completion_tokens", "?")
            print(f"  req[{idx:2d}] input~{input_tokens:>6d} -> output={out_tokens} in {elapsed:.1f}s")
            return result
    except Exception as e:
        print(f"  req[{idx:2d}] FAILED: {e}")
        return None


async def main():
    print(f"Sending {NUM_REQUESTS} concurrent requests to {API_BASE}")
    print()

    configs = [INPUT_CONFIGS[i % len(INPUT_CONFIGS)] for i in range(NUM_REQUESTS)]

    async with aiohttp.ClientSession() as session:
        t_start = time.time()
        tasks = [
            send_request(session, i, cfg[0], cfg[1])
            for i, cfg in enumerate(configs)
        ]
        results = await asyncio.gather(*tasks)
        t_total = time.time() - t_start

    success = sum(1 for r in results if r is not None)
    print(f"\nDone: {success}/{NUM_REQUESTS} succeeded in {t_total:.1f}s")
    print(f"\nCheck server log:")
    print(f"  cat /opt/server_profile_decode.log | grep -E 'Decode|decode|sparse|lightning|mlp'")


if __name__ == "__main__":
    asyncio.run(main())
