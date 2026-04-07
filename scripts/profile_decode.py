"""Profile decode kernels under CUDA graph using PyTorch profiler.

Usage:
  1. Launch server normally (see launch_and_profile.sh)
  2. Wait for server ready
  3. python3 /opt/oldMoney-Project/scripts/profile_decode.py

Output:
  /opt/profile_results.txt  -- kernel time summary (top 50 by CUDA time)
  /opt/trace.json           -- chrome trace (open in chrome://tracing)
"""
import json, requests, time, threading

def send_request():
    with open("/opt/oldMoney-Project/bench/competition_bench_64.jsonl") as f:
        first = json.loads(f.readline())
    messages = [{"role": c["role"], "content": c["content"][:50000]} for c in first["conversations"]]
    try:
        requests.post("http://127.0.0.1:31333/v1/chat/completions", json={
            "model": "default", "messages": messages, "max_tokens": 300
        }, timeout=300)
    except:
        pass

# Wait for server
print("Waiting for server...")
for i in range(60):
    try:
        r = requests.get("http://127.0.0.1:31333/server_info", timeout=2)
        if r.status_code == 200:
            print("Server ready")
            break
    except:
        time.sleep(2)
else:
    print("Server not ready after 120s, proceeding anyway")

# Start request
print("Sending long-context request...")
t = threading.Thread(target=send_request, daemon=True)
t.start()

# Wait for prefill to finish and decode to start
print("Waiting 30s for prefill to finish...")
time.sleep(30)

# Profile 5 seconds of decode
print("Starting profiler (5 seconds)...")
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
    time.sleep(5)

print("Writing results...")
table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
print(table)

with open("/opt/profile_results.txt", "w") as f:
    f.write(table)

prof.export_chrome_trace("/opt/trace.json")
print("Done. Results in /opt/profile_results.txt and /opt/trace.json")
