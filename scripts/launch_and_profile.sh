#!/bin/bash
# Profile decode kernels inside the server process (under CUDA graph).
#
# Usage: bash /opt/oldMoney-Project/scripts/launch_and_profile.sh
#
# After server is ready, send a long-context request:
#   python3 /opt/oldMoney-Project/scripts/profile_decode.py
#
# Results will appear in /opt/decode_profile.txt after ~250 decode steps.

set -e

fuser -k -9 31333/tcp 2>/dev/null || true
sleep 2

uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_cp

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DECODE_PROFILE=1

echo "Starting server with decode profiler enabled..."
nohup python3 -m sglang.launch_server \
    --model-path /opt/model_gptq_int4_dense_smooth \
    --port 31333 \
    --quantization gptq_marlin \
    --kv-cache-dtype fp8_e5m2 \
    --dtype bfloat16 \
    --disable-radix-cache \
    --max-running-requests 64 \
    --attention-backend flashinfer \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.82 \
    --max-mamba-cache-size 64 \
    --log-level info \
    > /opt/server.log 2>&1 &

echo "Waiting 90s for server startup..."
sleep 90

echo "Sending long-context request to trigger decode..."
python3 /opt/oldMoney-Project/scripts/profile_decode.py

echo ""
echo "Waiting 30s for profiler to finish (50 warmup + 200 profile steps)..."
sleep 30

echo ""
echo "=== RESULTS ==="
if [ -f /opt/decode_profile.txt ]; then
    cat /opt/decode_profile.txt
else
    echo "Profile not yet written. Check: grep PROFILER /opt/server.log"
    grep -i profiler /opt/server.log || true
fi
