#!/bin/bash
# Launch server + profile decode kernels under CUDA graph
# Usage: bash /opt/oldMoney-Project/scripts/launch_and_profile.sh

set -e

fuser -k -9 31333/tcp 2>/dev/null || true
sleep 2

uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_cp

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting server..."
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

echo "Running profiler..."
python3 /opt/oldMoney-Project/scripts/profile_decode.py

echo ""
echo "=== Results ==="
cat /opt/profile_results.txt
