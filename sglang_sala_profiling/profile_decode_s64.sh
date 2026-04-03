#!/bin/bash
# Profile decode internals at 64 concurrency
# Must disable CUDA graph to see per-component timing inside decode

fuser -k -9 31333/tcp
sleep 2

uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_profiling

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_SALA_PROFILE=1

nohup python3 -m sglang.launch_server \
    --model-path /opt/model_gptq_int4_dense_smooth \
    --port 31333 \
    --quantization gptq_marlin \
    --kv-cache-dtype fp8_e5m2 \
    --dtype bfloat16 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --max-running-requests 64 \
    --attention-backend flashinfer \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.82 \
    --max-mamba-cache-size 64 \
    --log-level info \
    > /opt/server_profile_decode.log 2>&1 &

echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:31333/model_info > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done
