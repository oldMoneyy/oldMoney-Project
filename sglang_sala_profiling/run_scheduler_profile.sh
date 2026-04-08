#!/bin/bash
# =============================================================================
# Scheduler Profiling: measures where wall-clock time goes
# Run on remote server with the competition benchmark
# =============================================================================
set -e

# --- Config ---
PORT=31333
MODEL_PATH=/opt/model_gptq_int4_dense_smooth
BENCH_DATA=/opt/oldMoney-Project/bench/competition_bench_64.jsonl
PROFILE_OUT=/tmp/scheduler_profile.jsonl
SERVER_LOG=/opt/server_profile.log

echo "=== Step 1: Install sglang_sala_flashinfer ==="
uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_flashinfer

echo "=== Step 2: Kill existing server ==="
fuser -k -9 ${PORT}/tcp 2>/dev/null || true
sleep 2

echo "=== Step 3: Launch server with scheduler profiler ==="
rm -f ${PROFILE_OUT}

export SGLANG_SCHEDULER_PROFILE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --port ${PORT} \
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
    > ${SERVER_LOG} 2>&1 &

echo "Waiting for server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server failed to start. Check ${SERVER_LOG}"
        exit 1
    fi
    sleep 1
done

echo "=== Step 4: Run benchmark (Smax) ==="
export SPEED_DATA_SMAX=${BENCH_DATA}
bash /opt/oldMoney-Project/SOAR-Toolkit/bench_serving.sh http://127.0.0.1:${PORT}

echo "=== Step 5: Analyze results ==="
python3 /opt/oldMoney-Project/sglang_sala_profiling/analyze_scheduler_profile.py ${PROFILE_OUT}

echo ""
echo "Raw data: ${PROFILE_OUT}"
echo "Server log: ${SERVER_LOG}"
echo "Done."
