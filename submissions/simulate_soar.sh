#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash simulate_soar.sh <your_submission.tar.gz>"
    exit 1
fi

TAR_FILE=$1
if [ ! -f "$TAR_FILE" ]; then
    echo "Error: File $TAR_FILE not found!"
    exit 1
fi

echo -e "\n======================================================="
echo "=== 1. Extracting Tarball ==="
echo "======================================================="
rm -rf /tmp/soar_workspace /tmp/model_nvfp4_awq_v2
mkdir -p /tmp/soar_workspace
tar -xzvf "$TAR_FILE" -C /tmp/soar_workspace
cd /tmp/soar_workspace

echo -e "\n======================================================="
echo "=== 2. Running prepare_env.sh ==="
echo "======================================================="
source prepare_env.sh 2>&1 | tee /tmp/simulate_env.log

echo -e "\n======================================================="
echo "=== 3. Running prepare_model.sh ==="
echo "======================================================="
bash prepare_model.sh --input /opt/model --output /tmp/model_nvfp4_awq_v2 2>&1 | tee /tmp/simulate_model.log

echo -e "\n======================================================="
echo "=== 4. Launching SGLang Server ==="
echo "======================================================="
fuser -k -9 31333/tcp || true
pkill -9 -f sglang || true

echo "Server Args: $SGLANG_SERVER_ARGS"
echo "-------------------------------------------------------"

nohup python3 -m sglang.launch_server \
    --model-path /tmp/model_nvfp4_awq_v2 \
    --host 127.0.0.1 \
    --port 31333 \
    $SGLANG_SERVER_ARGS > /tmp/soar_server.log 2>&1 &

echo -e "\nServer launched in background. Fake Torch Compile is active."
echo "Waiting for health check at http://127.0.0.1:31333/health ..."

while ! curl -s http://127.0.0.1:31333/health > /dev/null; do
    sleep 5
done

echo -e "\n======================================================="
echo "=== 5. Running Benchmark ==="
echo "======================================================="
echo "=== Smax (unlimited) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench.jsonl \
    --num-prompts 64 --flush-cache

echo -e "\n======================================================="
echo "Test Finished! Server logs are saved in /tmp/soar_server.log"
