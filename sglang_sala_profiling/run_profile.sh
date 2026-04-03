#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# MiniCPM-SALA Comprehensive Profiling: flashinfer vs minicpm_flashinfer
# Supports 3 concurrency tiers per SOAR 2026 competition scoring:
#   c1  = --max-concurrent 1   (40% weight, 4 prompts)
#   c8  = --max-concurrent 8   (30% weight, 16 prompts)
#   c64 = unlimited             (30% weight, 64 prompts)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash run_profile.sh all                          # Full: both backends × 3 concurrencies
#   bash run_profile.sh <backend>                    # Full: one backend × 3 concurrencies
#   bash run_profile.sh sala <backend> [c1|c8|c64]   # SALA profiler, one concurrency
#   bash run_profile.sh nsys <backend> [c1|c8|c64]   # Nsight Systems, one concurrency
#   bash run_profile.sh torch <backend> [c1|c8|c64]  # Torch profiler, one concurrency
#   bash run_profile.sh baseline <backend> [c1|c8|c64]
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_PATH="/opt/model_gptq_int4_dense_smooth"
BENCH_DATA="/opt/oldMoney-Project/bench/competition_bench_64.jsonl"
PORT=31333
SGLANG_PKG="/opt/oldMoney-Project/sglang_sala_profiling"
RESULTS_DIR="/opt/oldMoney-Project/sglang_sala_profiling/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Concurrency tier config: tier → (num_prompts, bench_serving_args)
# c1:  single request at a time, 4 prompts total
# c8:  up to 8 concurrent, 16 prompts
# c64: unlimited concurrency, 64 prompts (all at once)
declare -A CONC_PROMPTS=( [c1]=4 [c8]=16 [c64]=64 )
ALL_TIERS=(c1 c8 c64)

# Common server args (--max-running-requests varies per tier)
BASE_SERVER_ARGS=(
    --model-path "$MODEL_PATH"
    --port "$PORT"
    --quantization gptq_marlin
    --kv-cache-dtype fp8_e5m2
    --dtype bfloat16
    --disable-radix-cache
    --chunked-prefill-size 32768
    --mem-fraction-static 0.82
    --max-mamba-cache-size 64
    --log-level info
    --disable-cuda-graph
)

get_server_args() {
    local TIER="$1"
    local args=("${BASE_SERVER_ARGS[@]}")
    case "$TIER" in
        c1)  args+=(--max-running-requests 1) ;;
        c8)  args+=(--max-running-requests 8) ;;
        c64) args+=(--max-running-requests 64) ;;
    esac
    echo "${args[@]}"
}

get_bench_args() {
    local TIER="$1"
    local num_prompts="${CONC_PROMPTS[$TIER]}"
    local args=(
        --backend sglang --host 127.0.0.1 --port "$PORT"
        --dataset-name custom --dataset-path "$BENCH_DATA"
        --num-prompts "$num_prompts" --flush-cache
    )
    # For c1, limit client-side concurrency too
    if [ "$TIER" = "c1" ]; then
        args+=(--max-concurrency 1)
    elif [ "$TIER" = "c8" ]; then
        args+=(--max-concurrency 8)
    fi
    echo "${args[@]}"
}

tier_label() {
    case "$1" in
        c1)  echo "concurrency=1 (4 prompts, 40% weight)" ;;
        c8)  echo "concurrency=8 (16 prompts, 30% weight)" ;;
        c64) echo "concurrency=unlimited (64 prompts, 30% weight)" ;;
    esac
}

wait_for_server() {
    echo "Waiting for server on port $PORT..."
    local max_wait=300
    local waited=0
    while ! curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            tail -50 /opt/server.log
            return 1
        fi
    done
    echo "Server is ready (took ${waited}s)"
}

kill_server() {
    fuser -k -9 ${PORT}/tcp 2>/dev/null || true
    sleep 2
}

# ═══════════════════════════════════════════════════════════════════════════
# Launch server for a given backend + concurrency tier
# ═══════════════════════════════════════════════════════════════════════════
launch_server() {
    local BACKEND="$1"
    local TIER="$2"
    kill_server

    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    local server_args
    read -ra server_args <<< "$(get_server_args "$TIER")"

    nohup python3 -m sglang.launch_server \
        "${server_args[@]}" \
        --attention-backend "$BACKEND" \
        > /opt/server.log 2>&1 &

    wait_for_server
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase: SALA fine-grained profiler (CUDA events + NVTX)
# ═══════════════════════════════════════════════════════════════════════════
run_sala_profile() {
    local BACKEND="$1"
    local TIER="$2"
    local OUT_DIR="${RESULTS_DIR}/${BACKEND}/${TIER}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "── SALA Profiler: ${BACKEND} / $(tier_label $TIER) ──"

    export SGLANG_SALA_PROFILE=1
    export SGLANG_SALA_PROFILE_INTERVAL=20
    export SGLANG_SALA_PROFILE_LOG="${OUT_DIR}/sala_timings.csv"
    unset SGLANG_TORCH_PROFILER_DIR 2>/dev/null || true

    launch_server "$BACKEND" "$TIER"

    local bench_args
    read -ra bench_args <<< "$(get_bench_args "$TIER")"

    echo "Running benchmark with SALA profiler..."
    python3 -m sglang.bench_serving "${bench_args[@]}" \
        2>&1 | tee "${OUT_DIR}/bench_sala.txt"

    cp /opt/server.log "${OUT_DIR}/server_sala.log"
    grep -A 500 "SALA-PROFILE" "${OUT_DIR}/server_sala.log" > "${OUT_DIR}/sala_summary.txt" 2>/dev/null || true

    unset SGLANG_SALA_PROFILE SGLANG_SALA_PROFILE_INTERVAL SGLANG_SALA_PROFILE_LOG 2>/dev/null || true
    kill_server

    echo "  Results: ${OUT_DIR}/sala_summary.txt, sala_timings.csv"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase: Nsight Systems (nsys)
# ═══════════════════════════════════════════════════════════════════════════
run_nsys_profile() {
    local BACKEND="$1"
    local TIER="$2"
    local OUT_DIR="${RESULTS_DIR}/${BACKEND}/${TIER}"
    mkdir -p "$OUT_DIR"

    if ! command -v nsys &>/dev/null; then
        echo "WARNING: nsys not found, skipping Nsight Systems profiling"
        return 0
    fi

    echo ""
    echo "── Nsight Systems: ${BACKEND} / $(tier_label $TIER) ──"

    export SGLANG_SALA_PROFILE=1  # NVTX markers
    unset SGLANG_TORCH_PROFILER_DIR 2>/dev/null || true

    launch_server "$BACKEND" "$TIER"

    # Use fewer prompts for nsys to keep trace manageable
    local nsys_prompts=4
    [ "$TIER" = "c64" ] && nsys_prompts=8

    echo "Running nsys-profiled benchmark (${nsys_prompts} prompts)..."
    nsys profile \
        -o "${OUT_DIR}/nsys_trace" \
        -t cuda,nvtx,osrt \
        --force-overwrite true \
        python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$PORT" \
        --dataset-name custom --dataset-path "$BENCH_DATA" \
        --num-prompts "$nsys_prompts" --flush-cache \
        $([ "$TIER" = "c1" ] && echo "--max-concurrency 1") \
        $([ "$TIER" = "c8" ] && echo "--max-concurrency 8") \
        2>&1 | tee "${OUT_DIR}/bench_nsys.txt"

    # Generate stats
    NSYS_FILE=$(ls -t "${OUT_DIR}"/nsys_trace.* 2>/dev/null | head -1)
    if [ -n "${NSYS_FILE:-}" ]; then
        nsys stats "$NSYS_FILE" --report cuda_gpu_kern_sum \
            --format csv --output "${OUT_DIR}/nsys_kernel_summary" 2>/dev/null || true
        nsys stats "$NSYS_FILE" --report nvtx_sum \
            --format csv --output "${OUT_DIR}/nsys_nvtx_summary" 2>/dev/null || true
        echo "  Nsys trace: $NSYS_FILE (open with nsys-ui)"
    fi

    unset SGLANG_SALA_PROFILE 2>/dev/null || true
    kill_server
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase: Baseline benchmark + GPU utilization (no profiler overhead)
# ═══════════════════════════════════════════════════════════════════════════
run_baseline_bench() {
    local BACKEND="$1"
    local TIER="$2"
    local OUT_DIR="${RESULTS_DIR}/${BACKEND}/${TIER}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "── Baseline Benchmark: ${BACKEND} / $(tier_label $TIER) ──"

    unset SGLANG_TORCH_PROFILER_DIR SGLANG_SALA_PROFILE 2>/dev/null || true

    launch_server "$BACKEND" "$TIER"

    # GPU monitoring
    nvidia-smi dmon -s u -d 1 -c 1200 > "${OUT_DIR}/gpu_dmon.txt" 2>/dev/null &
    DMON_PID=$!
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
        --format=csv -l 2 > "${OUT_DIR}/gpu_mem.csv" 2>/dev/null &
    MEM_PID=$!

    local bench_args
    read -ra bench_args <<< "$(get_bench_args "$TIER")"

    echo "Running baseline benchmark..."
    python3 -m sglang.bench_serving "${bench_args[@]}" \
        2>&1 | tee "${OUT_DIR}/bench_baseline.txt"

    kill $DMON_PID 2>/dev/null || true
    kill $MEM_PID 2>/dev/null || true
    cp /opt/server.log "${OUT_DIR}/server_baseline.log"

    # Analyze
    python3 "${SCRIPT_DIR}/analyze_gpu.py" "${OUT_DIR}/gpu_dmon.txt" "${OUT_DIR}/gpu_mem.csv" \
        > "${OUT_DIR}/gpu_analysis.txt" 2>&1
    python3 "${SCRIPT_DIR}/analyze_scheduling.py" "${OUT_DIR}/server_baseline.log" \
        > "${OUT_DIR}/scheduling_analysis.txt" 2>&1

    cat "${OUT_DIR}/gpu_analysis.txt"
    kill_server
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase: Torch Profiler trace
# ═══════════════════════════════════════════════════════════════════════════
run_torch_profile() {
    local BACKEND="$1"
    local TIER="$2"
    local OUT_DIR="${RESULTS_DIR}/${BACKEND}/${TIER}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "── Torch Profiler: ${BACKEND} / $(tier_label $TIER) ──"

    export SGLANG_TORCH_PROFILER_DIR="/tmp/sglang_profile_${BACKEND}_${TIER}"
    export SGLANG_SALA_PROFILE=1  # record_function annotations
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

    launch_server "$BACKEND" "$TIER"

    local bench_args
    read -ra bench_args <<< "$(get_bench_args "$TIER")"

    echo "Running benchmark with torch profiler..."
    python3 -m sglang.bench_serving "${bench_args[@]}" \
        2>&1 | tee "${OUT_DIR}/bench_torch.txt"

    cp /opt/server.log "${OUT_DIR}/server_torch.log"

    TRACE_FILE=$(ls -t "${SGLANG_TORCH_PROFILER_DIR}"/*.trace.json.gz 2>/dev/null | head -1)
    if [ -n "${TRACE_FILE:-}" ]; then
        echo "Analyzing trace: $TRACE_FILE"
        python3 "${SCRIPT_DIR}/analyze_profile.py" "$TRACE_FILE" > "${OUT_DIR}/trace_analysis.txt" 2>&1
        cp "$TRACE_FILE" "${OUT_DIR}/"
    else
        echo "WARNING: No trace file found"
    fi

    unset SGLANG_TORCH_PROFILER_DIR SGLANG_SALA_PROFILE 2>/dev/null || true
    kill_server
}

# ═══════════════════════════════════════════════════════════════════════════
# Full profiling: all phases × all concurrency tiers for one backend
# ═══════════════════════════════════════════════════════════════════════════
run_full_profile() {
    local BACKEND="$1"

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  Full Profiling: ${BACKEND} × 3 concurrency tiers"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    for TIER in "${ALL_TIERS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  ${BACKEND} / $(tier_label $TIER)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        run_baseline_bench "$BACKEND" "$TIER"
        run_sala_profile "$BACKEND" "$TIER"
        run_torch_profile "$BACKEND" "$TIER"
        run_nsys_profile "$BACKEND" "$TIER"
    done

    echo ""
    echo "═══ All profiling complete for: ${BACKEND} ═══"
    echo "Results in: ${RESULTS_DIR}/${BACKEND}/"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

echo "Installing sglang from ${SGLANG_PKG}..."
uv pip install --no-deps -e "$SGLANG_PKG"

# Parse: command [backend] [tier]
CMD="${1:-}"
BACKEND_ARG="${2:-}"
TIER_ARG="${3:-}"

# Resolve tiers to run
resolve_tiers() {
    if [ -n "$TIER_ARG" ]; then
        echo "$TIER_ARG"
    else
        echo "${ALL_TIERS[@]}"
    fi
}

case "$CMD" in
    all)
        run_full_profile "flashinfer"
        run_full_profile "minicpm_flashinfer"
        echo "Generating comparison report..."
        python3 "${SCRIPT_DIR}/compare_backends.py" "${RESULTS_DIR}" > "${RESULTS_DIR}/comparison.txt" 2>&1
        cat "${RESULTS_DIR}/comparison.txt"
        ;;
    sala)
        BACKEND="${BACKEND_ARG:?Usage: run_profile.sh sala <backend> [c1|c8|c64]}"
        for t in $(resolve_tiers); do run_sala_profile "$BACKEND" "$t"; done
        ;;
    nsys)
        BACKEND="${BACKEND_ARG:?Usage: run_profile.sh nsys <backend> [c1|c8|c64]}"
        for t in $(resolve_tiers); do run_nsys_profile "$BACKEND" "$t"; done
        ;;
    torch)
        BACKEND="${BACKEND_ARG:?Usage: run_profile.sh torch <backend> [c1|c8|c64]}"
        for t in $(resolve_tiers); do run_torch_profile "$BACKEND" "$t"; done
        ;;
    baseline)
        BACKEND="${BACKEND_ARG:?Usage: run_profile.sh baseline <backend> [c1|c8|c64]}"
        for t in $(resolve_tiers); do run_baseline_bench "$BACKEND" "$t"; done
        ;;
    flashinfer|minicpm_flashinfer)
        run_full_profile "$CMD"
        ;;
    *)
        echo "MiniCPM-SALA Profiling (SOAR 2026 competition tiers)"
        echo ""
        echo "Usage:"
        echo "  bash run_profile.sh all                            # Both backends × 3 tiers"
        echo "  bash run_profile.sh <backend>                      # One backend × 3 tiers"
        echo "  bash run_profile.sh sala <backend> [c1|c8|c64]     # SALA profiler"
        echo "  bash run_profile.sh nsys <backend> [c1|c8|c64]     # Nsight Systems"
        echo "  bash run_profile.sh torch <backend> [c1|c8|c64]    # Torch profiler"
        echo "  bash run_profile.sh baseline <backend> [c1|c8|c64] # Baseline + GPU util"
        echo ""
        echo "Backends: flashinfer, minicpm_flashinfer"
        echo "Tiers:    c1  = --max-concurrent 1   (4 prompts,  40% score weight)"
        echo "          c8  = --max-concurrent 8   (16 prompts, 30% score weight)"
        echo "          c64 = unlimited            (64 prompts, 30% score weight)"
        echo ""
        echo "Results:  ${RESULTS_DIR}/<backend>/<tier>/"
        exit 1
        ;;
esac
