#!/bin/bash
set -e

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_DIR="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        *) exit 1 ;;
    esac
    shift
done

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH="$(which ptxas)"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$DIR/nvfp4_venv/bin/python3 $DIR/AWQ_NVFP4_dense_all.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --calib-data "$DIR/calib_dense_96.jsonl" \
    --max-samples 96 \
    --max-len 131072 \
    --mse-iters 120 \
    --smooth-alpha 0.5

uv pip install --no-deps -e $DIR/sglang_sala_cp || uv pip install --system --no-deps -e $DIR/sglang_sala_cp
