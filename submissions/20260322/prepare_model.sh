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
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$DIR/nvfp4_venv/bin/python3 $DIR/nvfp4_awq.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --calib-data "$DIR/optimal_64.jsonl" \
    --max-samples 64 \
    --max-len 131072 \
    --mse-iters 200 \
    --mse-max-shrink 0.60 \
    --mse-error-norm 2.0

uv pip install --no-deps -e $DIR/sglang_sala_cp || uv pip install --system --no-deps -e $DIR/sglang_sala_cp
