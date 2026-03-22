#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export SGLANG_SERVER_ARGS="--quantization modelopt --trust-remote-code --disable-radix-cache --kv-cache-dtype fp8_e5m2 --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-running-requests 64 --max-mamba-cache-size 64 --skip-server-warmup --dense-as-sparse --log-level info --mem-fraction-static 0.82 --enable-torch-compile"

export TORCHINDUCTOR_COMPILE_THREADS=20
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1

apt-get update && apt-get install -y libpcre3-dev

uv venv $DIR/nvfp4_venv --python 3.10

uv pip install --python $DIR/nvfp4_venv torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $DIR/nvfp4_venv $DIR/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
uv pip install --python $DIR/nvfp4_venv python-pcre regex nvidia-modelopt flash-linear-attention flashinfer-python tokenicer Pillow thefuzz numpy scipy tqdm safetensors sentencepiece protobuf huggingface-hub packaging tvm defuser gptqmodel accelerate datasets threadpoolctl logbar device-smi "transformers<5.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

export MAX_JOBS=16
export NVCC_THREADS=4
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
$DIR/nvfp4_venv/bin/python3 -m pip install -e $DIR/vendor/infllmv2_cuda_impl --no-build-isolation
$DIR/nvfp4_venv/bin/python3 -m pip install -e $DIR/vendor/sparse_kernel --no-build-isolation
