#!/bin/bash
set -e
cd /opt
python3 -m venv /opt/llmcompressor_venv
source /opt/llmcompressor_venv/bin/activate
pip install --upgrade pip setuptools wheel setuptools_scm

# Patch torch requirement and install
sed -i 's/torch>=2.9.0/torch>=2.8.0/g' /opt/llm-compressor/setup.py
cd /opt/llm-compressor
pip install -e .

# Pin torch 2.8 + matching torchvision + flash-attn
pip install https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
pip install torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flash-linear-attention

echo "Done! Activate with: source /opt/llmcompressor_venv/bin/activate"
