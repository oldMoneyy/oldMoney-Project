#!/bin/bash
set -e

# ============================================================
# env_sala.sh — Complete SALA environment setup
# Usage: nohup bash env_sala.sh > env_sala.log 2>&1 &
# Target: ~/compass_max_posttrain_1/.cz/sala/
# ============================================================

SALA_DIR="$HOME/compass_max_posttrain_1/.cz/sala"
ENV_DIR="$SALA_DIR/sglang_env_$(date +%Y%m%d)"
CONDA_DIR="$HOME/compass_max_posttrain_1/miniconda3"
REPO_DIR="$SALA_DIR/oldMoney-Project"

echo "================================================"
echo "SALA Environment Setup"
echo "Target: $SALA_DIR"
echo "================================================"

# ------------------------------------------------------------
# 1. Init conda
# ------------------------------------------------------------
echo "[1/8] Initializing conda..."
source "$CONDA_DIR/bin/activate"
eval "$(conda shell.bash hook)"

# Accept TOS if needed
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ------------------------------------------------------------
# 2. Create conda env with Python 3.10
# ------------------------------------------------------------
if [ -d "$ENV_DIR" ]; then
    echo "[2/8] Env already exists at $ENV_DIR, skipping creation..."
else
    echo "[2/8] Creating conda env with Python 3.10..."
    conda create -y -p "$ENV_DIR" python=3.10
fi

conda activate "$ENV_DIR"
echo "Python version: $(python --version)"

# ------------------------------------------------------------
# 3. Install PyTorch 2.9.1 + cu128
# ------------------------------------------------------------
echo "[3/8] Installing PyTorch 2.9.1+cu128..."
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# 4. Install CUDA 12.8 toolkit in env (for building extensions)
# ------------------------------------------------------------
echo "[4/8] Installing CUDA 12.8 toolkit via conda..."
conda install -y -c nvidia cuda-toolkit=12.8

# ------------------------------------------------------------
# 5. Install sgl-kernel + flashinfer
# ------------------------------------------------------------
echo "[5/8] Installing sgl-kernel and flashinfer..."
pip install sgl-kernel==0.3.20
pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3

# ------------------------------------------------------------
# 6. Clone repo and install sglang (editable)
# ------------------------------------------------------------
echo "[6/8] Setting up oldMoney-Project and sglang..."
if [ ! -d "$REPO_DIR" ]; then
    cd "$SALA_DIR"
    git clone https://github.com/oldMoneyy/oldMoney-Project.git
    cd "$REPO_DIR"
    git checkout ae1de29
else
    echo "Repo already exists at $REPO_DIR, skipping clone..."
fi

cd "$REPO_DIR/sglang_sala_cp"
pip install -e .

# ------------------------------------------------------------
# 7. Build vendor CUDA extensions
# ------------------------------------------------------------
echo "[7/8] Building vendor CUDA extensions..."
cd "$REPO_DIR/vendor/infllmv2_cuda_impl"
CUDA_HOME="$CONDA_PREFIX" pip install --no-build-isolation -e .

cd "$REPO_DIR/vendor/sparse_kernel"
CUDA_HOME="$CONDA_PREFIX" pip install --no-build-isolation -e .

# ------------------------------------------------------------
# 8. Install remaining dependencies
# ------------------------------------------------------------
echo "[8/8] Installing remaining dependencies..."
pip install \
    flash-linear-attention==0.4.1 \
    fla-core==0.4.1 \
    compressed-tensors==0.13.0 \
    torchao==0.9.0 \
    xgrammar==0.1.27 \
    tilelang==0.1.8 \
    transformers==4.57.1 \
    torch_memory_saver==0.0.9 \
    nvidia-cutlass-dsl==4.2.1 \
    nvidia-cudnn-frontend==1.18.0

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------
echo ""
echo "================================================"
echo "Verifying installation..."
echo "================================================"
python -c "
import sglang
import torch
import infllm_v2
import sparse_kernel_extension
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('All imports OK!')
"

echo ""
echo "================================================"
echo "Setup complete!"
echo ""
echo "To activate:"
echo "  source $CONDA_DIR/bin/activate"
echo "  conda activate $ENV_DIR"
echo ""
echo "To run server:"
echo "  fuser -k -9 31333/tcp"
echo "  export PYTORCH_ALLOC_CONF=expandable_segments:True"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python -m sglang.launch_server \\"
echo "      --model /path/to/model \\"
echo "      --trust-remote-code \\"
echo "      --disable-radix-cache \\"
echo "      --attention-backend minicpm_flashinfer \\"
echo "      --chunked-prefill-size 8192 \\"
echo "      --max-running-requests 32 \\"
echo "      --port 31333 \\"
echo "      --dense-as-sparse \\"
echo "      --mem-fraction-static 0.82"
echo "================================================"
