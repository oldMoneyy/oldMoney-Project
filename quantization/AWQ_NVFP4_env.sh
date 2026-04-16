nohup bash -lc '
set -e

echo "===== START ENV PREP $(date) ====="

cd ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization

# recreate venv if missing
if [ ! -d ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization/nvfp4_venv ]; then
  python -m venv ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization/nvfp4_venv
fi

source ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization/nvfp4_venv/bin/activate
unset PYTHONPATH
export MAX_JOBS=16
export NVCC_THREADS=4
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
export LD_LIBRARY_PATH=~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization/nvfp4_venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

echo "===== INSTALL TORCH ====="
# https://github.com/NVIDIA/TransformerEngine/issues/2771
uv pip install --python ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization/nvfp4_venv https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl

echo "===== VERIFY TORCH ====="
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list()); print(torch._C._GLIBCXX_USE_CXX11_ABI)"

echo "===== INSTALL FLASH ATTN ====="
cd ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/quantization
if [ ! -f flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl ]; then
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
fi
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
python3 -c "import flash_attn; print(flash_attn.__version__)"

echo "===== INSTALL PYTHON DEPS ====="
apt-get update && apt-get install -y libpcre3-dev
uv pip install python-pcre regex
uv pip install nvidia-modelopt
uv pip install flash-linear-attention
uv pip install flashinfer-python
uv pip install tokenicer
uv pip install Pillow thefuzz numpy scipy tqdm safetensors sentencepiece protobuf huggingface-hub packaging
uv pip install tvm
uv pip install defuser
uv pip install gptqmodel
uv pip install accelerate datasets threadpoolctl logbar device-smi
uv pip install "transformers<5.0"
pip install --upgrade pip

echo "===== VERIFY GPTQMODEL ====="
python -c "from gptqmodel import GPTQModel, QuantizeConfig; print(\"GPTQModel OK\")"
python -c "from flash_attn import flash_attn_func; print(\"flash_attn OK\")"

echo "===== PREPARE VENDOR SOURCES ====="
mkdir -p ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor
rm -rf ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/infllmv2_cuda_impl
rm -rf ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/sparse_kernel
cp -a ~/compass_max_posttrain_1/.cz/sala/SGLang-MiniCPM-SALA/packages/infllmv2_cuda_impl ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/
cp -a ~/compass_max_posttrain_1/.cz/sala/SGLang-MiniCPM-SALA/packages/sparse_kernel ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/

echo "===== CLEAN infllmv2_cuda_impl ====="
cd ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/infllmv2_cuda_impl
rm -rf build infllm_v2.egg-info
find . -name "*.so" -delete
find . -name "*.o" -delete
find . -name "*.obj" -delete

echo "===== BUILD infllmv2_cuda_impl ====="
pip install -e . --no-build-isolation

echo "===== VERIFY infllm_v2 ====="
python -c "from infllm_v2 import infllmv2_attn_stage1, max_pooling_1d_varlen; print(\"infllm_v2 OK\")"

echo "===== CLEAN sparse_kernel ====="
cd ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/vendor/sparse_kernel
rm -rf build *.egg-info
find . -name "*.so" -delete
find . -name "*.o" -delete
find . -name "*.obj" -delete

echo "===== BUILD sparse_kernel ====="
pip install -e . --no-build-isolation

echo "===== VERIFY sparse_kernel ====="
python -c "import sparse_kernel_extension; print(\"sparse_kernel_extension OK\")"

echo "===== FINAL VERIFY ====="
python -c "import transformers; print(\"transformers\", transformers.__version__, transformers.__file__)"
python -c "import torch; print(\"torch\", torch.__version__, torch.version.cuda)"
python -c "from flash_attn import flash_attn_func; print(\"flash_attn OK\")"
python -c "from infllm_v2 import infllmv2_attn_stage1; print(\"FINAL infllm_v2 OK\")"
python -c "import sparse_kernel_extension; print(\"FINAL sparse_kernel_extension OK\")"

echo "===== DONE ENV PREP $(date) ====="
' > ~/compass_max_posttrain_1/.cz/sala/oldMoney-Project/logs/AWQ_NVFP4_env.log 2>&1 &