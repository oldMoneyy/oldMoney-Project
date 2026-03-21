
Once access the server:

```bash
echo "root:123456" | chpasswd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
service ssh restart

cd /opt
git clone https://oldMoneyy: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@github.com/oldMoneyy/oldMoney-Project.git
```

Steps to push commits:
```bash
git config --global user.name "github_user_name"
git config --global user.email "github_user_email"

cd /opt/oldMoney-Project

git pull

git add .
git commit -m "What are the commits about"
git remote set-url origin https://oldMoneyy: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@github.com/oldMoneyy/oldMoney-Project.git
git push -u origin main
```

# TODO
1. Support `--kv-cache-dtype fp8_e5m2` for minicpm backend (**DONE**).
2. Test original model's smax performance with minicpm_flashinfer and flashinfer (uv pip install --no-deps -e /opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python) respectively.
3. Test dense, sparse GPTQ W4 and original model's smax performance and accuracy.
4. Create an attention backend router that process short inputs by flashinfer and long inputs by minicpm_flashinfer.



# SGLang Serving


## Environment for SGLang Serving

Download model, toolkit and uv:
```bash
# Download MiniCPM-SALA model:
cd /opt
cat << 'EOF' > download_minicpm_sala.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openbmb/MiniCPM-SALA",
    local_dir="./model",
    local_dir_use_symlinks=False,
    resume_download=True,
)
EOF

pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# export HF_ENDPOINT=https://hf-mirror.com
python download_minicpm_sala.py


# Get the acc / throughput test toolkit:
cd /opt
git clone https://github.com/OpenBMB/SOAR-Toolkit.git


# Get uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```




How to kill a sglang process:
```bash
pkill -f sglang
```

See GPU info:
```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}\nCompute Capability: SM{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}\nCUDA Version: {torch.version.cuda}\nHas FP8 E4M3: {hasattr(torch, \"float8_e4m3fn\")}')"
```




Copy a backup of original env:
```bash
# cp -r /opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python/* /opt/oldMoney-Project/sglang_sala_cp/
uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_cp
# uv pip install --no-deps -e /opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python
```

What I have done to the environment:
1. Updated `if model_runner.server_args.fuse_topk:` logic in minicpm_backend.py for JIT redundant compiling.
2. Added fp8_e5m2 KV Cache support.
3. Fixed minicpm_fuse_kernel.py import error.
4. Optimized `build_sparse_prefill_metadata`, `build_token_mappings`.










## Start a Serving

SALA official huggingface start command:
```bash
cd /opt
nohup python3 -m sglang.launch_server \
    --model /opt/model \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --port 31333 \
    --dense-as-sparse \
    --mem-fraction-static 0.82 \
    > server.log 2>&1 &
```



## Curl Test

(1) Send a simple request:

```bash
curl http://localhost:31333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-SALA",
    "messages": [{"role": "user", "content": "What is 32768 * 8 - 1 = ?, give me step by step solution."}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```








(2) Send three long requests (5k, 40k, 60k):

```bash
python /opt/oldMoney-Project/bench/long_context_test_case.py
```
The answers are: `Paris`, `BLUE-TIGER-42`, `Alice Zhang, 1987`.








(3) Simple profile:  
In `MiniCPMSparseBackend.forward_extend`, `MiniCPMSparseBackend.init_forward_metadata`, `MiniCPMDecoderLayer.forward`, `FlashInferKernel.forward` there are profiling codes.

```bash
python /opt/oldMoney-Project/bench/profile_prefill.py
```


## Performance Test

Concurrency test aligned with SOAR official metric:
```bash
echo "=== S1 (concurrency 1) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench.jsonl \
    --num-prompts 64 --flush-cache --max-concurrency 1

echo "=== S8 (concurrency 8) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench.jsonl \
    --num-prompts 64 --flush-cache --max-concurrency 8

echo "=== Smax (unlimited) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench.jsonl \
    --num-prompts 64 --flush-cache
```



## Accuracy Test

```bash
cd /opt/SOAR-Toolkit
nohup python3 eval_model.py \
  --api_base http://127.0.0.1:31333 \
  --model_path /opt/model \
  --data_path eval_dataset/perf_public_set.jsonl \
  --concurrency 64 \
  --num_samples 150 \
  --verbose \
  > eval.log 2>&1 &
```




# GPTQ


## Environment for GPTQ

First time to prepare:
```bash
cd /opt/oldMoney-Project/quantization
mkdir venv
python -m venv venv/
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# 2. Verify Blackwell support
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch._C._GLIBCXX_USE_CXX11_ABI)"

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

apt-get update && apt-get install -y libpcre3-dev
uv pip install python-pcre regex
uv pip install flash-linear-attention
uv pip install tokenicer
uv pip install Pillow thefuzz numpy scipy tqdm safetensors sentencepiece protobuf huggingface-hub packaging
uv pip install tvm
uv pip install gptqmodel --no-deps

# Install its lightweight deps
uv pip install accelerate datasets threadpoolctl logbar device-smi
uv pip install "transformers<5.0"
pip install --upgrade pip

# Verify it imports
python -c "from gptqmodel import GPTQModel, QuantizeConfig; print('GPTQModel OK')"

# or
# bash /opt/oldMoney-Project/quantization/gptq_env_prepare.sh
```

Activate the environment:
```bash
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```




## Prepare Calibration Data

```bash
python3 /opt/oldMoney-Project/quantization/generate_ultimate_64.py
```



## Quantize Model

Pure dense quantization:
```bash
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
nohup python3 opt/oldMoney-Project/quantization/quantiza_gptq_dense_disk.py \
    --input /opt/model \
    --output /opt/model_gptq_ultimate_64_dense \
    --bits 4 \
    --group-size 128 \
    --calib-data /opt/oldMoney-Project/quantization/ultimate_64_token_balanced.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    > /opt/oldMoney-Project/quantization/gptq_ultimate_64_dense.log 2>&1 &
```


Original sparse quantization (# config.sparse_config["dense_len"] = 655360):
```bash
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
nohup python3 opt/oldMoney-Project/quantization/quantiza_gptq_sparse_disk.py \
    --input /opt/model \
    --output /opt/model_gptq_ultimate_64_sparse \
    --bits 4 \
    --group-size 128 \
    --calib-data /opt/oldMoney-Project/quantization/ultimate_64_token_balanced.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    > /opt/oldMoney-Project/quantization/gptq_ultimate_64_sparse.log 2>&1 &
```


## Deploy Quantized Models

Sparse:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m sglang.launch_server \
    --model-path /opt/model_gptq_ultimate_64_sparse \
    --port 31333 \
    --quantization gptq_marlin \
    --dtype float16 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.6 \
    --max-mamba-cache-size 32 \
    --fuse-topk \
    --max-running-requests 32 \
    --log-level info \
    --num-continuous-decode-steps 2 \
    --enable-mixed-chunk \
    --enable-torch-compile
```

Findings:
1. `--chunked-prefill-size 8192` would affect the accuracy for sparse model.
2. float16 has better accuracy.
3. flashinfer is far far far far far faster than minicpm_flashinfer.
3. pure dense based quantized model with flashinfer can pass all 3 long context test cases while sparse based quantized model with minicpm_flashinfer can only pass the first one.
4. original model with minicpm_flashinfer can pass all 3 long context test cases while with flashinfer it can only pass the first two.

Dense:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m sglang.launch_server \
    --model-path /opt/model_gptq_ultimate_64_dense \
    --port 31333 \
    --quantization gptq_marlin \
    --dtype float16 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.6 \
    --max-mamba-cache-size 32 \
    --fuse-topk \
    --max-running-requests 32 \
    --log-level info \
    --num-continuous-decode-steps 2 \
    --enable-mixed-chunk \
    --enable-torch-compile
```


1. Original model + minicpm_flashinfer → passes all 3 tests ✓
2. Original model + flashinfer → passes only first 2 tests
3. Dense-quantized model + flashinfer → passes all 3 tests ✓
4. Dense-quantized model + minicpm_flashinfer → passes only first test ✗
5. Sparse-quantized model + minicpm_flashinfer → passes only first test ✗


# NVFP4


Working like shit... Still optimizing.

```bash
uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_cp

source /opt/oldMoney-Project/quantization/nvfp4_venv/bin/activate

nohup python /opt/oldMoney-Project/quantization/nvfp4_quantize_sala.py \
    --input /opt/model \
    --output /opt/model_nvfp4_gptq \
    --calib-data /opt/oldMoney-Project/quantization/deadly_32_max_profit.jsonl \
    --max-samples 32 --max-len 131072 \
    > /opt/oldMoney-Project/quantization/nvfp4_quantize_sala.log 2>&1 &
```


```bash
python -m sglang.launch_server \
    --model /opt/model_nvfp4_mlp_only \
    --quantization modelopt_fp4 \
    --trust-remote-code \
    --port 31333 \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --skip-server-warmup \
    --dense-as-sparse \
    --mem-fraction-static 0.82
```





# Key Files
```bash
# modeling:
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/models/minicpm.py

# quantization:
# /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/fp4_quantization.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/__init__.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/petit_utils.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/petit.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/modelopt_utils.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/modelopt_quant.py
# /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/sgl_kernel/gemm.py
# /opt/oldMoney-Project/quantization/quantize_gptq_sparse_cpu.py

# kernels:
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/attention/minicpm_backend.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/attention/minicpm_fuse_kernel.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/attention/minicpm_attention_kernels.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# /opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/attention/flashinfer_backend.py
```

```bash
bash /opt/oldMoney-Project/utils_prompt/export_files.sh
```



