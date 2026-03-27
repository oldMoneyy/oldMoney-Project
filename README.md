

# Pull, Push and Mocking

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

git config --global user.name "boris-dotv"
git config --global user.email "1322553126@qq.com"

cd /opt/oldMoney-Project

git pull

git add .
git commit -m "What are the commits about"
git remote set-url origin https://oldMoneyy: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@github.com/oldMoneyy/oldMoney-Project.git
git push -u origin main
```

Mock the process on SOAR official server:
```bash
# Contest server GPU:
# https://www.techpowerup.com/gpu-specs/rtx-6000d.c4363
cd /opt/oldMoney-Project/submissions
bash simulate_soar.sh submission_20260322.tar.gz
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

apt update
apt install psmisc lsof -y
```




How to kill a sglang process:
```bash
pkill -f sglang.launch
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
fuser -k -9 31333/tcp
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
    > /opt/server.log 2>&1 &
```



## Curl Test

(1) Send a simple request:

```bash
curl http://localhost:31333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-SALA",
    "messages": [{"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: '\''ANSWER: $LETTER'\'' (without quotes) where LETTER is one of ABCD. Think step by step before answering. Which of the following (effective) particles is not associated with a spontaneously-broken symmetry? A) Phonon B) Magnon C) Pion D) Skyrmion "}],
    "max_tokens": 8192,
    "temperature": 0.0
  }'
```








(2) Send three long requests (5k, 40k, 60k):

```bash
python /opt/oldMoney-Project/bench/long_context_test_case.py
```
The answers are: `Paris`, `BLUE-TIGER-42`, `Alice Zhang, 1987`.


KL divergence test:
```bash
cd /opt/oldMoney-Project/quantization && python /opt/oldMoney-Project/quantization/fast_eval.py --mode eval --api-base http://127.0.0.1:31333
```







(3) Simple profile:  
In `MiniCPMSparseBackend.forward_extend`, `MiniCPMSparseBackend.init_forward_metadata`, `MiniCPMDecoderLayer.forward`, `FlashInferKernel.forward` there are profiling codes.

```bash
python /opt/oldMoney-Project/bench/profile_prefill.py
```


## Performance Test
Generate the performance test set:
```bash
python /opt/oldMoney-Project/bench/gen_competition_bench.py
```

Concurrency test aligned with SOAR official metric:
```bash
echo "=== S1 (concurrency 1) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench_32.jsonl \
    --num-prompts 32 --flush-cache --max-concurrency 1

echo "=== S8 (concurrency 8) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench_32.jsonl \
    --num-prompts 32 --flush-cache --max-concurrency 8

echo "=== Smax (unlimited) ==="
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31333 \
    --dataset-name custom --dataset-path /opt/oldMoney-Project/bench/competition_bench_64.jsonl \
    --num-prompts 64 --flush-cache
```



## Accuracy Test

```bash
cd /opt/SOAR-Toolkit
nohup python3 eval_model.py \
  --api_base http://127.0.0.1:31333 \
  --model_path /opt/model_nvfp4_awq_v2 \
  --data_path eval_dataset/perf_public_set.jsonl \
  --concurrency 64 \
  --num_samples 150 \
  --verbose \
  > /opt/oldMoney-Project/logs/eval_nvfp4_awq_v2.log 2>&1 &
```




# GPTQ


## Environment for GPTQ

First time to prepare:
```bash
bash /opt/oldMoney-Project/quantization/GPTQ_INT4_env.sh
tail -f /opt/oldMoney-Project/logs/GPTQ_INT4_env.log
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
nohup python3 /opt/oldMoney-Project/quantization/GPTQ_int4_flashinfer_dense_gpu.py \
    --input /opt/model \
    --output /opt/model_gptq_int4_flashinfer_dense \
    --bits 4 \
    --group-size 128 \
    --calib-data /opt/oldMoney-Project/quantization/ultimate_64_token_balanced.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    > /opt/oldMoney-Project/logs/model_gptq_int4_flashinfer_dense.log 2>&1 &

tail -f /opt/oldMoney-Project/logs/model_gptq_int4_flashinfer_dense.log
```



Original sparse quantization (# config.sparse_config["dense_len"] = 655360):
```bash
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
nohup python3 /opt/oldMoney-Project/quantization/GPTQ_int4_minicpm_flashinfer_sparse_gpu.py \
    --input /opt/model \
    --output /opt/model_gptq_int4_minicpm_flashinfer_sparse \
    --bits 4 \
    --group-size 128 \
    --calib-data /opt/oldMoney-Project/quantization/ultimate_64_token_balanced.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    > /opt/oldMoney-Project/logs/model_gptq_int4_minicpm_flashinfer_sparse.log 2>&1 &

tail -f /opt/oldMoney-Project/logs/model_gptq_int4_minicpm_flashinfer_sparse.log
```

Smoothing sparse quantization:
```bash
source /opt/oldMoney-Project/quantization/venv/bin/activate
export LD_LIBRARY_PATH=/opt/oldMoney-Project/quantization/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
nohup python3 /opt/oldMoney-Project/quantization/GPTQ_int4_minicpm_flashinfer_sparse_smoothing_gpu.py \
    --input /opt/model \
    --output /opt/model_gptq_int4_minicpm_flashinfer_sparse \
    --bits 4 \
    --group-size 128 \
    --calib-data /opt/oldMoney-Project/quantization/calibration/optimal_64.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    > /opt/oldMoney-Project/logs/model_gptq_int4_minicpm_flashinfer_sparse.log 2>&1 &

tail -f /opt/oldMoney-Project/logs/model_gptq_int4_minicpm_flashinfer_sparse.log
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
    --model-path /opt/model_gptq_int4_minicpm_flashinfer_sparse \
    --port 31333 \
    --quantization gptq_marlin \
    --dtype float16 \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.8 \
    --max-mamba-cache-size 64 \
    --fuse-topk \
    --max-running-requests 64 \
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












# AWQ

Prepare environment for AWQ:
```bash
bash /opt/oldMoney-Project/quantization/AWQ_NVFP4_env.sh
tail -f /opt/oldMoney-Project/logs/AWQ_NVFP4_env.log
```

Working like shit... Still optimizing.

```bash
uv pip install --no-deps -e /opt/oldMoney-Project/sglang_sala_cp

export TRITON_PTXAS_PATH="$(which ptxas)"
source /opt/oldMoney-Project/quantization/nvfp4_venv/bin/activate
nohup python /opt/oldMoney-Project/quantization/AWQ_L_4_Mini_16.py \
 --input /opt/model \
 --output /opt/model_nvfp4_awq_v2 \
 --calib-data /opt/oldMoney-Project/quantization/calibration/optimal_64.jsonl \
 --max-samples 64 \
 --max-len 131072 \
 --mse-iters 200 \
 --mse-max-shrink 0.60 \
 --mse-error-norm 2.0 \
 > /opt/oldMoney-Project/logs/AWQ_L_4_Mini_16_train.log 2>&1 &

source /opt/oldMoney-Project/quantization/nvfp4_venv/bin/activate
python /opt/oldMoney-Project/quantization/AWQ_L_4_Mini_16.py \
 --input /opt/model \
 --output /opt/model_AWQ_L_4_Mini_16_calib_96 \
 --calib-data /opt/optimal_96.jsonl \
 --max-samples 96 \
 --max-len 131072 \
 --mse-iters 200 \
 --mse-max-shrink 0.60 \
 --mse-error-norm 2.0

export TRITON_PTXAS_PATH="$(which ptxas)"
source /opt/oldMoney-Project/quantization/nvfp4_venv/bin/activate
nohup python /opt/oldMoney-Project/quantization/AWQ_L_4_Mini_16_smoothed.py \
    --input /opt/model \
    --output /tmp/model_nvfp4_smoothed \
    --calib-data /opt/oldMoney-Project/quantization/calibration/optimal_64.jsonl \
    --max-samples 64 \
    --max-len 131072 \
    --smooth-alpha 0.5 \
    --mse-iters 80 \
 > /opt/oldMoney-Project/logs/AWQ_L_4_Mini_16_smoothed.log 2>&1 &
```


## KL Divergence Quick Evaluation

```bash


```




```bash
cd /opt
fuser -k -9 31333/tcp
nohup python3 -m sglang.launch_server \
    --model /opt/model_nvfp4_awq_v2 \
    --quantization modelopt \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --max-running-requests 64 \
    --max-mamba-cache-size 64 \
    --skip-server-warmup \
    --port 31333 \
    --dense-as-sparse \
    --mem-fraction-static 0.82 \
    > /opt/server.log 2>&1 &

cd /opt
fuser -k -9 31333/tcp
nohup python3 -m sglang.launch_server \
    --model /tmp/model_nvfp4_smoothed \
    --quantization modelopt \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --max-running-requests 64 \
    --port 31333 \
    --log-level info \
    --dense-as-sparse \
    --mem-fraction-static 0.82 \
    > /opt/server.log 2>&1 &

```


Duration result for `model_nvfp4_awq_v2` with no torch compile on RTX 6000D:
```text
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     64        
Benchmark duration (s):                  684.45    
Total input tokens:                      3885243   
Total input text tokens:                 3885243   
Total generated tokens:                  409876    
Total generated tokens (retokenized):    323190    
Request throughput (req/s):              0.09      
Input token throughput (tok/s):          5676.48   
Output token throughput (tok/s):         598.84    
Peak output token throughput (tok/s):    1882.00   
Peak concurrent requests:                64        
Total token throughput (tok/s):          6275.33   
Concurrency:                             30.86     
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



