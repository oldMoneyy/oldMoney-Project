"""
Quick profiling: inject timing into one forward pass to see where time goes.
Run this INSTEAD of the benchmark — it instruments one 150K prefill.

Usage: python /home/claude/profile_prefill.py
"""
import torch
import time

# ============================================================
# Monkey-patch key functions to measure time
# ============================================================

from sglang.srt.layers.attention import minicpm_backend
from sglang.srt.layers.attention import minicpm_sparse_utils
from sglang.srt.models import minicpm as minicpm_model

timings = {}

def make_timer(name, orig_fn):
    """Wrap a function with CUDA-synced timing."""
    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = orig_fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if name not in timings:
            timings[name] = {"total": 0.0, "count": 0}
        timings[name]["total"] += (t1 - t0)
        timings[name]["count"] += 1
        return result
    return timed

# Patch the sparse backend methods
orig_get_topk = minicpm_backend.MiniCPMSparseBackend.get_topk_for_sparse
orig_forward_extend = minicpm_backend.MiniCPMSparseBackend.forward_extend
orig_update_batch = minicpm_backend.MiniCPMSparseBackend.update_batch_for_sparse
orig_init_metadata = minicpm_backend.MiniCPMSparseBackend.init_forward_metadata

minicpm_backend.MiniCPMSparseBackend.get_topk_for_sparse = make_timer("get_topk_for_sparse", orig_get_topk)
minicpm_backend.MiniCPMSparseBackend.update_batch_for_sparse = make_timer("update_batch_for_sparse", orig_update_batch)
minicpm_backend.MiniCPMSparseBackend.init_forward_metadata = make_timer("init_forward_metadata", orig_init_metadata)

# Patch compress key functions
if hasattr(minicpm_sparse_utils, 'get_compress_k_v2'):
    orig_compress = minicpm_sparse_utils.get_compress_k_v2
    minicpm_sparse_utils.get_compress_k_v2 = make_timer("get_compress_k_v2", orig_compress)

if hasattr(minicpm_sparse_utils, 'allocate_and_compress_keys'):
    orig_alloc_compress = minicpm_sparse_utils.allocate_and_compress_keys
    minicpm_sparse_utils.allocate_and_compress_keys = make_timer("allocate_and_compress_keys", orig_alloc_compress)

if hasattr(minicpm_sparse_utils, 'compressed_attention'):
    orig_compressed_attn = minicpm_sparse_utils.compressed_attention
    minicpm_sparse_utils.compressed_attention = make_timer("compressed_attention", orig_compressed_attn)

if hasattr(minicpm_sparse_utils, 'compressed_attention_tilelang'):
    orig_compressed_attn_tl = minicpm_sparse_utils.compressed_attention_tilelang
    minicpm_sparse_utils.compressed_attention_tilelang = make_timer("compressed_attention_tilelang", orig_compressed_attn_tl)

# Patch the MiniCPM layer types
from sglang.srt.layers.attention.hybrid_linear_attn_backend import SimpleGLAAttnBackend
if hasattr(SimpleGLAAttnBackend, 'forward'):
    orig_gla = SimpleGLAAttnBackend.forward
    SimpleGLAAttnBackend.forward = make_timer("lightning_attn_forward", orig_gla)

# Patch attention kernel forward
from sglang.srt.layers.attention.minicpm_attention_kernels import AttentionKernel
if hasattr(AttentionKernel, 'forward'):
    orig_attn_kernel = AttentionKernel.forward
    AttentionKernel.forward = make_timer("attention_kernel_forward", orig_attn_kernel)

# Patch MLP
orig_mlp_forward = minicpm_model.MiniCPMMLP.forward
minicpm_model.MiniCPMMLP.forward = make_timer("mlp_forward", orig_mlp_forward)

# Patch sparse attention forward
orig_sparse_attn = minicpm_model.MiniCPMAttention.forward
minicpm_model.MiniCPMAttention.forward = make_timer("sparse_attn_layer_forward", orig_sparse_attn)

# Patch lightning mixer forward
orig_lightning = minicpm_model.MiniCPMLightningMixer.forward
minicpm_model.MiniCPMLightningMixer.forward = make_timer("lightning_mixer_forward", orig_lightning)

# Patch LayerNorm
from sglang.srt.layers.layernorm import RMSNorm
orig_rmsnorm = RMSNorm.forward
RMSNorm.forward = make_timer("rmsnorm_forward", orig_rmsnorm)

# ============================================================
# Now run one request
# ============================================================
import requests

print("=" * 70)
print("PROFILING: Sending one 150K-token prefill request...")
print("=" * 70)

parts = [f'Document {i}: The annual rainfall in region {i} averages {100+i}mm per year.' for i in range(8000)]
parts[2000] = 'Document 2000: The president of Greenfield Corp is Alice Zhang.'
parts[5000] = 'Document 5000: The founding year of Greenfield Corp is 1987.'
text = ' '.join(parts)
text += ' Question: Who is the president of Greenfield Corp and when was it founded? Answer in one word.'

t_start = time.perf_counter()
r = requests.post(
    'http://localhost:31333/v1/chat/completions',
    json={
        'model': 'MiniCPM-SALA',
        'messages': [{'role': 'user', 'content': text}],
        'max_tokens': 16,
        'temperature': 0.0
    },
    timeout=300
)
t_total = time.perf_counter() - t_start

print(f"\nTotal request time: {t_total:.2f}s")
print(f"Response: {r.json()['choices'][0]['message']['content'][:200]}")

# ============================================================
# Print results sorted by total time
# ============================================================
print("\n" + "=" * 70)
print(f"{'Function':<40} {'Total(s)':>10} {'Calls':>8} {'Avg(ms)':>10} {'% of Total':>10}")
print("=" * 70)

sorted_timings = sorted(timings.items(), key=lambda x: -x[1]["total"])
sum_all = sum(v["total"] for v in timings.values())

for name, data in sorted_timings:
    total = data["total"]
    count = data["count"]
    avg_ms = (total / count) * 1000 if count > 0 else 0
    pct = (total / t_total) * 100
    print(f"{name:<40} {total:>10.2f} {count:>8} {avg_ms:>10.2f} {pct:>9.1f}%")

print(f"\n{'Sum of instrumented':<40} {sum_all:>10.2f}s")
print(f"{'Uninstrumented overhead':<40} {t_total - sum_all:>10.2f}s")