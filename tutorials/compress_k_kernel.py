import torch
import triton
import triton.language as tl
import time

# =====================================================================
# 1. 💀 Official Original Triton Source (The Buggy Version Nailed to the Pillar of Shame) 💀
# Review: 16 chefs cook, throw away 15 dishes after cooking, and force chef 0 to wash veggies, cut meat, and cook all 16 dishes alone sequentially!
# =====================================================================
@triton.jit
def official_buggy_compress_kernel(
    key_cache_ptr, token_table_ptr, full_compressed_k_ptr,
    token_table_cols: tl.constexpr, head_num_k: tl.constexpr,
    head_dim: tl.constexpr, kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ⚙️ [On-Chip Scheduling] The SM scheduler allocates independent concurrent threads to each Head
    batch_idx = tl.program_id(0)
    new_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # 🌟 [Phase 1: Everything seems normal, everyone is working concurrently]
    acc = tl.zeros([head_dim], dtype=tl.float32)
    for token_offset in range(kernel_size):
        token_y = new_chunk_idx * kernel_stride + token_offset
        # ⬇️ From [HBM (Slow Memory)] -> Fetched to -> [GPU Registers]
        token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)
        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim
        x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)
        # 🧮 Register Accumulation
        acc += x

    # 16 threads have now calculated 'acc' in their respective [GPU Registers]
    acc = acc / kernel_size

    # 💀💀💀 [Super Disaster Strikes: The Fatal Official Bug] 💀💀💀
    # 🔪 Sin 1: Massacre! Except for thread 0, the 'acc' calculated by threads 1~15 are completely discarded and destroyed on the spot! They sleep idle in place! GPU concurrency utilization plummets to 1/16!
    if head_idx == 0:

        # 🔪 Sin 2: The birth of slave 0! What 16 people could have finished concurrently, thread 0 is forced to do [Sequentially] 16 times alone!
        for h in range(head_num_k):
            # ➡️ Create a new 'head_acc' in the [GPU Registers] of thread 0
            head_acc = tl.zeros([head_dim], dtype=tl.float32)

            for token_offset in range(kernel_size):
                token_y = new_chunk_idx * kernel_stride + token_offset

                # 🔪 Sin 3: Repeated Table Lookups!
                # ⬇️ [Useless Fetch] Read the lookup table index from [HBM (Slow Memory)] again!
                token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)

                # Note: The loop variable 'h' is used here to calculate the physical memory offset
                key_base_offset = token_k_indices * head_num_k * head_dim + h * head_dim

                # 🔪 Sin 4: Repeated Memory Reads! Forcefully re-fetch the data that other threads have obviously already read from the slow [HBM (Slow Memory)]!
                # ⬇️ [Wasteful Fetch] From [HBM (Slow Memory)] -> [GPU Register x]
                x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)

                # 🧮 Thread 0 miserably accumulates for other Heads
                head_acc += x

            head_acc = head_acc / kernel_size
            out_offset = (batch_idx * tl.num_programs(1) * head_num_k * head_dim) + (new_chunk_idx * head_num_k * head_dim) + (h * head_dim)

            # ⬆️ [Tragic Write] Write the result painstakingly calculated [Sequentially] by thread 0 into [HBM (Slow Memory)]
            tl.store(full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE), head_acc, mask=tl.arange(0, BLOCK_SIZE) < head_dim)


# =====================================================================
# 2. 🔥 Ultimate Optimized Triton (Operator Fusion: Textbook-level Microscopic Annotation) 🔥
# Review: 16 chefs fire up at the same time, each cooks their own dish, and serves it straight to the table, no nonsense!
# =====================================================================
@triton.jit
def optimized_compress_kernel(
    key_cache_ptr, token_table_ptr, full_compressed_k_ptr,
    token_table_cols: tl.constexpr, head_num_k: tl.constexpr,
    head_dim: tl.constexpr, kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ⚙️ [On-Chip Scheduling] The SM scheduler allocates thread IDs, 16 Heads launch concurrently!
    batch_idx = tl.program_id(0)
    new_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # 🌟 [Core Starting Point of Operator Fusion] 🌟
    # ➡️ Directly allocate the 'acc' array in [GPU Ultra-Fast Registers], never touching the physical HBM!
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for token_offset in range(kernel_size):
        # 🧮 [In-Register Calculation] Calculate logical address token_y
        token_y = new_chunk_idx * kernel_stride + token_offset

        # ⬇️ [Memory Fetch 1] (Table Lookup Operation)
        # ➡️ From [HBM (Slow Memory: token_table_ptr)] -> Fetch 1 int32 -> [GPU Register: token_k_indices]
        token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)

        # 🧮 [In-Register Calculation] Calculate physical address key_base_offset
        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim

        # ⬇️ [Memory Fetch 2] (Extract Physical Token Vector)
        # ➡️ From [HBM (Slow Memory: key_cache_ptr)] -> Fetch float vector -> [GPU Ultra-Fast On-Chip Register: variable x]
        x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                    mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)

        # 🌟 [Operator Fusion Highlight: Consume data directly in registers, never spit it back to HBM!] 🌟
        # 🧮 ➡️ Take [GPU Register: variable x] + [GPU Register: variable acc] -> Overwrite on the spot -> [GPU Register: variable acc]
        acc += x

    # 🧮 [In-Register Calculation] (Average) -> Resolve on the spot inside registers
    acc = acc / kernel_size

    # 🧮 [In-Register Calculation] Calculate physical write address for output
    out_offset = (batch_idx * tl.num_programs(1) * head_num_k * head_dim) + (new_chunk_idx * head_num_k * head_dim) + (head_idx * head_dim)

    # ⬆️ [Memory Write] (The only time data is poured into HBM in the whole process)
    # ➡️ Directly and concurrently write the final calculated result [GPU Register: variable acc] -> to -> [HBM (Slow Memory: full_compressed_k_ptr)]
    # 🔥 16 chefs serve their dishes at the same time!
    tl.store(full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE), acc, mask=tl.arange(0, BLOCK_SIZE) < head_dim)


def run_triton(kernel_fn, key_cache, token_table, kernel_size, kernel_stride):
    batch_size, token_table_cols = token_table.shape
    total_tokens, head_num_k, head_dim = key_cache.shape
    new_chunks = (token_table_cols - kernel_size) // kernel_stride + 1
    out = torch.empty((batch_size, new_chunks, head_num_k, head_dim), device=key_cache.device, dtype=key_cache.dtype)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (batch_size, new_chunks, head_num_k)
    kernel_fn[grid](key_cache, token_table, out, token_table_cols, head_num_k, head_dim, kernel_size, kernel_stride, BLOCK_SIZE)
    return out


# =====================================================================
# 3. 📉 Unfused PyTorch (Negative Example: Extremely Brutal VRAM Meat Grinder) 📉
# =====================================================================
def run_pytorch(key_cache, token_table, kernel_size, kernel_stride):
    batch_size, token_table_cols = token_table.shape
    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]
    new_chunks = (token_table_cols - kernel_size) // kernel_stride + 1

    # ⬇️ [Memory Allocation] ➡️ Allocate output space [out] in [HBM (Slow Memory)]
    out = torch.empty((batch_size, new_chunks, head_num_k, head_dim), device=key_cache.device, dtype=key_cache.dtype)

    # 💀 [Unfused Disaster 1: Massive VRAM Relocation] 💀
    # ⬇️ Fetch all data from [HBM] into GPU -> 🧮 GPU Reorganization -> ⬆️ Forcefully write back to [HBM (Temporary Variable: reconstructed_k)]
    # Equivalent to moving the entire library to the street just to find one book!
    reconstructed_k = key_cache[token_table]

    for i in range(new_chunks):
        start = i * kernel_stride
        end = start + kernel_size

        # 💀 [Unfused Disaster 2: Repeated VRAM I/O] 💀
        # 1. ⬇️ Slice and fetch from the newly built [HBM (reconstructed_k)] to -> [GPU Cores]
        # 2. 🧮 Calculate mean in [GPU Registers]
        # 3. ⬆️ Implicitly write back to -> [HBM (Implicit Temporary Variable: tmp_mean)]
        # 4. ⬇️ Fetch again from -> [HBM (Implicit Temporary Variable: tmp_mean)]
        # 5. ⬆️ Final write to -> [HBM (Target: out)]
        out[:, i, :, :] = reconstructed_k[:, start:end, :, :].mean(dim=1)

    return out


# =====================================================================
# Test Benchmark
# =====================================================================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "I need an NVIDIA GPU!"
    torch.manual_seed(42)
    device = "cuda"

    BATCH_SIZE = 4
    SEQ_LEN = 8192
    HEAD_NUM_K = 16   # 🚨 Set number of heads to 16 to amplify the tragedy of the official sequential Bug
    HEAD_DIM = 128
    KERNEL_SIZE = 32
    KERNEL_STRIDE = 16

    print(f"Building Paged KV Cache... (Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}, Heads: {HEAD_NUM_K})")
    total_physical_tokens = BATCH_SIZE * SEQ_LEN * 2
    key_cache = torch.randn((total_physical_tokens, HEAD_NUM_K, HEAD_DIM), dtype=torch.float32, device=device)
    token_table = torch.randint(0, total_physical_tokens, (BATCH_SIZE, SEQ_LEN), dtype=torch.int32, device=device)

    # Warmup
    _ = run_pytorch(key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    _ = run_triton(official_buggy_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    _ = run_triton(optimized_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()

    # 1. Benchmark PyTorch (VRAM Meat Grinder)
    start = time.perf_counter()
    for _ in range(50): out_pt = run_pytorch(key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_pt = time.perf_counter() - start

    # 2. Benchmark Original Triton (With Sequential Bug)
    start = time.perf_counter()
    for _ in range(50): out_tr_buggy = run_triton(official_buggy_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_tr_buggy = time.perf_counter() - start

    # 3. Benchmark Optimized Triton (Full Concurrent Direct Write)
    start = time.perf_counter()
    for _ in range(50): out_tr_opt = run_triton(optimized_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_tr_opt = time.perf_counter() - start

    # Verify correctness (No matter how slow the official one is, the calculated result is still correct, which is why the Bug hid for so long!)
    assert torch.allclose(out_pt, out_tr_buggy, atol=1e-3), "Original Triton calculation error!"
    assert torch.allclose(out_pt, out_tr_opt, atol=1e-3), "Optimized Triton calculation error!"

    print("-" * 60)
    print(f"[Baseline] PyTorch Unfused Time:           {t_pt:.4f} s")
    print(f"[Added] Triton (Official Buggy Version):     {t_tr_buggy:.4f} s")
    print(f"[Initial] Triton (Full Concurrent Optimized):  {t_tr_opt:.4f} s")
    print("-" * 60)
    print(f"🔥 Original Official Triton is faster than PyTorch by: {t_pt / t_tr_buggy:.2f}x")
    print(f"🔥 Our Bug-fixed Triton is faster by:                  {t_pt / t_tr_opt:.2f}x!")
    print(f"👉 Extra free speedup from fixing the official Bug:    {t_tr_buggy / t_tr_opt:.2f}x!")
    print("=" * 60)