import torch
import triton
import triton.language as tl
import time

# =====================================================================
# 1. 💀 官方源码原版 Triton (钉在耻辱柱上的带 Bug 版) 💀
# 评价：16个厨师炒菜，炒完把15盘倒掉，逼0号厨师一个人重新洗菜切肉连炒16盘！
# =====================================================================
@triton.jit
def official_buggy_compress_kernel(
    key_cache_ptr, token_table_ptr, full_compressed_k_ptr,
    token_table_cols: tl.constexpr, head_num_k: tl.constexpr,
    head_dim: tl.constexpr, kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ⚙️ 【片上调度】SM 调度器给每个 Head 分配了独立的并发线程
    batch_idx = tl.program_id(0)
    new_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # 🌟【第一阶段：一切看似正常，大家都在并发干活】
    acc = tl.zeros([head_dim], dtype=tl.float32)
    for token_offset in range(kernel_size):
        token_y = new_chunk_idx * kernel_stride + token_offset
        # ⬇️ 从 [HBM(慢速显存)] -> 拉取到 -> [GPU寄存器]
        token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)
        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim
        x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)
        # 🧮 寄存器累加
        acc += x

    # 16个线程此时都在各自的 [GPU寄存器] 里算好了 acc
    acc = acc / kernel_size

    # 💀💀💀【超级大灾难降临：官方的致命 Bug】💀💀💀
    # 🔪 罪状1：大屠杀！除了 0 号线程，其他 1~15 号线程之前算的 acc 全部作废，当场销毁！直接原地闲置休眠！GPU 并发利用率暴跌至 1/16！
    if head_idx == 0:

        # 🔪 罪状2：0号奴隶的诞生！本来 16 个人能同时干完的事，偏偏逼着 0 号线程一个人【串行】干 16 遍！
        for h in range(head_num_k):
            # ➡️ 在 0 号线程的 [GPU寄存器] 里新建一个 head_acc
            head_acc = tl.zeros([head_dim], dtype=tl.float32)

            for token_offset in range(kernel_size):
                token_y = new_chunk_idx * kernel_stride + token_offset

                # 🔪 罪状3：重复查表！
                # ⬇️ 【无用功拉取】 再次从 [HBM(慢速显存)] 读查表索引！
                token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)

                # 注意这里用的是循环变量 h 去计算物理显存的偏移
                key_base_offset = token_k_indices * head_num_k * head_dim + h * head_dim

                # 🔪 罪状4：重复读显存！把刚才别的线程明明读过的数据，强行从慢速的 [HBM(慢速显存)] 再拉取一遍！
                # ⬇️ 【暴殄天物拉取】 从 [HBM(慢速显存)] -> [GPU寄存器 x]
                x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)

                # 🧮 0号线程苦逼地帮别的 Head 做累加
                head_acc += x

            head_acc = head_acc / kernel_size
            out_offset = (batch_idx * tl.num_programs(1) * head_num_k * head_dim) + (new_chunk_idx * head_num_k * head_dim) + (h * head_dim)

            # ⬆️ 【悲惨写入】 把 0 号线程辛辛苦苦【串行】算出来的结果，写入 [HBM(慢速显存)]
            tl.store(full_compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE), head_acc, mask=tl.arange(0, BLOCK_SIZE) < head_dim)


# =====================================================================
# 2. 🔥 终极优化版 Triton (算子融合：教科书级的显微镜标注) 🔥
# 评价：16个厨师同时开火，各自炒各自的菜，炒完直接端上桌，绝不废话！
# =====================================================================
@triton.jit
def optimized_compress_kernel(
    key_cache_ptr, token_table_ptr, full_compressed_k_ptr,
    token_table_cols: tl.constexpr, head_num_k: tl.constexpr,
    head_dim: tl.constexpr, kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # ⚙️ 【片上调度】SM 调度器分配线程 ID，16个 Head 并发启动！
    batch_idx = tl.program_id(0)
    new_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # 🌟【算子融合核心起点】🌟
    # ➡️ 在 [GPU超高速寄存器 Register] 中直接开辟数组 acc，绝不触碰物理显存 HBM！
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for token_offset in range(kernel_size):
        # 🧮【寄存器内计算】计算逻辑地址 token_y
        token_y = new_chunk_idx * kernel_stride + token_offset

        # ⬇️ 【内存拉取 1】 (查表操作)
        # ➡️ 从 [HBM(慢速显存 token_table_ptr)] -> 拉取 1个 int32 -> [GPU寄存器 token_k_indices]
        token_k_indices = tl.load(token_table_ptr + batch_idx * token_table_cols + token_y).to(tl.int32)

        # 🧮【寄存器内计算】计算物理地址 key_base_offset
        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim

        # ⬇️ 【内存拉取 2】 (提取物理 Token 向量)
        # ➡️ 从 [HBM(慢速显存 key_cache_ptr)] -> 拉取浮点向量 -> [GPU超高速片上寄存器 变量 x]
        x = tl.load(key_cache_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
                    mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0).to(tl.float32)

        # 🌟【算子融合高光时刻：在寄存器里直接吃掉数据，绝不吐回显存！】🌟
        # 🧮 ➡️ 取 [GPU寄存器 变量 x] + [GPU寄存器 变量 acc] -> 当场覆盖 -> [GPU寄存器 变量 acc]
        acc += x

    # 🧮【寄存器内计算】(求平均) -> 在寄存器内部当场解决
    acc = acc / kernel_size

    # 🧮【寄存器内计算】计算输出的物理写入地址
    out_offset = (batch_idx * tl.num_programs(1) * head_num_k * head_dim) + (new_chunk_idx * head_num_k * head_dim) + (head_idx * head_dim)

    # ⬆️ 【内存写入】 (全场唯一一次往显存倒数据)
    # ➡️ 将算好的最终结果 [GPU寄存器 变量 acc] -> 直接并发写入到 -> [HBM(慢速显存 full_compressed_k_ptr)]
    # 🔥 16 个厨师同时出锅端菜！
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
# 3. 📉 未融合版 PyTorch (反面教材：极度残暴的显存绞肉机) 📉
# =====================================================================
def run_pytorch(key_cache, token_table, kernel_size, kernel_stride):
    batch_size, token_table_cols = token_table.shape
    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]
    new_chunks = (token_table_cols - kernel_size) // kernel_stride + 1

    # ⬇️ 【内存分配】 ➡️ 在 [HBM(慢速显存)] 开辟输出空间 [out]
    out = torch.empty((batch_size, new_chunks, head_num_k, head_dim), device=key_cache.device, dtype=key_cache.dtype)

    # 💀【未融合灾难 1：巨型显存大搬家】💀
    # ⬇️ 从 [HBM] 拉取全部数据进 GPU -> 🧮 GPU重组 -> ⬆️ 强行写回 [HBM(临时变量 reconstructed_k)]
    # 相当于为了找一本书，把整个图书馆搬到了大街上！
    reconstructed_k = key_cache[token_table]

    for i in range(new_chunks):
        start = i * kernel_stride
        end = start + kernel_size

        # 💀【未融合灾难 2：显存反复吞吐】💀
        # 1. ⬇️ 从刚建好的 [HBM(reconstructed_k)] 中切片拉取到 -> [GPU核心]
        # 2. 🧮 在 [GPU寄存器] 算 mean
        # 3. ⬆️ 隐式写回到 -> [HBM(隐式临时变量 tmp_mean)]
        # 4. ⬇️ 再次从 -> [HBM(隐式临时变量 tmp_mean)] 拉取
        # 5. ⬆️ 最终写入到 -> [HBM(目标 out)]
        out[:, i, :, :] = reconstructed_k[:, start:end, :, :].mean(dim=1)

    return out


# =====================================================================
# 测试 Benchmark
# =====================================================================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "老子需要一张 NVIDIA 显卡！"
    torch.manual_seed(42)
    device = "cuda"

    BATCH_SIZE = 4
    SEQ_LEN = 8192
    HEAD_NUM_K = 16   # 🚨 头数设为 16，放大官方串行 Bug 的惨状
    HEAD_DIM = 128
    KERNEL_SIZE = 32
    KERNEL_STRIDE = 16

    print(f"正在构建 Paged KV Cache... (Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}, Heads: {HEAD_NUM_K})")
    total_physical_tokens = BATCH_SIZE * SEQ_LEN * 2
    key_cache = torch.randn((total_physical_tokens, HEAD_NUM_K, HEAD_DIM), dtype=torch.float32, device=device)
    token_table = torch.randint(0, total_physical_tokens, (BATCH_SIZE, SEQ_LEN), dtype=torch.int32, device=device)

    # 预热 (Warmup)
    _ = run_pytorch(key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    _ = run_triton(official_buggy_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    _ = run_triton(optimized_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()

    # 1. 测 PyTorch (显存绞肉机)
    start = time.perf_counter()
    for _ in range(50): out_pt = run_pytorch(key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_pt = time.perf_counter() - start

    # 2. 测 原版 Triton (带串行 Bug)
    start = time.perf_counter()
    for _ in range(50): out_tr_buggy = run_triton(official_buggy_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_tr_buggy = time.perf_counter() - start

    # 3. 测 优化版 Triton (全并发直写)
    start = time.perf_counter()
    for _ in range(50): out_tr_opt = run_triton(optimized_compress_kernel, key_cache, token_table, KERNEL_SIZE, KERNEL_STRIDE)
    torch.cuda.synchronize()
    t_tr_opt = time.perf_counter() - start

    # 验证正确性（哪怕官方再慢，算出来的结果也是对的，这就是为什么 Bug 能藏这么久！）
    assert torch.allclose(out_pt, out_tr_buggy, atol=1e-3), "原版 Triton 计算错误!"
    assert torch.allclose(out_pt, out_tr_opt, atol=1e-3), "优化版 Triton 计算错误!"

    print("-" * 60)
    print(f"[基线] PyTorch 未融合耗时:        {t_pt:.4f} 秒")
    print(f"[新增] Triton (官方带 Bug 版):    {t_tr_buggy:.4f} 秒")
    print(f"[初版] Triton (全并发优化版):     {t_tr_opt:.4f} 秒")
    print("-" * 60)
    print(f"🔥 原版官方 Triton 比 PyTorch 快: {t_pt / t_tr_buggy:.2f} 倍")
    print(f"🔥 我们修复 Bug 后的 Triton 快:   {t_pt / t_tr_opt:.2f} 倍！")
    print(f"👉 修复官方 Bug 带来的额外白嫖加速: {t_tr_buggy / t_tr_opt:.2f} 倍！")
    print("=" * 60)