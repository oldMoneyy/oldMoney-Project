#!/usr/bin/env python3
"""
Enhanced trace analyzer for MiniCPM-SALA profiling.
Covers: GPU kernel breakdown, FLA vs FlashInfer vs Marlin, CPU-GPU sync, CUDA graph coverage.

Usage: python3 analyze_profile.py <trace.json.gz>
"""
import gzip, json, sys, collections

def main():
    trace_file = sys.argv[1]
    print(f"Loading {trace_file}...")
    if trace_file.endswith('.gz'):
        with gzip.open(trace_file, 'rt') as f:
            data = json.load(f)
    else:
        with open(trace_file) as f:
            data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", data.get("events", []))
    print(f"Total events: {len(events):,}")

    gpu_kernels = collections.defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
    cpu_ops = collections.defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
    item_callers = collections.Counter()
    nonzero_callers = collections.Counter()
    sync_calls = collections.defaultdict(lambda: {"count": 0, "total_us": 0})

    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)
        ph = ev.get("ph", "")

        if ph != "X":
            continue

        cat_lower = cat.lower()

        if "kernel" in cat_lower or "gpu" in cat_lower:
            gpu_kernels[name]["count"] += 1
            gpu_kernels[name]["total_us"] += dur
            gpu_kernels[name]["max_us"] = max(gpu_kernels[name]["max_us"], dur)
        elif "cpu_op" in cat or "python" in cat_lower or "aten" in cat_lower or "user_annotation" in cat_lower:
            cpu_ops[name]["count"] += 1
            cpu_ops[name]["total_us"] += dur
            cpu_ops[name]["max_us"] = max(cpu_ops[name]["max_us"], dur)

        # Track CPU-GPU sync calls
        if name in ["aten::item", "aten::is_nonzero", "aten::_local_scalar_dense",
                     "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"]:
            sync_calls[name]["count"] += 1
            sync_calls[name]["total_us"] += dur
            args = ev.get("args", {})
            for key in ["Python traceback", "Traceback", "Python Line", "External id"]:
                if key in args:
                    tb = args[key]
                    caller = tb[-1] if isinstance(tb, list) else str(tb)[:150]
                    if "item" in name:
                        item_callers[caller] += 1
                    elif "nonzero" in name:
                        nonzero_callers[caller] += 1
                    break

    # ═══════════════════════════════════════════════════════════════
    # 1. TOP 30 GPU KERNELS BY TOTAL TIME
    # ═══════════════════════════════════════════════════════════════
    total_gpu = sum(v["total_us"] for v in gpu_kernels.values())
    print(f"\n{'='*100}")
    print(f"TOP 30 GPU KERNELS BY TOTAL TIME (total: {total_gpu/1e6:.2f}s)")
    print(f"{'='*100}")
    print(f"{'Count':>8}  {'Total ms':>10}  {'%':>6}  {'Avg ms':>10}  {'Max ms':>10}  {'Kernel name'}")
    print(f"{'-'*100}")
    sorted_gpu = sorted(gpu_kernels.items(), key=lambda x: -x[1]["total_us"])
    for name, d in sorted_gpu[:30]:
        pct = d["total_us"] * 100 / total_gpu if total_gpu else 0
        avg = d["total_us"] / d["count"] / 1000
        print(f"{d['count']:>8}  {d['total_us']/1000:>10.1f}  {pct:>5.1f}%  {avg:>10.3f}  {d['max_us']/1000:>10.3f}  {name[:80]}")

    # ═══════════════════════════════════════════════════════════════
    # 2. KERNEL CATEGORY SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"KERNEL CATEGORY SUMMARY")
    print(f"{'='*100}")

    categories = {
        "FLA chunk (prefill)":       ["chunk_fwd_kernel_h", "chunk_fwd_kernel_o", "chunk_bwd"],
        "FLA recurrent (decode)":    ["fused_recurrent_fwd_kernel", "fused_recurrent_bwd"],
        "FLA cumsum":                ["chunk_local_cumsum"],
        "FLA norm/gate":             ["layernorm_gated", "l2norm"],
        "Marlin GEMM (int4)":        ["Marlin", "marlin"],
        "cuBLAS GEMM":               ["cublas", "cutlass", "sm90_xmma", "sm80_xmma", "ampere_"],
        "FlashInfer Prefill":        ["BatchPrefillWith", "prefill_with"],
        "FlashInfer Decode":         ["BatchDecodeWith", "decode_with"],
        "FlashInfer Other":          ["flashinfer", "page_table"],
        "MiniCPM Sparse TopK":       ["topk", "fused_attn_pooling", "online_topk"],
        "MiniCPM Sparse Attention":  ["sparse_attn", "block_sparse"],
        "RMSNorm":                   ["RMSNorm", "rmsnorm", "FusedAddRMSNorm"],
        "Sigmoid/Activation":        ["sigmoid", "silu", "gelu"],
        "QKNorm":                    ["qknorm", "q_norm", "k_norm"],
        "Elementwise":               ["elementwise_kernel", "vectorized_elementwise"],
        "Memory ops":                ["Memcpy", "memcpy", "Memset", "memset"],
        "CUDA Graph":                ["cudaGraphLaunch"],
        "NCCL/Comm":                 ["nccl", "ncclKernel"],
    }

    uncategorized_time = total_gpu
    cat_results = []

    for cat_name, patterns in categories.items():
        cat_count = 0
        cat_time = 0
        matching_kernels = []
        for kname, d in gpu_kernels.items():
            if any(p.lower() in kname.lower() for p in patterns):
                cat_count += d["count"]
                cat_time += d["total_us"]
                matching_kernels.append((kname, d["total_us"]))
        if cat_time > 0:
            uncategorized_time -= cat_time
            top_kernel = max(matching_kernels, key=lambda x: x[1])[0][:50] if matching_kernels else ""
            cat_results.append((cat_name, cat_count, cat_time, top_kernel))

    cat_results.sort(key=lambda x: -x[2])
    print(f"{'Category':<30}  {'Count':>8}  {'Total ms':>10}  {'%':>6}  {'Top kernel'}")
    print(f"{'-'*100}")
    for cat_name, cat_count, cat_time, top_kernel in cat_results:
        pct = cat_time * 100 / total_gpu if total_gpu else 0
        print(f"{cat_name:<30}  {cat_count:>8}  {cat_time/1000:>10.1f}  {pct:>5.1f}%  {top_kernel}")

    if uncategorized_time > 0:
        pct = uncategorized_time * 100 / total_gpu if total_gpu else 0
        print(f"{'Uncategorized':<30}  {'':>8}  {uncategorized_time/1000:>10.1f}  {pct:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 3. CPU-GPU SYNC ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"CPU-GPU SYNC CALLS")
    print(f"{'='*100}")

    if sync_calls:
        for name, d in sorted(sync_calls.items(), key=lambda x: -x[1]["total_us"]):
            print(f"  {name:<40}  {d['count']:>8} calls  {d['total_us']/1000:>10.1f} ms total")
    else:
        print("  No sync calls detected (good!)")

    if item_callers:
        print(f"\n  Top aten::item callers:")
        for caller, count in item_callers.most_common(10):
            print(f"    {count:>8}  {caller[:120]}")

    if nonzero_callers:
        print(f"\n  Top aten::is_nonzero callers:")
        for caller, count in nonzero_callers.most_common(10):
            print(f"    {count:>8}  {caller[:120]}")

    # ═══════════════════════════════════════════════════════════════
    # 4. CUDA GRAPH COVERAGE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"CUDA GRAPH COVERAGE")
    print(f"{'='*100}")

    cg_launches = sum(d["count"] for n, d in gpu_kernels.items() if "cudaGraphLaunch" in n)
    cg_time = sum(d["total_us"] for n, d in gpu_kernels.items() if "cudaGraphLaunch" in n)
    fwd_calls = sum(d["count"] for n, d in cpu_ops.items()
                    if ("forward" in n.lower() and ("minicpm" in n.lower() or "model" in n.lower())))

    print(f"  CUDA Graph launches:     {cg_launches}")
    print(f"  CUDA Graph total time:   {cg_time/1000:.1f} ms")
    print(f"  Model forward calls:     {fwd_calls}")
    if cg_launches > 0 and fwd_calls > 0:
        print(f"  Graph coverage ratio:    {cg_launches/fwd_calls:.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 5. TOP CPU OPS
    # ═══════════════════════════════════════════════════════════════
    total_cpu = sum(v["total_us"] for v in cpu_ops.values())
    print(f"\n{'='*100}")
    print(f"TOP 20 CPU OPS BY TOTAL TIME (total: {total_cpu/1e6:.2f}s)")
    print(f"{'='*100}")
    print(f"{'Count':>8}  {'Total ms':>10}  {'Avg ms':>10}  {'Op name'}")
    print(f"{'-'*80}")
    sorted_cpu = sorted(cpu_ops.items(), key=lambda x: -x[1]["total_us"])
    for name, d in sorted_cpu[:20]:
        avg = d["total_us"] / d["count"] / 1000
        print(f"{d['count']:>8}  {d['total_us']/1000:>10.1f}  {avg:>10.3f}  {name[:70]}")

    # ═══════════════════════════════════════════════════════════════
    # 6. LONG-TAIL KERNEL ANALYSIS (kernels with high max vs avg)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"KERNELS WITH HIGH VARIANCE (max/avg > 10x, possible scheduling stalls)")
    print(f"{'='*100}")
    print(f"{'Count':>8}  {'Avg ms':>10}  {'Max ms':>10}  {'Max/Avg':>8}  {'Kernel'}")
    print(f"{'-'*90}")
    for name, d in sorted_gpu:
        avg = d["total_us"] / d["count"]
        if avg > 0 and d["max_us"] / avg > 10 and d["count"] > 5:
            print(f"{d['count']:>8}  {avg/1000:>10.3f}  {d['max_us']/1000:>10.3f}  {d['max_us']/avg:>7.1f}x  {name[:60]}")

    print(f"\n{'='*100}")
    print("TRACE ANALYSIS COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
