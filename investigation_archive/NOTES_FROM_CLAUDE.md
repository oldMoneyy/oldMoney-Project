# Notes from Claude: Inferences, Gaps, and Verification Status

This file documents what was assumed vs verified when producing RESULTS.md and FINDINGS.md, so the reader knows exactly where to be skeptical.

---

## 1. Hardware identification

**Status: UNVERIFIED**

The archive contains no nvidia-smi output, no GPU model string, and no CUDA compute capability identifier. I grepped both server logs for `H100`, `H800`, `A100`, `A800`, `compute_cap`, `Device 0:`, `GPU 0:`, `cuda capability`, `sm_`, `nvidia`, and `NVIDIA` -- all returned zero matches.

The ~80GB inference comes from: KV cache allocates 27.90 GB K + 27.90 GB V = 55.80 GB, model weights ~5.78 GB, Mamba state ~3.05 GB, CUDA graphs ~0.60 GB, and 12.70 GB remains available after all allocations. Total: ~78 GB used + 12.70 GB free = ~80 GB. This is consistent with H100-80GB, H800-80GB, or A800-80GB, but I cannot distinguish between them from the archived data.

An earlier draft of RESULTS.md incorrectly stated "H100 80GB" as a fact. This was corrected to "unverified."

## 2. CUDA graph capture time discrepancy

**Status: INVESTIGATED AND REMOVED**

The raw server logs show:
- Dense server (31333): CUDA graph capture took 73.32s (at 08:57:58, first server started on this GPU session)
- Sparse server (31335): CUDA graph capture took 3.64s (at 09:23:12, ~26 minutes after dense server started)

The 20x difference is almost certainly because the dense server was the first process to capture CUDA graphs on this GPU, paying the full cost of CUDA driver JIT compilation and FlashInfer wrapper planning. By the time the sparse server started, the GPU's planning cache was warm. This is a cold-vs-warm artifact, not a sparse-vs-dense difference.

An earlier draft of RESULTS.md included this as a configuration comparison row. It was removed as misleading.

## 3. Phase breakdown coverage

**Status: PARTIAL**

Phase breakdown was successfully computed for workloads A and B using server log time windows inferred from matrix_progress.log:
- A1: 09:37:02 to 10:02:39 (sparse server log)
- A2: 10:02:39 to 10:28:12 (dense server log)
- B1: 10:28:12 to 10:32:51 (sparse server log)
- B2: 10:32:51 to 10:37:28 (dense server log)

Phase breakdown for workload C is **not available**. The C tests (64_concurrency_sparse.log, 64_concurrency_dense.log) ran after the A/B matrix and the bs sweep. The server logs are truncated to the last ~2000 lines and end at 10:32:49 (sparse) and 10:37:28 (dense), before the C tests would have started. The bs_progress.log shows the sweep started at 11:45:14, and the C tests likely ran even later.

## 4. Workload B phase split vs stated expectation

The user expected "~85% prefill, ~15% decode regardless of sparse/dense." This is confirmed for the "regardless of sparse/dense" part (both A1/A2 show identical 85.5/14.5 splits, both B1/B2 show identical 17.5/82.5 splits). However, the split is **not** 85/15 for workload B -- short inputs with relatively more output tokens produce an 82.5% decode-dominated profile. I reported the actual numbers; the "regardless" statement is true across sparse/dense but not across workloads.

## 5. sparse_debug_tail.log

**Status: EMPTY, EXPLAINED**

The archive description mentions "a truncated sparse debug log showing filtered_tokens / full_prefix ratios per layer per chunk." The actual file is empty (0 bytes of content). The git history shows commits removing debug instrumentation (`cuda.synchronize` + `/tmp/sparse_debug.log` writes) before the final benchmarking run. The debug data was generated during an earlier investigation phase and was not present during the archived benchmark runs.

## 6. bs=1 / bs=8 sweep data

**Status: MISSING**

- A1_bs1_sparse.log exists but contains only a Python traceback (ImportError: libcusparseLt.so.0). No benchmark results were produced.
- No files matching A2_bs1, A1_bs8, or A2_bs8 exist in the archive.
- The bs_progress.log contains only one entry: `[Sat Apr 18 11:45:14 UTC 2026] S1 sparse`, suggesting the sweep was started but did not complete (or subsequent entries were lost).

This is the most significant data gap. The S8 regime is where the sparse-vs-dense tradeoff under moderate concurrency would be measured, and it has no data.

## 7. Retokenized output token counts

In the concurrency benchmarks, "Total generated tokens (retokenized)" differs from "Total generated tokens" in some runs:
- A1 sparse: 108,722 generated vs 108,753 retokenized (+0.03%)
- A2 dense: 108,722 generated vs 107,350 retokenized (-1.3%)
- C sparse: 216,805 generated vs 204,087 retokenized (-5.9%)
- C dense: 216,805 generated vs 199,282 retokenized (-8.1%)

The retokenized count is sglang's check of re-encoding the output text. The discrepancies on workload C are larger than expected and could indicate tokenizer edge cases or output truncation. I used the "Total generated tokens" (not retokenized) numbers throughout RESULTS.md because those are what the throughput calculations are based on, but the discrepancy on C is worth noting.

## 8. Competition bench input distributions

The input data was generated by generate_compare_expr.py with:
- A: 32 requests, inputs 128K-512K, output distribution matching competition spec
- B: 64 requests, inputs 100-32K, output distribution matching competition spec
- C: 64 requests from competition_bench_64.jsonl (mixed distribution, long+short)

I verified the A and B distributions from the script source but did not read the JSONL files to confirm actual token counts. The benchmark logs report total input/output tokens that are consistent with the script's intent.

## 9. Two-server interference

Both servers (sparse on 31335, dense on 31333) were running on the same node simultaneously. The matrix_progress.log shows tests were run sequentially (A1 finished before A2 started, etc.), so they were not competing for GPU at the same time. However, the idle server still consumes ~5.78 GB of GPU memory for model weights. This was confirmed by both servers showing identical `avail mem=12.70 GB` after initialization.

The sweep tests (sparse_critical_sweep.py) explicitly note "Sequential firing (never parallel) to avoid GPU contention" -- the script fires one request at a time, alternating ports.

## 10. Node identity

The node name `sft-57-1676929-master-0` and IP `10.249.32.26` are mentioned in the user's correction request but do not appear in any archived file. I included them in the environment context as instructed but cannot independently verify them from the archive.
