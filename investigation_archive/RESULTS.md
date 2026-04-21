# Sparse vs Dense Attention Benchmark Results

## Environment context -- read before reusing any number

These results were produced on a specific hardware + software combination
on 2026-04-18. Numbers should not be assumed to generalize. Re-benchmark
before making decisions on different hardware or after any stack upgrade.

- Date: 2026-04-18
- Node: sft-57-1676929-master-0 (10.249.32.26)
- GPU: unverified (no nvidia-smi output or GPU model string in archived server logs; ~80GB inferred from memory allocations)
- Model: MiniCPM-SALA W4A16 GPTQ-Marlin (model_gptq_int4_dense_smooth)
- sglang: custom fork at oldMoney-Project/sglang_sala_flashinfer
- Sparse extensions: oldMoney-Project/vendor_flashinfer + vendor/infllmv2_cuda_impl

Known sources of run-to-run variance on the same hardware: NUMA placement,
other tenants on the node, thermal state, FlashInfer wrapper planning cache
warmth. Expect 5-10% noise even on identical hardware.

Cross-hardware generalization warnings:
- Different GPU SKUs will shift the bs=64 sparse-vs-dense crossover because
  prefill GEMM cost scales differently (A100 vs H100 vs H800 differ by 2-3x
  in bf16 TFLOPs and by 1.5-2x in HBM bandwidth).
- The bs=1 single-request speedup is more portable; the bs=64 aggregate
  finding may not replicate on different hardware.
- Different CUDA driver versions change FlashInfer kernel selection and
  wrapper planning time.

---

**Model**: MiniCPM-SALA (MiniCPM4 hybrid, W4A16 GPTQ-Marlin, 32 layers: 8 MiniCPM4-attn + 24 Lightning-attn)
**Hardware**: Single GPU (~80GB memory, model unverified -- see NOTES_FROM_CLAUDE.md)
**Runtime**: sglang with FlashInfer backend, chunked_prefill_size=32768, max_running_requests=64, radix_cache disabled
**Sparse config**: topk=64, dense_len=65536 (inputs < 65536 tokens bypass sparse), cross-layer block sharing

---

## 1. Single-Request Sweep (bs=1)

Sequential requests on idle GPU, max_output=64, 2 runs per size, best-of-2 reported.

### Baseline sweep (sparse_vs_dense.py)

| Target size | Actual tokens | Sparse (s) | Dense (s) | Speedup |
|------------|--------------|-----------|----------|---------|
| 20,000 | 28,035 | 2.13 | 2.11 | -1.0% |
| 40,000 | 56,035 | 4.48 | 4.47 | -0.2% |
| 80,000 | 112,035 | 9.47 | 10.76 | +12.0% |
| 150,000 | 210,035 | 18.21 | 26.90 | +32.3% |
| 300,000 | 420,035 | 40.48 | 83.29 | +51.4% |

### Fine-grained sweep (50K-90K region)

| Target size | Actual tokens | Sparse (s) | Dense (s) | Speedup |
|------------|--------------|-----------|----------|---------|
| 50,000 | 70,035 | 5.54 | 5.85 | +5.3% |
| 55,000 | 77,035 | 6.31 | 6.61 | +4.5% |
| 60,000 | 84,035 | 7.07 | 7.36 | +3.9% |
| 65,000 | 91,035 | 7.84 | 8.15 | +3.7% |
| 70,000 | 98,035 | 8.63 | 8.94 | +3.4% |
| 75,000 | 105,035 | 8.56 | 9.88 | +13.4% |
| 85,000 | 119,035 | 10.40 | 11.72 | +11.3% |
| 90,000 | 126,035 | 11.34 | 12.66 | +10.4% |

Crossover: sparse meaningfully faster (>=5%) starting around 50K target / 70K actual tokens.
Non-monotonic jump between 70K (3.4%) and 75K (13.4%) suggests a quantized threshold effect.

---

## 2. Concurrency Matrix (bs=32/64, all requests fired at rate=inf)

### Workload A: Long inputs (32 requests, all >128K tokens)

| Metric | A1 (Sparse) | A2 (Dense) | Delta |
|--------|------------|-----------|-------|
| Benchmark duration (s) | 1,484.97 | 1,480.56 | +0.3% (sparse slower) |
| Total token throughput (tok/s) | 5,581.87 | 5,598.48 | -0.3% |
| Input token throughput (tok/s) | 5,508.65 | 5,525.05 | -0.3% |
| Output token throughput (tok/s) | 73.21 | 73.43 | -0.3% |
| Mean TTFT (ms) | 452,077 | 506,598 | -10.8% (sparse faster) |
| Median TTFT (ms) | 322,793 | 449,585 | -28.2% (sparse faster) |
| P99 TTFT (ms) | 1,240,936 | 1,237,083 | +0.3% |
| Mean TPOT (ms) | 2,316.55 | 2,298.26 | +0.8% |
| Median TPOT (ms) | 1,318.59 | 1,285.17 | +2.6% |
| P99 TPOT (ms) | 9,151.88 | 8,819.15 | +3.8% |
| Successful requests | 32 | 32 | - |
| Total input tokens | 8,180,177 | 8,180,177 | - |
| Total output tokens | 108,722 | 108,722 | - |
| Peak concurrent requests | 32 | 32 | - |

### Workload B: Short inputs (64 requests, all <32K tokens)

| Metric | B1 (Sparse) | B2 (Dense) | Delta |
|--------|------------|-----------|-------|
| Benchmark duration (s) | 249.79 | 249.71 | +0.03% |
| Total token throughput (tok/s) | 3,601.83 | 3,603.01 | -0.03% |
| Mean TTFT (ms) | 40,960 | 41,044 | -0.2% |
| Median TTFT (ms) | 42,087 | 42,179 | -0.2% |
| P99 TTFT (ms) | 42,101 | 42,193 | -0.2% |
| Mean TPOT (ms) | 15.84 | 15.95 | -0.7% |
| Median TPOT (ms) | 16.04 | 16.21 | -1.0% |
| P99 TPOT (ms) | 64.87 | 65.33 | -0.7% |
| Successful requests | 64 | 64 | - |
| Total input tokens | 664,893 | 664,893 | - |
| Total output tokens | 234,802 | 234,802 | - |

### Workload C: Mixed distribution (64 requests, competition_bench_64.jsonl)

| Metric | C Sparse | C Dense | Delta |
|--------|---------|--------|-------|
| Benchmark duration (s) | 1,658.88 | 1,642.98 | +1.0% (sparse slower) |
| Total token throughput (tok/s) | 5,684.47 | 5,739.49 | -1.0% |
| Mean TTFT (ms) | 688,168 | 673,174 | +2.2% (sparse slower) |
| Median TTFT (ms) | 495,131 | 489,502 | +1.1% |
| P99 TTFT (ms) | 1,415,088 | 1,406,158 | +0.6% |
| Mean TPOT (ms) | 3,181.30 | 3,163.81 | +0.6% |
| Median TPOT (ms) | 695.36 | 804.75 | -13.6% |
| P99 TPOT (ms) | 31,722.54 | 31,533.60 | +0.6% |
| Successful requests | 64 | 64 | - |
| Total input tokens | 9,213,034 | 9,213,034 | - |
| Total output tokens | 216,805 | 216,805 | - |

---

## 3. Phase Breakdown (from server logs via phase_breakdown.py)

Time attribution: each gap between consecutive log lines is attributed to whichever phase (prefill/decode) was last logged.

### Workload A: Long inputs (32 requests)

| Phase | A1 Sparse | A1 % | A2 Dense | A2 % |
|-------|----------|------|---------|------|
| Prefill | 1,284s | 85.5% | 1,281s | 85.5% |
| Decode | 218s | 14.5% | 218s | 14.5% |
| **Total** | **1,502s** | | **1,499s** | |
| Prefill throughput | 6,496 tok/s | | 6,511 tok/s | |
| Decode throughput (aggregate) | 434.1 tok/s | | 435.2 tok/s | |
| Prefill batches | 255 | | 255 | |
| Decode batches | 750 | | 750 | |

### Workload B: Short inputs (64 requests)

| Phase | B1 Sparse | B1 % | B2 Dense | B2 % |
|-------|----------|------|---------|------|
| Prefill | 44s | 17.5% | 44s | 17.5% |
| Decode | 207s | 82.5% | 207s | 82.5% |
| **Total** | **251s** | | **251s** | |
| Prefill throughput | 15,161 tok/s | | 15,161 tok/s | |
| Decode throughput (aggregate) | 1,049.8 tok/s | | 1,049.9 tok/s | |
| Prefill batches | 29 | | 29 | |
| Decode batches | 781 | | 781 | |

Note: phase breakdown for workload C is unavailable -- the server logs (truncated to last ~2000 lines) do not cover the C test time window, which ran after the A/B matrix completed.

---

## 4. Batch-Size Sweep (bs=1 isolation)

**A1_bs1_sparse.log**: Crashed before producing results (ImportError: libcusparseLt.so.0 not found).
No A2_bs1, A1_bs8, or A2_bs8 logs are present in the archive.

---

## 5. Server Configuration Notes

Both servers use identical model weights and sglang configuration except:

| Parameter | Sparse (31335) | Dense (31333) |
|-----------|---------------|--------------|
| Sparse prefill | ENABLED (cross-layer block sharing, anchor=layer 0) | Not logged (default off) |
| sparse_topk_override | 64 | N/A |
| dense_len override | 65536 | N/A |

Both: GPTQ-Marlin quantization, bf16 compute, fp8_e5m2 KV cache, chunked_prefill_size=32768, max_running_requests=64, disable_radix_cache=True, attention_backend=flashinfer, nsa_prefill_backend=flashmla_sparse, nsa_decode_backend=fa3, context_len=524288.
