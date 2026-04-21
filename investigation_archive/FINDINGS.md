# Findings: Sparse vs Dense Attention under Concurrency

## Core finding

Sparse attention delivers 5-51% wall-clock speedup on single requests (bs=1) for inputs above ~70K tokens, scaling roughly linearly with input length. At bs=32/64 concurrency with all requests fired simultaneously, this advantage disappears: sparse and dense are within 1% of each other on total throughput and benchmark duration across all three workloads (A=long, B=short, C=mixed).

## Why the speedup disappears at high concurrency

At bs=1, sparse attention reduces the number of KV tokens each query head attends to during prefill. For a 420K-token input, sparse selects ~topk=64 blocks instead of scanning the full prefix, cutting attention FLOPs by roughly 50% and delivering a 51% speedup.

At bs=32/64, the server processes 32 concurrent long-input requests. Each request's prefill is chunked into 32K-token pieces (chunked_prefill_size=32768). The GPU is saturated by the GEMM workload from QKV projections, FFN layers (including the 24 Lightning-attn layers that are always dense), and the MoE routing -- all of which are identical between sparse and dense. The attention kernel itself (8 out of 32 layers, only during prefill) becomes a small fraction of per-chunk wall time. Reducing its cost by 50% translates to a negligible reduction in overall batch throughput.

Concretely, the phase breakdown shows prefill occupies 85.5% of wall-clock for the long-input workload, but this prefill time is dominated by the non-attention components (linear projections, LayerNorm, Lightning-attn, MoE). The attention FLOPs that sparse eliminates are a small share of total prefill FLOPs in this hybrid architecture.

Additionally, at high concurrency the scheduler interleaves prefill chunks from different requests with decode steps. The per-request prefill latency is spread across the full benchmark duration, so reducing one request's attention cost doesn't free the GPU earlier -- it just slightly shortens one chunk within a long pipeline of interleaved chunks.

## TTFT effect at bs=32

While total throughput and duration are within noise, sparse shows a notable TTFT advantage on workload A: mean TTFT is 10.8% lower (452s vs 507s) and median TTFT is 28.2% lower (323s vs 450s). This suggests sparse attention does help individual requests reach their first token faster, even though this doesn't translate to aggregate throughput because the GPU is immediately occupied by the next request's prefill chunk.

This TTFT difference is only visible at bs=32 (not bs=64 C workload) and only for long inputs. It may matter for latency-sensitive applications even when throughput is unchanged.

## DENSE_LEN gate validation

The dense_len=65536 threshold correctly bypasses sparse attention for short inputs. Workload B (all inputs <32K tokens) shows a 0.03% difference between sparse and dense servers -- within measurement noise. The sparse server incurs no overhead on requests that don't need sparse attention.

## Implications for S1/S8/S64 competition scoring

- **S1 (single request)**: Sparse delivers clear wins on long inputs. At 300K tokens, 51% speedup. This is the regime where sparse attention is designed to shine.
- **S8 (batch size 8)**: No data in the archive. The bs=1 sweep crashed (libcusparseLt.so.0 missing), and no bs=8 logs exist. S8 is the most uncertain regime -- it sits between "GPU idle enough for sparse to help" and "GPU saturated enough to hide it." The crossover likely depends on input length distribution.
- **S64 (batch size 64)**: Sparse provides no throughput benefit. On workload C (mixed distribution), sparse is actually 1.0% slower, likely due to the overhead of sparse block selection and index computation on inputs that are long enough to trigger sparse but not long enough for the attention savings to dominate.

For a competition scoring metric that weights S1/S8/S64 equally, sparse attention helps on S1, is neutral-to-slightly-negative on S64, and is unknown on S8. The net impact depends on the weighting formula and the S8 crossover point.

## What remains uncertain

1. **S8 numbers**: No bs=8 data exists in the archive. The bs=1 sweep script crashed before producing results. This is the critical gap -- the S8 regime likely determines whether sparse is net-positive or net-neutral for the combined competition score.

2. **FA3 decode backend**: Both servers use nsa_decode_backend=fa3 (FlashAttention-3). The interaction between FA3 decode kernels and sparse prefill was not independently evaluated. FA3 may have different performance characteristics than FlashInfer's native decode path.

3. **Piecewise CUDA graph**: enable_piecewise_cuda_graph=False in both servers. Piecewise CUDA graph captures separate graphs for different prefill chunk sizes and could change the prefill-vs-overhead tradeoff at high concurrency. Not tested.

4. **Larger chunked_prefill_size**: Both servers use chunked_prefill_size=32768. A larger chunk size would process more tokens per prefill step, potentially increasing the attention kernel's share of per-chunk time and recovering some sparse advantage at high concurrency. Not tested.

5. **Non-monotonic speedup in the 70K-105K range**: The fine-grained sweep shows speedup dropping from 5.3% at 70K tokens to 3.4% at 98K, then jumping to 13.4% at 105K. This suggests a threshold effect (possibly the dense_len=65536 gate activating at a different actual-token boundary than expected, or a chunked-prefill boundary effect). The mechanism is unclear.

6. **GPU SKU**: No nvidia-smi output or GPU model identifier exists in the archived logs. The ~80GB memory inference is consistent with H100/H800/A800, but the specific SKU affects absolute FLOP rates and could shift the concurrency crossover point.

7. **sparse_debug_tail.log**: The archived file is empty (1 line). The filtered_tokens/full_prefix ratios per layer per chunk that were mentioned as being in the archive are not present, suggesting the debug instrumentation was removed before the final benchmarking run (consistent with the git history showing debug probe removal).
