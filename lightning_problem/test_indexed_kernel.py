import torch
from fla.ops.common.fused_recurrent import fused_recurrent_fwd

torch.manual_seed(42)

B, T, H, K, V = 1, 1, 32, 128, 128
pool_size = 8

q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g_gamma = torch.randn(H, device='cuda', dtype=torch.float32) * -0.1
scale = K ** -0.5
cu_seqlens = torch.tensor([0, 1], dtype=torch.long, device='cuda')

pool = torch.randn(pool_size, H, K, V, device='cuda', dtype=torch.float32)
h0_indices = torch.tensor([3], dtype=torch.int32, device='cuda')

# Gather the state BEFORE any kernel runs
gathered = pool[3:4].clone().contiguous()  # [1, H, K, V]

# Run GATHER path (does NOT modify pool)
o_gather, ht_gather = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma,
    scale=scale, initial_state=gathered,
    output_final_state=True, cu_seqlens=cu_seqlens,
    h0_indices=None,
)

# Run INDEXED path on a COPY of pool (so we don't interfere)
pool_copy = pool.clone()
o_indexed, _ht = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma,
    scale=scale, initial_state=pool_copy,
    output_final_state=True, cu_seqlens=cu_seqlens,
    h0_indices=h0_indices,
)

# Compare outputs
o_diff = (o_indexed - o_gather).abs().max().item()
o_rdiff = (o_indexed - o_gather).abs().mean().item()
print(f"Output max diff: {o_diff:.8e}")
print(f"Output mean diff: {o_rdiff:.8e}")

# Compare state writeback
state_indexed = pool_copy[3]   # kernel wrote back here
state_gather = ht_gather[0]    # kernel wrote to ht
s_diff = (state_indexed - state_gather).abs().max().item()
s_rdiff = (state_indexed - state_gather).abs().mean().item()
print(f"State max diff: {s_diff:.8e}")
print(f"State mean diff: {s_rdiff:.8e}")

# Verify pool[3] was actually modified (not all zeros or unchanged)
pool_changed = (pool_copy[3] - pool[3]).abs().max().item()
print(f"Pool[3] was modified: {pool_changed > 0} (diff={pool_changed:.8e})")

# Print some values for sanity
print(f"o_gather[:5]: {o_gather.flatten()[:5].tolist()}")
print(f"o_indexed[:5]: {o_indexed.flatten()[:5].tolist()}")
print(f"ht_gather[0,0,0,:5]: {ht_gather[0,0,0,:5].tolist()}")
print(f"pool_copy[3,0,0,:5]: {pool_copy[3,0,0,:5].tolist()}")
