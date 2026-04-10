import torch
from fla.ops.common.fused_recurrent import fused_recurrent_fwd

torch.manual_seed(42)
B, T, H, K, V = 1, 1, 32, 128, 128

q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g_gamma = torch.randn(H, device='cuda', dtype=torch.float32) * -0.1
scale = K ** -0.5
cu_seqlens = torch.tensor([0, 1], dtype=torch.long, device='cuda')

pool = torch.randn(8, H, K, V, device='cuda', dtype=torch.float32)
gathered = pool[3:4].clone()

# Reference: non-indexed
o_ref, ht_ref = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=gathered, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)

# Test A: int64 indices
pool_a = pool.clone()
o_a, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool_a, output_final_state=True,
    cu_seqlens=cu_seqlens,
    h0_indices=torch.tensor([3], dtype=torch.int64, device='cuda'),
)

# Test B: int32 indices
pool_b = pool.clone()
o_b, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool_b, output_final_state=True,
    cu_seqlens=cu_seqlens,
    h0_indices=torch.tensor([3], dtype=torch.int32, device='cuda'),
)

print(f"ref[:3]:   {o_ref.flatten()[:3].tolist()}")
print(f"int64[:3]: {o_a.flatten()[:3].tolist()}")
print(f"int32[:3]: {o_b.flatten()[:3].tolist()}")
print(f"int64 vs ref max diff: {(o_a - o_ref).abs().max().item():.8e}")
print(f"int32 vs ref max diff: {(o_b - o_ref).abs().max().item():.8e}")
print(f"int64 has NaN: {torch.isnan(o_a).any().item()}")
print(f"int32 has NaN: {torch.isnan(o_b).any().item()}")

# Also test with pool_size=65 to match server
pool65 = torch.randn(65, H, K, V, device='cuda', dtype=torch.float32)
gathered65 = pool65[0:1].clone()
o_ref65, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=gathered65, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)
pool65c = pool65.clone()
o_idx65, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool65c, output_final_state=True,
    cu_seqlens=cu_seqlens,
    h0_indices=torch.tensor([0], dtype=torch.int32, device='cuda'),
)
print(f"\npool_size=65, int32, index=0:")
print(f"  ref[:3]:   {o_ref65.flatten()[:3].tolist()}")
print(f"  idx[:3]:   {o_idx65.flatten()[:3].tolist()}")
print(f"  max diff:  {(o_idx65 - o_ref65).abs().max().item():.8e}")
print(f"  has NaN:   {torch.isnan(o_idx65).any().item()}")
