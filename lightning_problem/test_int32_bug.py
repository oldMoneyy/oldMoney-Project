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
pool = torch.randn(65, H, K, V, device='cuda', dtype=torch.float32)
gathered = pool[3:4].clone()

o_ref, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=gathered, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)

# INT32 (what server uses)
pool_a = pool.clone()
o_i32, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool_a, output_final_state=True,
    cu_seqlens=cu_seqlens,
    h0_indices=torch.tensor([3], dtype=torch.int32, device='cuda'),
)

# INT64
pool_b = pool.clone()
o_i64, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool_b, output_final_state=True,
    cu_seqlens=cu_seqlens,
    h0_indices=torch.tensor([3], dtype=torch.int64, device='cuda'),
)

print(f"int32 has NaN: {torch.isnan(o_i32).any().item()}")
print(f"int64 has NaN: {torch.isnan(o_i64).any().item()}")
print(f"int32 vs ref: {(o_i32 - o_ref).abs().max().item():.8e}")
print(f"int64 vs ref: {(o_i64 - o_ref).abs().max().item():.8e}")
