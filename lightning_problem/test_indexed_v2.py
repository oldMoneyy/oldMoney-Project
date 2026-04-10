import torch
from fla.ops.common.fused_recurrent import fused_recurrent_fwd

torch.manual_seed(42)
B, T, H, K, V = 1, 1, 32, 128, 128

q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.zeros(B, T, H, K, device='cuda', dtype=torch.bfloat16)  # zero k,v so output = q * initial_state
v = torch.zeros(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g_gamma = torch.zeros(H, device='cuda', dtype=torch.float32)  # no decay
scale = K ** -0.5
cu_seqlens = torch.tensor([0, 1], dtype=torch.long, device='cuda')

# TEST 1: pool size 1, index 0 — should be IDENTICAL to non-indexed
print("=== TEST 1: pool_size=1, index=0 ===")
state = torch.randn(1, H, K, V, device='cuda', dtype=torch.float32)

o_normal, ht_normal = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=state.clone(), output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)
pool1 = state.clone()
o_idx, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool1, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=torch.tensor([0], dtype=torch.int64, device='cuda'),
)
print(f"  o_normal[:3]: {o_normal.flatten()[:3].tolist()}")
print(f"  o_idx[:3]:    {o_idx.flatten()[:3].tolist()}")
print(f"  max diff: {(o_normal - o_idx).abs().max().item():.8e}")
has_nan = torch.isnan(o_idx).any().item()
print(f"  has NaN: {has_nan}")

# TEST 2: pool size 4, index 0
print("\n=== TEST 2: pool_size=4, index=0 ===")
pool4 = torch.randn(4, H, K, V, device='cuda', dtype=torch.float32)
state0 = pool4[0:1].clone()

o_normal2, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=state0, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)
pool4c = pool4.clone()
o_idx2, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool4c, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=torch.tensor([0], dtype=torch.int64, device='cuda'),
)
print(f"  o_normal[:3]: {o_normal2.flatten()[:3].tolist()}")
print(f"  o_idx[:3]:    {o_idx2.flatten()[:3].tolist()}")
print(f"  max diff: {(o_normal2 - o_idx2).abs().max().item():.8e}")
print(f"  has NaN: {torch.isnan(o_idx2).any().item()}")

# TEST 3: pool size 4, index 3
print("\n=== TEST 3: pool_size=4, index=3 ===")
state3 = pool4[3:4].clone()

o_normal3, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=state3, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)
pool4c2 = pool4.clone()
o_idx3, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool4c2, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=torch.tensor([3], dtype=torch.int64, device='cuda'),
)
print(f"  o_normal[:3]: {o_normal3.flatten()[:3].tolist()}")
print(f"  o_idx[:3]:    {o_idx3.flatten()[:3].tolist()}")
print(f"  max diff: {(o_normal3 - o_idx3).abs().max().item():.8e}")
print(f"  has NaN: {torch.isnan(o_idx3).any().item()}")

# TEST 4: Check what the kernel actually reads — write known pattern
print("\n=== TEST 4: known pattern in pool slot 2 ===")
pool_known = torch.zeros(4, H, K, V, device='cuda', dtype=torch.float32)
pool_known[2] = 1.0  # slot 2 is all 1s, everything else is 0

o_idx4, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=pool_known, output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=torch.tensor([2], dtype=torch.int64, device='cuda'),
)
# If reading correctly, output should be same as using all-ones initial state
o_ones, _ = fused_recurrent_fwd(
    q=q, k=k, v=v, g_gamma=g_gamma, scale=scale,
    initial_state=torch.ones(1, H, K, V, device='cuda', dtype=torch.float32),
    output_final_state=True,
    cu_seqlens=cu_seqlens, h0_indices=None,
)
print(f"  o_ones[:3]:   {o_ones.flatten()[:3].tolist()}")
print(f"  o_idx[:3]:    {o_idx4.flatten()[:3].tolist()}")
print(f"  max diff: {(o_ones - o_idx4).abs().max().item():.8e}")
print(f"  has NaN: {torch.isnan(o_idx4).any().item()}")
print(f"  pool_known[2] after: min={pool_known[2].min().item():.4f} max={pool_known[2].max().item():.4f}")
print(f"  pool_known[0] after: min={pool_known[0].min().item():.4f} max={pool_known[0].max().item():.4f}")
