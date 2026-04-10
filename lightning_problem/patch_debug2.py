path = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/simple_gla/fused_recurrent.py"

# Read the entire file
with open(path, 'r') as f:
    content = f.read()

# Find and replace the entire indexed function
old_start = "def fused_recurrent_simple_gla_indexed("
assert old_start in content, "Cannot find indexed function"

# Find function start and end
idx = content.index(old_start)
# Find the next function or end of file
rest = content[idx:]
# The function ends at the next unindented def or EOF
lines_rest = rest.split('\n')
func_end = len(rest)
for i, line in enumerate(lines_rest):
    if i > 0 and line and not line.startswith(' ') and not line.startswith('#') and line.strip():
        func_end = sum(len(l)+1 for l in lines_rest[:i])
        break

new_func = '''def fused_recurrent_simple_gla_indexed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_gamma: torch.Tensor,
    scale: float,
    h0_source: torch.Tensor,
    h0_indices: torch.Tensor,
    output_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Indexed-access variant for decode. Reads/writes state directly from pool."""
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}"
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Ensure contiguous layout (bypassing @input_guard which normally does this)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    h0_source = h0_source.contiguous()
    if h0_indices is not None:
        h0_indices = h0_indices.contiguous()

    # === DEBUG: save initial state BEFORE any kernel modifies h0_source ===
    saved_initial = h0_source[h0_indices.long(), :].clone()

    # Run indexed path
    o_indexed, _ht = fused_recurrent_fwd(
        q=q, k=k, v=v,
        g=None,
        g_gamma=g_gamma,
        gk=None, gv=None,
        scale=scale,
        initial_state=h0_source,
        output_final_state=output_final_state,
        reverse=False,
        cu_seqlens=cu_seqlens,
        h0_indices=h0_indices,
    )

    # Run gather path with SAVED (pre-mutation) initial state
    o_gather, _ht2 = fused_recurrent_fwd(
        q=q, k=k, v=v,
        g=None,
        g_gamma=g_gamma,
        gk=None, gv=None,
        scale=scale,
        initial_state=saved_initial,
        output_final_state=True,
        reverse=False,
        cu_seqlens=cu_seqlens,
        h0_indices=None,
    )

    o_diff = (o_indexed - o_gather).abs().max().item()
    if o_diff > 1e-5:
        N = h0_indices.shape[0]
        print(f"[DEBUG2] o diff: {o_diff:.6e}, h0_source.shape={list(h0_source.shape)}, "
              f"h0_indices={h0_indices.tolist()}, h0_indices.dtype={h0_indices.dtype}, "
              f"N={N}, q.shape={list(q.shape)}, cu_seqlens={cu_seqlens.tolist() if cu_seqlens is not None else None}",
              flush=True)
        # Check if indexed kernel read the right initial state
        # After indexed kernel, h0_source[idx] has been overwritten with new state
        # Compare with what gather produced as final state
        for i in range(min(N, 3)):
            idx = h0_indices[i].long().item()
            pool_after = h0_source[idx]
            gather_final = _ht2[i]
            state_diff = (pool_after.float() - gather_final.float()).abs().max().item()
            print(f"[DEBUG2]   seq {i}: pool_idx={idx}, final_state diff={state_diff:.6e}", flush=True)

    return o_indexed.to(q.dtype)
'''

content = content[:idx] + new_func + content[idx+func_end:]

with open(path, 'w') as f:
    f.write(content)
print("Debug2 patch applied")
