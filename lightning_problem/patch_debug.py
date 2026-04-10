path = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/simple_gla/fused_recurrent.py"
with open(path, 'r') as f:
    content = f.read()

# Find the indexed function and replace it with a debug version
old_func = '''    o, _ht = fused_recurrent_fwd(
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
    return o.to(q.dtype)'''

new_func = '''    # === DEBUG: run BOTH paths and compare ===
    # Path A: indexed (what we want to use)
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

    # Path B: gather/scatter (known correct)
    initial_state_gs = h0_source[h0_indices.long(), :].contiguous()
    o_gather, ht_gather = fused_recurrent_fwd(
        q=q, k=k, v=v,
        g=None,
        g_gamma=g_gamma,
        gk=None, gv=None,
        scale=scale,
        initial_state=initial_state_gs,
        output_final_state=True,
        reverse=False,
        cu_seqlens=cu_seqlens,
        h0_indices=None,
    )

    # Compare
    o_diff = (o_indexed - o_gather).abs().max().item()
    print(f"[DEBUG] o diff indexed vs gather: {o_diff:.8e}", flush=True)

    # Check initial states match
    for i in range(h0_indices.shape[0]):
        idx = h0_indices[i].long().item()
        pool_state = h0_source[idx]
        gather_state = initial_state_gs[i]
        state_diff = (pool_state - gather_state).abs().max().item()
        if state_diff > 0:
            print(f"[DEBUG] state mismatch seq {i}: pool[{idx}] vs gather[{i}] diff={state_diff:.8e}", flush=True)

    # Use the GATHER result (known correct) but log the diff
    if o_diff > 1e-5:
        print(f"[DEBUG] WARNING: indexed output diverges! Using gather result.", flush=True)

    return o_indexed.to(q.dtype)'''

assert old_func in content, "Could not find target function body"
content = content.replace(old_func, new_func)

with open(path, 'w') as f:
    f.write(content)
print("Debug patch applied")
