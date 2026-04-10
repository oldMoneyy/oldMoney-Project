path = "/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Lines 1584-1597 (0-indexed 1583-1596) are the indexed decode block
# Replace with gather/scatter decode (same as baseline cp does)
old = lines[1584]  # line 1585: "            # === INDEXED PATH..."
assert 'INDEXED PATH' in old, f"Line 1585 mismatch: {old!r}"

new_block = '''            # === TEMPORARILY DISABLED: using gather/scatter for debug ===
            initial_state = layer_cache.temporal[mamba_indices, :].contiguous()
            o, final_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=g_gamma,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
            if final_state is not None:
                layer_cache.temporal[mamba_indices, :] = final_state
'''

# Replace lines 1585-1597 (0-indexed 1584-1596)
lines[1584:1597] = [new_block]

with open(path, 'w') as f:
    f.writelines(lines)
print("Decode path replaced with gather/scatter")
