path = "/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Verify line 1585 (0-indexed 1584)
assert 'if is_decode:' in lines[1584], f"Line 1585: {lines[1584]!r}"
# Verify line 1599 (0-indexed 1598)
assert 'else:' in lines[1598], f"Line 1599: {lines[1598]!r}"

# Replace lines 1586-1598 (0-indexed 1585-1597) — body of if is_decode
new_body = [
    "            # === TEMPORARILY DISABLED: gather/scatter for debug ===\n",
    "            initial_state = layer_cache.temporal[mamba_indices, :].contiguous()\n",
    "            o, final_state = fused_recurrent_simple_gla(\n",
    "                q=q, k=k, v=v,\n",
    "                g_gamma=g_gamma, scale=scale,\n",
    "                initial_state=initial_state,\n",
    "                output_final_state=True,\n",
    "                cu_seqlens=cu_seqlens,\n",
    "            )\n",
    "            if final_state is not None:\n",
    "                layer_cache.temporal[mamba_indices, :] = final_state\n",
]
lines[1585:1598] = new_body

with open(path, 'w') as f:
    f.writelines(lines)

# Verify
with open(path, 'r') as f:
    v = f.readlines()
assert 'if is_decode:' in v[1584], "verify 1585 failed"
assert 'gather/scatter' in v[1585], "verify 1586 failed"
assert 'else:' in v[1596], f"else shifted wrong, line 1597: {v[1596]!r}"
print("CHECK 3 patch applied and verified")
