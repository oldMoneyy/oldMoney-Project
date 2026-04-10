path = "/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Find the debug block we inserted
assert 'gather/scatter for debug' in lines[1585], f"Line 1586: {lines[1585]!r}"

# Replace with original indexed path
new_block = [
    "            # === INDEXED PATH: no gather/scatter copies ===\n",
    "            # Decode always has valid state in pool (set during prefill)\n",
    "            o = fused_recurrent_simple_gla_indexed(\n",
    "                q=q,\n",
    "                k=k,\n",
    "                v=v,\n",
    "                g_gamma=g_gamma,\n",
    "                scale=scale,\n",
    "                h0_source=layer_cache.temporal,\n",
    "                h0_indices=mamba_indices,\n",
    "                output_final_state=True,\n",
    "                cu_seqlens=cu_seqlens,\n",
    "            )\n",
]

# The debug block replaced 13 lines (1585-1597) with variable number of lines
# Find the 'else:' line that follows
else_idx = None
for i in range(1585, min(1610, len(lines))):
    if lines[i].strip() == 'else:' and 'Prefill' in lines[i+1]:
        else_idx = i
        break
assert else_idx is not None, "Could not find 'else: # Prefill' line"
print(f"Found 'else:' at line {else_idx+1}")

lines[1585:else_idx] = new_block

with open(path, 'w') as f:
    f.writelines(lines)
print("Indexed path restored")
