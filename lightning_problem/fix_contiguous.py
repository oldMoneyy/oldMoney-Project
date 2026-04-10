path = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/simple_gla/fused_recurrent.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Find "o, _ht = fused_recurrent_fwd(" in the indexed function
target_idx = None
for i, line in enumerate(lines):
    if 'o, _ht = fused_recurrent_fwd(' in line:
        target_idx = i
        break
assert target_idx is not None, "Could not find 'o, _ht = fused_recurrent_fwd('"
print(f"Found fused_recurrent_fwd call at line {target_idx+1}")

# Insert contiguous calls before it
contiguous_lines = [
    "    # Ensure contiguous layout (bypassing @input_guard which normally does this)\n",
    "    q = q.contiguous()\n",
    "    k = k.contiguous()\n",
    "    v = v.contiguous()\n",
    "    h0_source = h0_source.contiguous()\n",
    "    if h0_indices is not None:\n",
    "        h0_indices = h0_indices.contiguous()\n",
    "\n",
]

lines[target_idx:target_idx] = contiguous_lines

with open(path, 'w') as f:
    f.writelines(lines)

# Verify
with open(path, 'r') as f:
    verify = f.read()
assert 'q = q.contiguous()' in verify
assert 'o, _ht = fused_recurrent_fwd(' in verify
print("Contiguous fix applied and verified")
