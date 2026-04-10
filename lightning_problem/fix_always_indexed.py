path = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/common/fused_recurrent.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Find "h0 = initial_state" in fused_recurrent_fwd (around line 355)
target = None
for i, line in enumerate(lines):
    if line.strip() == 'h0 = initial_state' and i > 300:
        target = i
        break

assert target is not None, "Could not find 'h0 = initial_state'"
print(f"Found 'h0 = initial_state' at line {target+1}")

# Insert auto-creation of identity h0_indices after this line
insert = [
    "    # Always use h0_indices to ensure same Triton binary (same FP rounding)\n",
    "    if h0_indices is None and h0 is not None:\n",
    "        h0_indices = torch.arange(N, device=h0.device, dtype=torch.int64)\n",
]

lines[target+1:target+1] = insert

with open(path, 'w') as f:
    f.writelines(lines)

# Verify
with open(path, 'r') as f:
    content = f.read()
assert 'torch.arange(N, device=h0.device' in content
print("Auto-identity h0_indices inserted")
