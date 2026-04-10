path = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/common/fused_recurrent.py"
with open(path, 'r') as f:
    lines = f.readlines()

# Find the STORE_FINAL_STATE block in fwd kernel (should be around line 122-130)
# We need to revert it to ALWAYS write to ht, removing the USE_H0_INDICES branch
store_start = None
store_end = None
for i, line in enumerate(lines):
    if 'if STORE_FINAL_STATE:' in line and i < 140:
        store_start = i
    if store_start and i > store_start and (line.strip() == '' or (not line.startswith(' ') and line.strip())):
        store_end = i
        break

print(f"STORE block found at lines {store_start+1}-{store_end}")
print("Current STORE block:")
for j in range(store_start, store_end):
    print(f"  {j+1}: {lines[j].rstrip()}")

# Replace with simple non-indexed store (always write to ht)
new_store = [
    "    if STORE_FINAL_STATE:\n",
    "        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]\n",
    "        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)\n",
    "\n",
]

lines[store_start:store_end] = new_store

with open(path, 'w') as f:
    f.writelines(lines)

# Verify
with open(path, 'r') as f:
    content = f.read()
assert 'ht_idx' not in content, "ht_idx should be removed from store path"
assert 'h0_idx' in content, "h0_idx should still be in load path"
print("\nKernel store path fixed: always writes to ht")
