path = "/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
with open(path, 'r') as f:
    lines = f.readlines()

# === Edit 1: Line 17 (0-indexed: 16) — change import ===
old_line17 = lines[16]
assert 'fused_recurrent_simple_gla_indexed' in old_line17, f"Line 17 mismatch: {old_line17!r}"
lines[16] = "from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla_indexed\n"
print(f"Edit 1: Replaced line 17")
print(f"  OLD: {old_line17.rstrip()}")
print(f"  NEW: {lines[16].rstrip()}")

# === Edit 2: Line 1596 (0-indexed: 1595) — store_final_state -> output_final_state ===
old_line1596 = lines[1595]
assert 'store_final_state=True' in old_line1596, f"Line 1596 mismatch: {old_line1596!r}"
lines[1595] = old_line1596.replace('store_final_state=True', 'output_final_state=True')
print(f"Edit 2: Replaced line 1596")
print(f"  OLD: {old_line1596.rstrip()}")
print(f"  NEW: {lines[1595].rstrip()}")

with open(path, 'w') as f:
    f.writelines(lines)

# === Verify ===
with open(path, 'r') as f:
    verify = f.readlines()

v17 = verify[16]
assert 'from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla_indexed' in v17, f"Verify line 17 FAILED: {v17!r}"
print("Verify line 17: OK")

v1596 = verify[1595]
assert 'output_final_state=True' in v1596, f"Verify line 1596 FAILED: {v1596!r}"
assert 'store_final_state' not in v1596, f"Verify line 1596 still has store_final_state: {v1596!r}"
print("Verify line 1596: OK")

print("File 3 patched and verified successfully")
