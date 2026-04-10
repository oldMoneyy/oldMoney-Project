#!/bin/bash
set -e

F1="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/common/fused_recurrent.py"
F2="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/fla/ops/simple_gla/fused_recurrent.py"
F3="/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
F4="/opt/oldMoney-Project/sglang_sala_lightning/sglang/srt/layers/attention/fla/fused_recurrent_simple_gla_indexed.py"

# Backups
cp "$F1" "${F1}.orig"
cp "$F2" "${F2}.orig"
cp "$F3" "${F3}.orig2"

# Deploy patched files
cp /opt/draft/patched_file1.py "$F1"
cp /opt/draft/patched_file2.py "$F2"

# File 3: surgical line edits only (line 17 and line 1596)
# We use python with explicit line-number edits and verification
python3 /opt/draft/patch_file3_lineno.py

# Delete standalone kernel
rm -f "$F4"
rm -f "${F3}.bak"

# Clear triton cache
rm -rf ~/.triton/cache/*

# Verify imports
python3 -c "import fla.ops.common.fused_recurrent; print('File 1 import OK')"
python3 -c "from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla_indexed; print('File 2 import OK')"

# Verify no h0_indices in bwd kernel
python3 -c "
with open('$F1') as f:
    lines = f.readlines()
in_bwd = False
for i, line in enumerate(lines, 1):
    if 'def fused_recurrent_bwd_kernel(' in line:
        in_bwd = True
    if in_bwd and 'def fused_recurrent_fwd(' in line:
        in_bwd = False
    if in_bwd and 'h0_indices' in line:
        print(f'ERROR: h0_indices found in bwd kernel at line {i}: {line.rstrip()}')
        exit(1)
print('Verified: no h0_indices in bwd kernel')
"

# Show key sections of patched file 1
echo "=== Heuristics block (lines 12-17) ==="
sed -n '12,17p' "$F1"
echo "=== Kernel params h0..cu_seqlens ==="
sed -n '35,39p' "$F1"
echo "=== Constexprs IS_VARLEN..USE_H0_INDICES ==="
sed -n '53,55p' "$F1"
echo "=== Initial state load ==="
sed -n '85,92p' "$F1"
echo "=== Final state store ==="
sed -n '122,130p' "$F1"
echo "=== fused_recurrent_fwd signature end ==="
sed -n '348,351p' "$F1"
echo "=== kernel launch h0/ht/h0_indices ==="
sed -n '372,376p' "$F1"

echo ""
echo "=== File 3 line 17 ==="
sed -n '17p' "$F3"
echo "=== File 3 line 1596 ==="
sed -n '1594,1598p' "$F3"

echo ""
echo "ALL DONE"
