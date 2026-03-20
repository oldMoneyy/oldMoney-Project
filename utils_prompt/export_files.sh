#!/bin/bash

OUTPUT_FILE="/opt/oldMoney-Project/utils_prompt/reference_files.txt"

# 清空或创建目标文件
> "$OUTPUT_FILE"

# 定义所有需要导出的文件绝对路径
FILES=(
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/models/minicpm.py"
    "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/flashinfer/fp4_quantization.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/__init__.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/petit_utils.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/petit.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/modelopt_utils.py"
    "/opt/oldMoney-Project/sglang_sala_cp/sglang/srt/layers/quantization/modelopt_quant.py"
    "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages/sgl_kernel/gemm.py"
    "/opt/oldMoney-Project/quantization/quantize_gptq_sparse_cpu.py"
    "/opt/model/config.json"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo -e "\n================================================================================" >> "$OUTPUT_FILE"
        echo "FILE_PATH: $f" >> "$OUTPUT_FILE"
        echo -e "================================================================================\n" >> "$OUTPUT_FILE"
        cat "$f" >> "$OUTPUT_FILE"
        echo "Successfully exported: $f"
    else
        echo "WARNING: File not found - $f"
    fi
done

echo ""
echo "All done! Check your file at: $OUTPUT_FILE"