import torch
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelopt.torch.export import export_hf_checkpoint

model = AutoModelForCausalLM.from_pretrained(
    "/opt/model",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

# Quantize — no forward_loop needed for weight-only!
model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG)

# Export
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir="/opt/model_nvfp4")

# Copy tokenizer + custom code
AutoTokenizer.from_pretrained("/opt/model").save_pretrained("/opt/model_nvfp4")
import shutil, glob
for f in glob.glob("/opt/model/*.py"):
    shutil.copy2(f, "/opt/model_nvfp4")