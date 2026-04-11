import json
import argparse
import torch
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    DynamicType,
)
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

def parse_args():
    parser = argparse.ArgumentParser(description="NVFP4 W4A4 GPTQ quantization via llm-compressor")
    parser.add_argument("--input", type=str, default="/opt/model", help="Input model path")
    parser.add_argument("--output", type=str, default="/opt/model_nvfp4_w4a4_gptq", help="Output path")
    parser.add_argument("--calib-data", type=str,
                        default="/opt/oldMoney-Project/quantization/calibration_dense/calib_dense_96.jsonl",
                        help="Calibration data JSONL path")
    parser.add_argument("--max-samples", type=int, default=96, help="Max calibration samples")
    parser.add_argument("--max-len", type=int, default=131072, help="Max sequence length")
    parser.add_argument("--group-size", type=int, default=16, help="Quantization group size")
    parser.add_argument("--dense-len", type=int, default=655360, help="Dense attention length override")
    parser.add_argument("--bf16-layers", type=str, default="0,29,30,31",
                        help="Comma-separated layer indices to keep in BF16 (e.g. '0,29,30,31')")
    parser.add_argument("--ignore", type=str, default="lm_head",
                        help="Comma-separated module names to ignore (e.g. 'lm_head')")
    parser.add_argument("--smooth-alpha", type=float, default=None,
                        help="SmoothQuant alpha (None=disabled)")
    parser.add_argument("--actorder", type=str, default="static",
                        choices=["static", "dynamic", "weight"],
                        help="GPTQ activation ordering")
    parser.add_argument("--no-activation-quant", action="store_true",
                        help="Skip activation quantization (W4A16 instead of W4A4)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("NVFP4 GPTQ Quantizer via llm-compressor")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    config = AutoConfig.from_pretrained(args.input, trust_remote_code=True)
    if hasattr(config, "sparse_config") and isinstance(config.sparse_config, dict):
        config.sparse_config["dense_len"] = args.dense_len

    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Load calibration data
    samples = []
    with open(args.calib_data, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            obj = json.loads(line.strip())
            text = obj.get("question", obj.get("text", ""))
            samples.append({"text": text})

    ds = Dataset.from_list(samples)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Build quantization config
    weight_args = QuantizationArgs(
        num_bits=4,
        actorder=None,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=args.group_size,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
        observer="memoryless_minmax",
    )

    nvfp4_config = dict(weights=weight_args, targets=["Linear"])

    if not args.no_activation_quant:
        nvfp4_config["input_activations"] = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            symmetric=True,
            dynamic=DynamicType.LOCAL,
            group_size=args.group_size,
            observer="static_minmax",
            scale_dtype=FP8_E4M3_DATA.dtype,
            zp_dtype=FP8_E4M3_DATA.dtype,
        )

    # Build ignore list
    ignore = [x.strip() for x in args.ignore.split(",") if x.strip()]

    if args.bf16_layers:
        for layer_idx in args.bf16_layers.split(","):
            layer_idx = layer_idx.strip()
            if layer_idx:
                ignore.append(f"re:model.layers.{layer_idx}\\..*")

    print(f"  Ignore list: {ignore}")

    recipe = GPTQModifier(config_groups={"group_0": nvfp4_config}, ignore=ignore)

    # Build optional modifiers list
    modifiers = [recipe]

    if args.smooth_alpha is not None:
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        smooth = SmoothQuantModifier(smoothing_strength=args.smooth_alpha)
        modifiers = [smooth, recipe]

    oneshot(
        model=model,
        dataset=ds,
        recipe=modifiers if len(modifiers) > 1 else recipe,
        max_seq_length=args.max_len,
        num_calibration_samples=args.max_samples,
        trust_remote_code_model=True,
    )

    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
