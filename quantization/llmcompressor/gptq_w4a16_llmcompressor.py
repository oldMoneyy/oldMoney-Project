import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

def parse_args():
    parser = argparse.ArgumentParser(description="GPTQ W4A16 quantization via llm-compressor")
    parser.add_argument("--input", type=str, default="/opt/model")
    parser.add_argument("--output", type=str, default="/opt/model_gptq_w4a16_llmcompressor")
    parser.add_argument("--calib-data", type=str,
                        default="/opt/oldMoney-Project/quantization/calibration_dense/calib_dense_96.jsonl")
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dense-len", type=int, default=655360)
    parser.add_argument("--bf16-layers", type=str, default="",
                        help="Comma-separated layer indices to keep in BF16 (e.g. '0,29,30,31')")
    parser.add_argument("--ignore", type=str, default="lm_head",
                        help="Comma-separated module names to ignore")
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--actorder", type=str, default=None)
    parser.add_argument("--sym", action="store_true", default=True)
    parser.add_argument("--no-sym", dest="sym", action="store_false")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("GPTQ W4A16 Quantizer via llm-compressor")
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

    ignore = [x.strip() for x in args.ignore.split(",") if x.strip()]
    if args.bf16_layers:
        for layer_idx in args.bf16_layers.split(","):
            layer_idx = layer_idx.strip()
            if layer_idx:
                ignore.append(f"re:model.layers.{layer_idx}\\..*")

    print(f"  Ignore list: {ignore}")

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=ignore,
        dampening_frac=args.dampening_frac,
        actorder=args.actorder,
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_len,
        num_calibration_samples=args.max_samples,
        trust_remote_code_model=True,
    )

    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
