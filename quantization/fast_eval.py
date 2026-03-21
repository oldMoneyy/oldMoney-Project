import os
import json
import math
import argparse
import numpy as np
import requests

# 测试用的一组 Prompt（覆盖长文本、推理、代码等不同场景）
TEST_PROMPTS = [
    "Please explain the concept of quantum entanglement in simple terms.",
    "Write a Python script to perform binary search on a sorted array.",
    "If Jane has 3 apples and gives 1 to Bob, and Bob gives 2 to Alice who already had 5, how many apples does Alice have? Let's think step by step.",
    "Translate the following English text to Chinese: 'The quick brown fox jumps over the lazy dog.'"
]

def get_logprobs_from_sglang(api_base, model_name, prompt, max_tokens=128, top_k=256):
    """
    通过 SGLang (OpenAI 兼容 API) 拿到生成过程中的 Top-K logprobs
    """
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0, # 必须是 Greedy 解码，保证生成路径一致
        "logprobs": True,
        "top_logprobs": top_k
    }
    
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    # 提取 logprobs 列表，每一个元素代表生成的一个 Token
    content_logprobs = data["choices"][0]["logprobs"]["content"]
    
    step_distributions = []
    for token_obj in content_logprobs:
        # 将 logprob (对数概率) 转换为真正的概率 (e^x)
        top_list = token_obj["top_logprobs"]
        prob_dict = {item["token"]: math.exp(item["logprob"]) for item in top_list}
        step_distributions.append(prob_dict)
        
    return step_distributions

def compute_metrics(base_dist, quant_dist):
    """
    计算单个 Token 步骤的 KL 散度 和 余弦相似度
    处理了曹议提到的 "剩余概率平滑分配" 问题
    """
    # 找到在两个分布中出现过的所有 Token 的集合的并集
    all_tokens = set(base_dist.keys()).union(set(quant_dist.keys()))
    
    # 平滑系数：对于没有出现在 Top-256 的 Token，给予一个极小的默认概率
    EPSILON = 1e-7 
    
    P = [] # 基线 (BF16)
    Q = [] # 量化 (NVFP4)
    
    for tok in all_tokens:
        P.append(base_dist.get(tok, EPSILON))
        Q.append(quant_dist.get(tok, EPSILON))
        
    P = np.array(P)
    Q = np.array(Q)
    
    # 归一化，保证概率总和为 1.0
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # 1. 计算 KL 散度: sum(P * log(P/Q))
    # KL越接近0越好
    kl_div = np.sum(P * np.log(P / Q))
    
    # 2. 计算 余弦相似度
    # 越接近1.000000越好
    cos_sim = np.dot(P, Q) / (np.linalg.norm(P) * np.linalg.norm(Q))
    
    return kl_div, cos_sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "eval"], required=True, help="baseline(提取BF16特征) 或 eval(对比量化模型)")
    parser.add_argument("--api-base", default="http://127.0.0.1:31333", help="SGLang API 地址")
    parser.add_argument("--model-name", default="openbmb/MiniCPM-SALA", help="模型名称")
    parser.add_argument("--baseline-file", default="bf16_baseline_logprobs.json", help="基线文件保存路径")
    args = parser.parse_args()

    if args.mode == "baseline":
        print(">>> [Phase 1] 正在采集 BF16 基线数据...")
        baseline_data = {}
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"  -> 生成 Prompt {i+1}/{len(TEST_PROMPTS)}...")
            dist = get_logprobs_from_sglang(args.api_base, args.model_name, prompt)
            baseline_data[f"prompt_{i}"] = dist
            
        with open(args.baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False)
        print(f"✅ 基线数据已保存至纯文本文件 (0 GPU占用): {args.baseline_file}")

    elif args.mode == "eval":
        if not os.path.exists(args.baseline_file):
            print("❌ 找不到基线文件，请先启动 BF16 模型并运行 --mode baseline")
            return
            
        print(">>> [Phase 2] 正在加载基线并测试 NVFP4 量化精度...")
        with open(args.baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            
        all_kl = []
        all_cos = []
        
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"  -> 测试 Prompt {i+1}/{len(TEST_PROMPTS)}...")
            base_dists = baseline_data[f"prompt_{i}"]
            quant_dists = get_logprobs_from_sglang(args.api_base, args.model_name, prompt)
            
            # 对齐生成的长度 (防止量化模型提前输出 EOS)
            min_len = min(len(base_dists), len(quant_dists))
            
            step_kl = []
            step_cos = []
            for step in range(min_len):
                kl, cos = compute_metrics(base_dists[step], quant_dists[step])
                step_kl.append(kl)
                step_cos.append(cos)
                
            avg_kl = np.mean(step_kl)
            avg_cos = np.mean(step_cos)
            all_kl.append(avg_kl)
            all_cos.append(avg_cos)
            print(f"     [Result] KL 散度: {avg_kl:.6f} | 余弦相似度: {avg_cos:.6f}")
            
        print("\n" + "="*50)
        print("🎯 最终量化质量体检报告")
        print("="*50)
        print(f"Average KL Divergence ↓ : {np.mean(all_kl):.6f} (越接近0越好，< 0.001 为极佳)")
        print(f"Average Cosine Sim  ↑ : {np.mean(all_cos):.6f} (越接近1越好，> 0.999 为极佳)")
        print("="*50)

if __name__ == "__main__":
    main()