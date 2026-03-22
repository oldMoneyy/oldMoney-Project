#!/usr/bin/env python3
import os
import json
import math
import argparse
import numpy as np
import requests

# # Step 1: Start BF16 model, run baseline
# python fast_eval.py --mode baseline --api-base http://127.0.0.1:31333

# # Step 2: Start quantized model, run eval
# python /opt/oldMoney-Project/quantization/fast_eval.py --mode eval --api-base http://127.0.0.1:31333

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
    [已修复] 严格对齐信息论的 Top-K KL 散度计算
    1. 绝对不进行局部重归一化 (保持真实的置信度)
    2. 引入长尾概率质量 (Tail Probability Mass) 处理未见 Token
    """
    all_tokens = set(base_dist.keys()).union(set(quant_dist.keys()))
    
    P_top = []
    Q_top = []
    
    sum_p_top = 0.0
    sum_q_top = 0.0
    
    # 1. 提取 Top-K 交集中的真实绝对概率
    for tok in all_tokens:
        p_val = base_dist.get(tok, 0.0)
        q_val = quant_dist.get(tok, 0.0)
        P_top.append(p_val)
        Q_top.append(q_val)
        sum_p_top += p_val
        sum_q_top += q_val
        
    # 2. 计算长尾概率 (落在 Top-K 交集之外的所有词汇的概率总和)
    # 浮点数相加可能极微小地大于 1.0，使用 max(0, x) 兜底
    p_tail = max(0.0, 1.0 - sum_p_top)
    q_tail = max(0.0, 1.0 - sum_q_top)
    
    # 3. 计算 KL 散度
    kl_div = 0.0
    EPSILON = 1e-10  # 仅用于防止除以0或log(0)，不改变真实概率质量
    
    # 计算 Top 集合内的 KL 惩罚
    for p, q in zip(P_top, Q_top):
        if p > 0: # 只有 P(x) > 0 时，KL 才有定义
            q_safe = max(q, EPSILON)
            kl_div += p * math.log(p / q_safe)
            
    # 计算长尾集合的 KL 惩罚 (将所有未见 Token 视为一个整体)
    if p_tail > 0:
        q_tail_safe = max(q_tail, EPSILON)
        kl_div += p_tail * math.log(p_tail / q_tail_safe)
        
    # 4. 计算 Cosine Similarity (仅供参考)
    P_array = np.array(P_top)
    Q_array = np.array(Q_top)
    norm_p = np.linalg.norm(P_array)
    norm_q = np.linalg.norm(Q_array)
    if norm_p > 0 and norm_q > 0:
        cos_sim = np.dot(P_array, Q_array) / (norm_p * norm_q)
    else:
        cos_sim = 0.0
        
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
            
        print("\n" + "="*60)
        print("🎯 最终量化质量体检报告 (严谨信息论版)")
        print("="*60)
        final_kl = np.mean(all_kl)
        final_cos = np.mean(all_cos)
        print(f"Average KL Divergence ↓ : {final_kl:.6f}")
        
        # 增加客观的评判标准提示
        if final_kl < 0.05:
            print("   ↳ [评价] 极佳 (Excellent)！分布几乎无损，生成轨迹完美对齐。")
        elif final_kl < 0.15:
            print("   ↳ [评价] 优秀 (Good)。存在微小扰动，但不影响主干逻辑和语义。")
        elif final_kl < 0.30:
            print("   ↳ [评价] 警告 (Warning)。分布有明显偏移，可能出现长文本失忆或幻觉。")
        else:
            print("   ↳ [评价] 崩盘 (Critical)。量化完全破坏了模型，底层算子或权重打包可能存在错误。")
            
        print(f"Average Cosine Sim  ↑ : {final_cos:.6f} (注：概率分布的 Cosine 仅供参考)")
        print("="*60)

if __name__ == "__main__":
    main()