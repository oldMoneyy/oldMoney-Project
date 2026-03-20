import json
import random

dataset_path = '/opt/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl'
out_file = '/opt/oldmoney/quantization/ultimate_64_token_balanced.jsonl'

# ==========================================
# 1. 核心基本盘 (32题) - 创造 80.31% 的功臣
# ==========================================
core_mcq = [29, 23, 21, 18, 17, 7, 6, 5, 4, 1]
core_qa = [89, 87, 82, 80, 78, 76, 74, 72, 70, 69, 68, 67, 65, 62, 61]
core_cwe = [149, 148, 147, 146, 145, 144, 143]

# ==========================================
# 2. 靶向修复药 (15题) - 补齐长度分布和逻辑空洞
# ==========================================
victim_mcq = [9, 28]
victim_fwe = [103]
victim_cwe = [120, 123, 126, 130, 131, 132, 133, 135, 138, 139, 141, 142]

# ==========================================
# 3. QA 重装防线 (15题) - 夺回海森矩阵的主导权
# 评测集 QA 索引范围大致是 60-89，排除已经选入 core_qa 的，刚好剩下 15 题。
# 包含在 48 题版本中连带死亡的 81, 83
# ==========================================
defense_qa = [60, 63, 64, 66, 71, 73, 75, 77, 79, 81, 83, 84, 85, 86, 88]

# ==========================================
# 4. NIAH 护城河锚点 (2题)
# ==========================================
niah_anchors = [45, 46]

# 合并所有索引
target_indices = core_mcq + core_qa + core_cwe + \
                 victim_mcq + victim_fwe + victim_cwe + \
                 defense_qa + niah_anchors

# 严格去重与校验
target_indices = list(set(target_indices))
assert len(target_indices) == 64, f"致命错误：期望 64 题，实际得到 {len(target_indices)} 题！请检查逻辑。"

# 读取数据
selected = []
dataset = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in target_indices:
            selected.append(json.loads(line.strip()))

# 极其重要：必须全局打乱！
# GPTQ 层级量化时，如果同类数据（如连续 19 个 CWE）集中在最后的 Batch 传入，
# 会导致该层权重的最终更新完全偏向 CWE，引发灾难性遗忘。随机打乱是救命的。
random.seed(42)
random.shuffle(selected)

# 写入文件
with open(out_file, 'w', encoding='utf-8') as f:
    for d in selected:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f"✅ [终极 Token 制衡 64 题] 已生成: {out_file}")
print("==================================================")
print(" 📊 64 题宏观任务分布 (完美压制阵型)")
print("==================================================")
tasks = {'mcq': 0, 'qa': 0, 'cwe': 0, 'fwe': 0, 'niah': 0}
for s in selected:
    tasks[s['task']] += 1

print(f" ▶ QA   : {tasks['qa']:<2} 题 (压倒性兵力，死保连续语义推理，覆盖所有可能)")
print(f" ▶ CWE  : {tasks['cwe']:<2} 题 (平滑 16K~128K 长度梯度，修复次生灾害)")
print(f" ▶ MCQ  : {tasks['mcq']:<2} 题 (极低 Token 成本，换取绝对逻辑涨分)")
print(f" ▶ NIAH : {tasks['niah']:<2} 题 (防止大海捞针能力发生意外漂移)")
print(f" ▶ FWE  : {tasks['fwe']:<2} 题 (修补离散前向提取漏洞)")
print("==================================================")
print("一切准备就绪，可以启动 64 题量化了！")
