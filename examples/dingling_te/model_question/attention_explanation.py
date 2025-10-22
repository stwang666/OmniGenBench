# -*- coding: utf-8 -*-
# 注意力机制详解与可视化

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("👁️ 注意力机制（Attention Mechanism）详解")
print("="*70)

# ============= 1. 什么是注意力？ =============
print("\n📌 1. 什么是注意力机制？")
print("-"*70)

print("""
注意力机制的核心思想：
  💡 让模型学会"关注"输入序列中的重要部分

类比人类阅读：
  "The cat sat on the mat because it was tired"
  
  当理解"it"指代什么时，人类会：
  1. 回看前面的单词
  2. 重点关注"cat"而不是"mat"或"on"
  3. 根据上下文判断"it"="cat"
  
  这就是注意力！模型学习哪些位置更重要。
""")

# ============= 2. 注意力的数学原理 =============
print("\n📌 2. 注意力的数学原理")
print("-"*70)

print("""
Self-Attention计算步骤：

步骤1: 计算Query, Key, Value
  Q = X @ W_q    # [seq_len, d_model] @ [d_model, d_k] = [seq_len, d_k]
  K = X @ W_k    # [seq_len, d_model] @ [d_model, d_k] = [seq_len, d_k]
  V = X @ W_v    # [seq_len, d_model] @ [d_model, d_v] = [seq_len, d_v]

步骤2: 计算注意力得分（相似度）
  scores = Q @ K^T / sqrt(d_k)    # [seq_len, seq_len]
  
步骤3: 归一化（Softmax）
  attention_weights = softmax(scores)    # [seq_len, seq_len]
  
步骤4: 加权求和
  output = attention_weights @ V    # [seq_len, d_v]

其中:
  - Q (Query): "我想找什么？"
  - K (Key): "我是什么？"
  - V (Value): "我的内容是什么？"
  - attention_weights: "每个位置的重要性"
""")

# ============= 3. DNA序列的注意力示例 =============
print("\n📌 3. DNA序列的注意力示例")
print("-"*70)

# 简化的DNA序列
sequence = "ATCGATCGTAGC"
seq_len = len(sequence)

print(f"DNA序列: {sequence}")
print(f"序列长度: {seq_len}\n")

# 模拟注意力权重矩阵（每个位置对其他位置的关注程度）
np.random.seed(42)
# 创建一些结构化的注意力模式
attention_matrix = np.random.rand(seq_len, seq_len)

# 让注意力集中在附近的位置（局部注意力模式）
for i in range(seq_len):
    for j in range(seq_len):
        dist = abs(i - j)
        attention_matrix[i, j] = np.exp(-dist / 2.0)  # 距离越近，注意力越大

# 归一化（每行和为1）
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

print("注意力矩阵示例:")
print("  行: 查询位置 (Query)")
print("  列: 被关注的位置 (Key)")
print("  值: 注意力权重 (0-1之间)")
print(f"  形状: [{seq_len}, {seq_len}]\n")

# 可视化注意力矩阵
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：注意力热力图
ax1 = axes[0]
im = ax1.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(seq_len))
ax1.set_yticks(range(seq_len))
ax1.set_xticklabels(list(sequence))
ax1.set_yticklabels(list(sequence))
ax1.set_xlabel('被关注的位置 (Key/Value)', fontsize=11)
ax1.set_ylabel('查询位置 (Query)', fontsize=11)
ax1.set_title('DNA序列的注意力权重矩阵\n（每个碱基对其他碱基的关注程度）', 
             fontsize=12, fontweight='bold')

# 添加colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('注意力权重', rotation=270, labelpad=20)

# 右图：某一个位置的注意力分布
query_pos = 5  # 选择位置5（G）
ax2 = axes[1]
attention_dist = attention_matrix[query_pos]
colors = ['#FF6B6B' if nt=='A' else '#4ECDC4' if nt=='T' else 
          '#45B7D1' if nt=='C' else '#95E1D3' for nt in sequence]

bars = ax2.bar(range(seq_len), attention_dist, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(seq_len))
ax2.set_xticklabels(list(sequence))
ax2.set_xlabel('序列位置', fontsize=11)
ax2.set_ylabel('注意力权重', fontsize=11)
ax2.set_title(f'位置{query_pos}的碱基"{sequence[query_pos]}"对其他位置的注意力\n'
             f'（高度表示关注程度）', 
             fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 标注最关注的3个位置
top_3_idx = np.argsort(attention_dist)[-3:]
for idx in top_3_idx:
    ax2.text(idx, attention_dist[idx], f'{attention_dist[idx]:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/attention_visualization.png', 
           dpi=300, bbox_inches='tight')
print("💾 注意力可视化已保存: attention_visualization.png\n")

# ============= 4. Multi-Head Attention =============
print("\n📌 4. Multi-Head Attention（多头注意力）")
print("-"*70)

print("""
为什么需要多头？
  🤔 单个注意力头可能只关注一种模式
  💡 多个注意力头可以同时关注多种模式

例如在DNA序列中：
  头1: 关注局部motif（如TATA box）
  头2: 关注远距离相互作用
  头3: 关注重复序列
  头4: 关注GC含量变化
  ...

实现:
  1. 将输入分成h个头
  2. 每个头独立计算注意力
  3. 拼接所有头的输出
  4. 通过线性层融合
""")

num_heads = 8
d_model = 768
d_k = d_model // num_heads  # 每个头的维度

print(f"\nOmniGenome-52M的多头注意力配置:")
print(f"  总维度 (d_model): {d_model}")
print(f"  注意力头数 (num_heads): {num_heads}")
print(f"  每个头的维度 (d_k): {d_k}")
print(f"\n计算流程:")
print(f"  输入: [batch, seq_len, {d_model}]")
print(f"  → Split成{num_heads}个头: [batch, {num_heads}, seq_len, {d_k}]")
print(f"  → 每个头计算attention: [batch, {num_heads}, seq_len, {d_k}]")
print(f"  → Concat所有头: [batch, seq_len, {d_model}]")

# ============= 5. 注意力的作用 =============
print("\n\n📌 5. 注意力机制的作用")
print("-"*70)

print("""
在转座子表达预测任务中的作用：

1️⃣ 识别重要的调控元件
   - 启动子区域（Promoter）
   - 增强子（Enhancer）
   - 转录因子结合位点（TFBS）
   
2️⃣ 捕捉长距离依赖
   - 远端调控元件与基因的互作
   - 染色质环（Chromatin Loop）
   
3️⃣ 关注序列特异性motif
   - TATA box
   - CAAT box
   - GC-rich区域
   
4️⃣ 建模碱基间相互作用
   - 配对碱基
   - 二级结构
   - 密码子使用偏好
""")

# ============= 6. 如何可视化注意力？ =============
print("\n\n📌 6. 如何可视化注意力？")
print("-"*70)

print("""
方法1: 提取Transformer的注意力权重
  
  # 在模型前向传播时
  outputs = model(input_ids, output_attentions=True)
  attentions = outputs.attentions  # Tuple of attention matrices
  
  # attentions[layer][batch, num_heads, seq_len, seq_len]
  
  # 可视化第12层的第1个头
  layer_12_head_1 = attentions[11][0, 0, :, :]  # [seq_len, seq_len]
  
  # 绘制热力图
  sns.heatmap(layer_12_head_1, cmap='viridis')

方法2: 平均多个头的注意力
  
  # 平均所有头
  avg_attention = attentions[layer].mean(dim=1)  # [batch, seq_len, seq_len]
  
  # 可视化平均注意力
  plt.imshow(avg_attention[0].cpu().numpy())

方法3: 关注特定位置的注意力
  
  # 例如：看位置50对其他位置的注意力
  attention_from_pos_50 = avg_attention[0, 50, :]
  
  # 映射回序列
  for i, (nt, att) in enumerate(zip(sequence, attention_from_pos_50)):
      print(f"{i:3d} {nt}  {'█' * int(att*50)}")
""")

# ============= 7. 实际应用示例代码 =============
print("\n\n📌 7. 实际应用代码示例")
print("-"*70)

code_example = '''
def visualize_attention_for_sequence(model, sequence, tokenizer):
    """可视化序列的注意力模式"""
    
    # 1. Tokenize
    inputs = tokenizer(sequence, return_tensors="pt")
    
    # 2. 前向传播并获取注意力
    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    
    # 3. 提取注意力权重
    attentions = outputs.attentions  # Tuple: (layer1, layer2, ...)
    
    # 4. 选择最后一层，平均所有头
    last_layer_attention = attentions[-1]  # [1, num_heads, seq_len, seq_len]
    avg_attention = last_layer_attention.mean(dim=1)[0]  # [seq_len, seq_len]
    
    # 5. 可视化
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 获取token序列（用于标签）
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    sns.heatmap(
        avg_attention.cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        ax=ax
    )
    
    ax.set_title('Attention Pattern (Average of All Heads)')
    plt.tight_layout()
    plt.savefig('attention_pattern.png', dpi=300)
    
    return avg_attention

# 使用示例
sequence = "ATCGATCGTAGCTAGCTAGC"
attention = visualize_attention_for_sequence(model, sequence, tokenizer)
'''

print(code_example)

# ============= 8. 注意力的解释 =============
print("\n📌 8. 注意力权重的生物学解释")
print("-"*70)

print("""
高注意力权重可能表示：

✅ 功能相关的motif
   如果两个位置之间有高注意力，它们可能：
   - 形成转录因子结合位点
   - 参与二级结构形成
   - 共同影响基因表达

✅ 保守区域
   重要的功能元件通常进化保守
   模型学会关注这些保守位置

✅ 组织特异性调控
   不同注意力头可能对应不同组织
   头1关注根特异性元件
   头2关注叶特异性元件

⚠️ 注意：
   注意力权重≠因果关系
   只是模型认为"重要"的位置
   需要结合生物学知识解释
""")

plt.show()

print("\n" + "="*70)
print("✅ 注意力机制详解完成！")
print("="*70)



