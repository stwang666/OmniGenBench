# -*- coding: utf-8 -*-
# 多头注意力机制详解（针对hidden_size=480的情况）

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("🎯 多头注意力机制（Multi-Head Attention）详解")
print("="*70)

# ============= 1. 是的！Transformer都使用多头注意力 =============
print("\n📌 1. 你的模型使用多头注意力吗？")
print("-"*70)

print("""
答案：✅ 是的！

所有基于Transformer的模型都使用多头注意力（Multi-Head Attention）

OmniGenome系列模型：
  - OmniGenome-52M: ✅ 使用多头注意力
  - OmniGenome-v1.5: ✅ 使用多头注意力
  - 其他Transformer模型: ✅ 都使用

这是Transformer架构的核心组件之一！
""")

# ============= 2. 以hidden_size=480为例 =============
print("\n📌 2. 假设你的模型配置（hidden_size=480）")
print("-"*70)

# 模型配置
hidden_size = 480
num_heads = 8  # 常见配置
head_dim = hidden_size // num_heads

print(f"""
模型配置示例：
  - hidden_size (d_model): {hidden_size}
  - num_attention_heads: {num_heads}
  - 每个头的维度 (d_k): {head_dim} (= {hidden_size} / {num_heads})

验证：{num_heads} 个头 × {head_dim} 维/头 = {num_heads * head_dim} 维 ✅
""")

# ============= 3. 单头 vs 多头对比 =============
print("\n📌 3. 单头注意力 vs 多头注意力")
print("-"*70)

print("""
┌─────────────────────────────────────────────────────────┐
│  方案A: 单头注意力（假设的，实际不用）                    │
└─────────────────────────────────────────────────────────┘

输入: [batch, seq_len, 480]
  ↓
Q = X @ W_q  →  [batch, seq_len, 480]
K = X @ W_k  →  [batch, seq_len, 480]
V = X @ W_v  →  [batch, seq_len, 480]
  ↓
Attention = softmax(QK^T/√480) @ V
  ↓
输出: [batch, seq_len, 480]

❌ 问题：
  - 只有一种注意力模式
  - 只能学习一种特征关系
  - 表达能力受限


┌─────────────────────────────────────────────────────────┐
│  方案B: 多头注意力（实际使用，8个头）                     │
└─────────────────────────────────────────────────────────┘

输入: [batch, seq_len, 480]
  ↓
Split成8个头
  ↓
头1: [batch, seq_len, 60]  ← 关注局部motif
头2: [batch, seq_len, 60]  ← 关注远距离互作
头3: [batch, seq_len, 60]  ← 关注GC含量
头4: [batch, seq_len, 60]  ← 关注重复序列
头5: [batch, seq_len, 60]  ← 关注密码子
头6: [batch, seq_len, 60]  ← 关注剪接位点
头7: [batch, seq_len, 60]  ← 关注二级结构
头8: [batch, seq_len, 60]  ← 关注保守区域
  ↓
每个头独立计算attention
  ↓
Concat所有头: [batch, seq_len, 480]
  ↓
线性变换: [batch, seq_len, 480]
  ↓
输出: [batch, seq_len, 480]

✅ 优点：
  - 8种不同的注意力模式
  - 可以同时关注多种特征
  - 表达能力强大
""")

# ============= 4. 详细计算过程 =============
print("\n📌 4. 多头注意力的详细计算过程")
print("-"*70)

print(f"""
假设序列长度 seq_len = 100

步骤1: 线性投影
  输入: X [batch, 100, 480]
  
  Q = X @ W_q  →  [batch, 100, 480]
  K = X @ W_k  →  [batch, 100, 480]
  V = X @ W_v  →  [batch, 100, 480]

步骤2: 分割成多个头
  Q: [batch, 100, 480] → [batch, {num_heads}, 100, {head_dim}]
  K: [batch, 100, 480] → [batch, {num_heads}, 100, {head_dim}]
  V: [batch, 100, 480] → [batch, {num_heads}, 100, {head_dim}]
  
  reshape操作：
    [batch, 100, 480]
    → [batch, 100, {num_heads}, {head_dim}]
    → [batch, {num_heads}, 100, {head_dim}]

步骤3: 每个头独立计算注意力
  对于每个头 h=1,2,...,{num_heads}:
    
    3.1: 计算注意力分数
      scores_h = Q_h @ K_h^T / √{head_dim}
      [batch, 100, {head_dim}] @ [batch, {head_dim}, 100] = [batch, 100, 100]
    
    3.2: Softmax归一化
      attention_weights_h = softmax(scores_h)
      [batch, 100, 100]  # 每行和为1
    
    3.3: 加权求和
      output_h = attention_weights_h @ V_h
      [batch, 100, 100] @ [batch, 100, {head_dim}] = [batch, 100, {head_dim}]

步骤4: 拼接所有头的输出
  outputs = [output_1, output_2, ..., output_{num_heads}]
  concat(outputs) → [batch, 100, {num_heads} × {head_dim}]
                 = [batch, 100, {hidden_size}]

步骤5: 最终线性变换
  final_output = concat(outputs) @ W_o
  [batch, 100, 480] @ [480, 480] = [batch, 100, 480]
""")

# ============= 5. 可视化多头注意力 =============
print("\n📌 5. 可视化不同头的注意力模式")
print("-"*70)

# 模拟一个简短的DNA序列
sequence = "ATCGATCGTAGCTAGC"
seq_len = len(sequence)

# 为8个头生成不同的注意力模式
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

np.random.seed(42)

attention_patterns = [
    "局部注意力（近邻）",
    "全局注意力（均匀）",
    "稀疏注意力（远距离）",
    "自注意力（对角线）",
    "左侧注意力（前文）",
    "右侧注意力（后文）",
    "周期性注意力",
    "随机混合注意力"
]

for head_idx, (ax, pattern_name) in enumerate(zip(axes, attention_patterns)):
    # 为每个头生成不同模式的注意力矩阵
    attention = np.zeros((seq_len, seq_len))
    
    if head_idx == 0:  # 局部注意力
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                attention[i, j] = np.exp(-abs(i-j))
    
    elif head_idx == 1:  # 全局注意力
        attention = np.ones((seq_len, seq_len)) * 0.5
        attention += np.random.randn(seq_len, seq_len) * 0.1
    
    elif head_idx == 2:  # 稀疏注意力
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i-j) > 3:
                    attention[i, j] = np.random.rand() * 0.5
    
    elif head_idx == 3:  # 自注意力
        for i in range(seq_len):
            attention[i, i] = 1.0
            if i > 0:
                attention[i, i-1] = 0.3
            if i < seq_len-1:
                attention[i, i+1] = 0.3
    
    elif head_idx == 4:  # 左侧注意力
        for i in range(seq_len):
            for j in range(i+1):
                attention[i, j] = (1.0 - (i-j)/seq_len)
    
    elif head_idx == 5:  # 右侧注意力
        for i in range(seq_len):
            for j in range(i, seq_len):
                attention[i, j] = (1.0 - (j-i)/seq_len)
    
    elif head_idx == 6:  # 周期性
        for i in range(seq_len):
            for j in range(seq_len):
                if (i - j) % 3 == 0:
                    attention[i, j] = 0.8
    
    else:  # 随机混合
        attention = np.random.rand(seq_len, seq_len)
    
    # 归一化（每行和为1）
    attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-9)
    
    # 可视化
    sns.heatmap(attention, ax=ax, cmap='YlOrRd', cbar=True,
               xticklabels=list(sequence), yticklabels=list(sequence),
               vmin=0, vmax=attention.max())
    ax.set_title(f'头 {head_idx+1}: {pattern_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Key位置')
    ax.set_ylabel('Query位置')

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/multi_head_patterns.png', 
           dpi=300, bbox_inches='tight')
print("💾 多头注意力模式图已保存: multi_head_patterns.png\n")

# ============= 6. 多头的优点 =============
print("\n📌 6. 多头注意力的优点详解")
print("-"*70)

print(f"""
优点1: 🎯 捕捉多种特征关系
  
  在DNA序列分析中，不同的头可以关注：
    头1: TATA box（启动子元件）
    头2: 剪接位点（GT-AG）
    头3: CpG岛（甲基化相关）
    头4: 密码子使用偏好
    头5: 重复序列（LINE, SINE）
    头6: 保守域
    头7: 二级结构配对
    头8: GC含量变化
  
  → 一个头做不到这么多！需要多个头并行工作

优点2: 📊 提高模型容量
  
  参数量对比（假设hidden_size={hidden_size}）:
  
  单头：
    W_q, W_k, W_v: 3 × ({hidden_size} × {hidden_size}) = {3 * hidden_size * hidden_size:,}
    总参数: ~{3 * hidden_size * hidden_size:,}
  
  多头（{num_heads}个头）：
    每个头: 3 × ({hidden_size} × {head_dim}) = {3 * hidden_size * head_dim:,}
    {num_heads}个头: {num_heads} × {3 * hidden_size * head_dim:,} = {num_heads * 3 * hidden_size * head_dim:,}
    输出投影: {hidden_size} × {hidden_size} = {hidden_size * hidden_size:,}
    总参数: ~{num_heads * 3 * hidden_size * head_dim + hidden_size * hidden_size:,}
  
  → 参数量相近，但多头表达能力更强！

优点3: 🛡️ 防止过拟合
  
  分成多个小头（每个只有{head_dim}维）：
    - 每个头的参数更少
    - 强制学习不同的特征
    - 类似于集成学习的效果
  
  → 比一个大头更鲁棒

优点4: ⚡ 并行计算效率
  
  {num_heads}个头可以并行计算：
    - GPU并行处理多个头
    - 计算时间 ≈ 单头的时间（理想情况）
    - 但获得了{num_heads}倍的特征提取能力
  
  → 性能提升明显！

优点5: 🎨 可解释性
  
  可以查看不同头关注的内容：
    - 头1关注什么？→ 可视化
    - 头2关注什么？→ 可视化
    - 哪些头对任务最重要？→ 分析
  
  → 帮助理解模型的工作机制
""")

# ============= 7. 为什么要分成8个头？ =============
print("\n📌 7. 为什么通常是8个头？")
print("-"*70)

print("""
头数选择的考量：

太少（2-4个头）：
  ❌ 特征多样性不足
  ❌ 表达能力有限
  
适中（8-12个头）：✅ 常用配置
  ✅ 平衡性能和效率
  ✅ 足够的特征多样性
  ✅ 计算开销合理
  
太多（32+个头）：
  ❌ 计算开销大
  ❌ 每个头维度太小
  ❌ 收益递减

常见配置：
  - BERT-base: 12个头 (hidden=768, head_dim=64)
  - GPT-2: 12个头
  - OmniGenome: 8个头（可能）
  - 你的模型: 假设8个头 (hidden=480, head_dim=60)

经验规则：
  - head_dim通常是64或更小
  - 总维度hidden_size要能被头数整除
  - 头数是2的幂次（8, 16, 32）方便计算
""")

# ============= 8. 实际代码示例 =============
print("\n📌 8. PyTorch实现多头注意力")
print("-"*70)

code = f'''
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size={hidden_size}, num_heads={num_heads}):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # {head_dim}
        
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        # Q, K, V的投影矩阵
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 步骤1: 线性投影
        Q = self.W_q(x)  # [batch, seq_len, {hidden_size}]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 步骤2: 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, {num_heads}, seq_len, {head_dim}]
        
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)
        
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)
        
        # 步骤3: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, {num_heads}, seq_len, seq_len]
        
        # 步骤4: Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 步骤5: 加权求和
        attention_output = torch.matmul(attention_weights, V)
        # [batch, {num_heads}, seq_len, {head_dim}]
        
        # 步骤6: 拼接所有头
        attention_output = attention_output.transpose(1, 2)
        # [batch, seq_len, {num_heads}, {head_dim}]
        
        attention_output = attention_output.contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        # [batch, seq_len, {hidden_size}]
        
        # 步骤7: 输出投影
        output = self.W_o(attention_output)
        
        return output, attention_weights

# 使用示例
mha = MultiHeadAttention(hidden_size={hidden_size}, num_heads={num_heads})
x = torch.randn(2, 100, {hidden_size})  # [batch=2, seq_len=100, hidden={hidden_size}]
output, attn_weights = mha(x)

print(f"输入形状: {{x.shape}}")  # [2, 100, {hidden_size}]
print(f"输出形状: {{output.shape}}")  # [2, 100, {hidden_size}]
print(f"注意力权重形状: {{attn_weights.shape}}")  # [2, {num_heads}, 100, 100]
'''

print(code)

# ============= 9. 维度变化可视化 =============
print("\n📌 9. 多头注意力中的维度变化")
print("-"*70)

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# 绘制流程图
steps = [
    (0.5, 0.95, f"输入: [batch, seq_len, {hidden_size}]", "lightblue"),
    (0.5, 0.85, f"↓ 线性投影 (W_q, W_k, W_v)", "white"),
    (0.5, 0.75, f"Q, K, V: [batch, seq_len, {hidden_size}]", "lightgreen"),
    (0.5, 0.65, f"↓ Reshape + Transpose", "white"),
    (0.5, 0.55, f"Q, K, V: [batch, {num_heads}, seq_len, {head_dim}]", "lightyellow"),
    (0.5, 0.45, f"↓ 每个头计算 Attention", "white"),
    (0.5, 0.35, f"输出: [batch, {num_heads}, seq_len, {head_dim}]", "lightcoral"),
    (0.5, 0.25, f"↓ Transpose + Reshape", "white"),
    (0.5, 0.15, f"Concat: [batch, seq_len, {hidden_size}]", "lightpink"),
    (0.5, 0.05, f"↓ 输出投影 (W_o)\n最终: [batch, seq_len, {hidden_size}]", "lightblue"),
]

for x, y, text, color in steps:
    if "↓" in text:
        ax.text(x, y, text, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0))
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor=color, edgecolor='black', linewidth=2),
               fontweight='bold')

# 添加说明
ax.text(0.5, 0.02, 
       f'注意: hidden_size保持{hidden_size}不变，只是中间临时分成{num_heads}个头处理',
       ha='center', fontsize=10, style='italic',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/multi_head_dimensions.png', 
           dpi=300, bbox_inches='tight')
print("💾 维度变化图已保存: multi_head_dimensions.png\n")

plt.show()

print("\n" + "="*70)
print("✅ 多头注意力机制详解完成！")
print("="*70)

print("\n💡 关键要点总结：")
print("  1. ✅ 你的模型使用多头注意力（所有Transformer都用）")
print(f"  2. hidden_size={hidden_size} 通常分成{num_heads}个头")
print(f"  3. 每个头处理{head_dim}维的子空间")
print("  4. 多头让模型同时关注多种特征模式")
print("  5. 提高表达能力的同时保持计算效率")


