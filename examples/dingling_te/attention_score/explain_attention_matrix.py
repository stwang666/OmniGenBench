"""
解释Self-Attention权重矩阵的含义
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print(" 📚 Self-Attention权重矩阵详解")
print("="*80)

# 模拟一个简单的attention矩阵
seq_len = 10
np.random.seed(42)

# 创建一个示例attention矩阵
# 让它更容易理解：第i个token主要关注位置 i-1, i, i+1
attention_matrix = np.zeros((seq_len, seq_len))
for i in range(seq_len):
    # 每个位置主要关注自己和相邻位置
    if i > 0:
        attention_matrix[i, i-1] = 0.2
    attention_matrix[i, i] = 0.6
    if i < seq_len - 1:
        attention_matrix[i, i+1] = 0.2

print("\n1️⃣  Self-Attention计算过程：")
print("""
步骤1: 计算 Query, Key, Value
   Query (Q) = Input × W_Q  [seq_len, d_model]
   Key   (K) = Input × W_K  [seq_len, d_model]
   Value (V) = Input × W_V  [seq_len, d_model]

步骤2: 计算注意力分数（这就是我们可视化的！）
   Scores = Q × K^T  [seq_len, seq_len]
   
步骤3: 归一化（softmax）
   Attention_weights = softmax(Scores / √d_k)  [seq_len, seq_len]
   
步骤4: 加权求和
   Output = Attention_weights × V  [seq_len, d_model]
""")

print("\n2️⃣  注意力矩阵的含义：")
print("""
注意力矩阵形状: [Query_positions, Key_positions]
              = [序列长度, 序列长度]

矩阵元素 [i, j] 的含义:
   - i (行): 第i个位置作为 Query（"我在问问题"）
   - j (列): 第j个位置作为 Key（"我被查询"）
   - 值  : Query_i 对 Key_j 的注意力权重
   
每一行的和 = 1.0 (因为经过了softmax归一化)
""")

print("\n3️⃣  示例注意力矩阵：")
print("\n每个token主要关注自己和相邻位置：")
print(f"矩阵形状: {attention_matrix.shape}")
print(f"\n前5行（Query位置0-4对所有Key位置的注意力）：")
print(attention_matrix[:5, :])

print("\n4️⃣  具体含义解释：")
for i in range(min(3, seq_len)):
    print(f"\n位置 {i} (作为Query) 的注意力分布:")
    for j in range(seq_len):
        if attention_matrix[i, j] > 0:
            print(f"   → 关注位置 {j} (Key): {attention_matrix[i, j]:.2f}")

print("\n5️⃣  可视化：")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: 标准热图
im1 = axes[0].imshow(attention_matrix, cmap='Blues', aspect='auto')
axes[0].set_title("Standard Attention Heatmap", fontsize=14)
axes[0].set_xlabel("Key Positions (j)")
axes[0].set_ylabel("Query Positions (i)")
plt.colorbar(im1, ax=axes[0])

# 添加网格
for i in range(seq_len):
    for j in range(seq_len):
        if attention_matrix[i, j] > 0.1:
            axes[0].text(j, i, f'{attention_matrix[i, j]:.1f}', 
                        ha="center", va="center", color="red", fontsize=8)

# 子图2: 带标注的示例
axes[1].imshow(attention_matrix, cmap='Blues', aspect='auto')
axes[1].set_title("With Annotations", fontsize=14)
axes[1].set_xlabel("Key Positions (j)")
axes[1].set_ylabel("Query Positions (i)")

# 标注几个关键点
example_i, example_j = 5, 5
axes[1].plot(example_j, example_i, 'ro', markersize=15, markerfacecolor='none', markeredgewidth=3)
axes[1].annotate(f'Query_{example_i} → Key_{example_j}\nWeight={attention_matrix[example_i, example_j]:.2f}',
                xy=(example_j, example_i), xytext=(example_j+2, example_i-2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red')

# 子图3: 一行的注意力分布（柱状图）
query_pos = 5
axes[2].bar(range(seq_len), attention_matrix[query_pos, :], color='blue', alpha=0.7)
axes[2].set_title(f"Attention Distribution for Query Position {query_pos}", fontsize=14)
axes[2].set_xlabel("Key Positions")
axes[2].set_ylabel("Attention Weight")
axes[2].set_ylim([0, 1.0])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("EXPLAIN_attention_matrix.png", dpi=300, bbox_inches='tight')
print("   ✅ 可视化已保存: EXPLAIN_attention_matrix.png")

print("\n" + "="*80)
print(" 🎯 总结")
print("="*80)
print("""
✅ 是的，self-attention weight 就是 Query 对应的 Key 的权重！

注意力矩阵 [i, j]:
   - 表示: 位置i (Query) 对位置j (Key) 的注意力
   - 含义: "位置i 有多关注位置j 的信息"
   - 对角线: 表示每个位置对自己的关注程度
   - 非对角线: 表示位置间的相互关注

在您的微调模型中：
   - 对角线较强 → 每个位置主要关注自己（局部特征）
   - 这是正常的！说明您的基因组任务主要依赖局部信息
""")

print("\n📖 补充：Multi-Head Attention")
print("""
您的模型有 24 个attention heads，每个head学习不同的注意力模式：
   - Head 0 可能关注: 局部序列模式
   - Head 1 可能关注: GC含量
   - Head 2 可能关注: 重复序列
   - ... 等等

我们画的平均注意力 = 所有24个heads的平均
""")

