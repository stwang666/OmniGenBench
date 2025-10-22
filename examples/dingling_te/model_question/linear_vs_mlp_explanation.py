# -*- coding: utf-8 -*-
# 线性分类器 vs MLP分类器对比示例

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("🔢 线性分类器 vs MLP分类器详解")
print("="*70)

# ============= 1. 线性分类器（单层） =============
print("\n📌 1. 线性分类器（Linear Classifier）")
print("-"*70)

hidden_size = 768  # 假设transformer输出768维
num_classes = 27   # 9个组织 × 3个类别 = 27

# 单层线性分类器
linear_classifier = nn.Linear(hidden_size, num_classes)

print(f"输入维度: {hidden_size}")
print(f"输出维度: {num_classes}")
print(f"参数数量: {hidden_size * num_classes + num_classes:,}")
print(f"  = 权重矩阵 W: {hidden_size} × {num_classes} = {hidden_size * num_classes:,}")
print(f"  + 偏置向量 b: {num_classes}")

# 示例前向传播
batch_size = 2
input_features = torch.randn(batch_size, hidden_size)
output = linear_classifier(input_features)

print(f"\n前向传播示例:")
print(f"  输入: {input_features.shape}  # [batch_size, hidden_size]")
print(f"  输出: {output.shape}  # [batch_size, num_classes]")

print(f"\n数学公式:")
print(f"  output = input @ W^T + b")
print(f"  其中:")
print(f"    input: [{batch_size}, {hidden_size}]")
print(f"    W: [{num_classes}, {hidden_size}]")
print(f"    b: [{num_classes}]")
print(f"    output: [{batch_size}, {num_classes}]")

# ============= 2. MLP分类器（多层） =============
print("\n\n📌 2. MLP分类器（Multi-Layer Perceptron）")
print("-"*70)

# 多层MLP分类器
mlp_classifier = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),  # 第1层: 768 → 384
    nn.ReLU(),                                  # 激活函数
    nn.Dropout(0.4),                           # Dropout防止过拟合
    nn.Linear(hidden_size // 2, num_classes)   # 第2层: 384 → 27
)

# 计算参数量
mlp_params = (hidden_size * (hidden_size // 2) + hidden_size // 2) + \
             ((hidden_size // 2) * num_classes + num_classes)

print(f"层结构:")
print(f"  第1层: Linear({hidden_size} → {hidden_size // 2})")
print(f"  第2层: ReLU()")
print(f"  第3层: Dropout(0.4)")
print(f"  第4层: Linear({hidden_size // 2} → {num_classes})")

print(f"\n参数数量: {mlp_params:,}")
print(f"  = 第1层: {hidden_size * (hidden_size // 2) + hidden_size // 2:,}")
print(f"  + 第2层: {(hidden_size // 2) * num_classes + num_classes:,}")

# 示例前向传播
output_mlp = mlp_classifier(input_features)
print(f"\n前向传播示例:")
print(f"  输入: {input_features.shape}  # [batch_size, hidden_size]")
print(f"  → 第1层: [{batch_size}, {hidden_size // 2}]")
print(f"  → ReLU激活")
print(f"  → Dropout")
print(f"  → 第2层: [{batch_size}, {num_classes}]")
print(f"  输出: {output_mlp.shape}")

# ============= 3. 对比分析 =============
print("\n\n📌 3. 线性 vs MLP 对比")
print("-"*70)

linear_params = hidden_size * num_classes + num_classes
comparison = [
    ["特性", "线性分类器", "MLP分类器"],
    ["层数", "1层", "2+层"],
    ["参数量", f"{linear_params:,}", f"{mlp_params:,}"],
    ["非线性", "❌ 无", "✅ 有（ReLU）"],
    ["表达能力", "较弱（线性变换）", "较强（可拟合复杂函数）"],
    ["过拟合风险", "较低", "较高（但可用Dropout控制）"],
    ["训练速度", "快", "稍慢"],
    ["适用场景", "简单任务、特征已好", "复杂任务、需要特征变换"],
]

for row in comparison:
    print(f"{row[0]:15s} | {row[1]:25s} | {row[2]:35s}")

# ============= 4. 可视化决策边界 =============
print("\n\n📌 4. 决策边界可视化（2D示例）")
print("-"*70)

# 为了可视化，我们用2D输入、2类输出的简化版本
np.random.seed(42)

# 生成2D数据
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 简单线性可分

# 线性分类器（2D → 2类）
linear_2d = nn.Linear(2, 2)
linear_2d.weight.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
linear_2d.bias.data = torch.tensor([0.0, 0.0])

# MLP分类器（2D → 2类）
mlp_2d = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# 创建网格
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.FloatTensor(grid)

# 预测
with torch.no_grad():
    Z_linear = linear_2d(grid_tensor).argmax(dim=1).numpy().reshape(xx.shape)
    Z_mlp = mlp_2d(grid_tensor).argmax(dim=1).numpy().reshape(xx.shape)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 线性分类器决策边界
axes[0].contourf(xx, yy, Z_linear, alpha=0.3, levels=1, cmap='RdYlBu')
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', label='类别0', edgecolors='k', alpha=0.7)
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', label='类别1', edgecolors='k', alpha=0.7)
axes[0].set_title('线性分类器\n决策边界是直线', fontsize=12, fontweight='bold')
axes[0].set_xlabel('特征1')
axes[0].set_ylabel('特征2')
axes[0].legend()
axes[0].grid(alpha=0.3)

# MLP分类器决策边界
axes[1].contourf(xx, yy, Z_mlp, alpha=0.3, levels=1, cmap='RdYlBu')
axes[1].scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', label='类别0', edgecolors='k', alpha=0.7)
axes[1].scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', label='类别1', edgecolors='k', alpha=0.7)
axes[1].set_title('MLP分类器\n决策边界可以是曲线', fontsize=12, fontweight='bold')
axes[1].set_xlabel('特征1')
axes[1].set_ylabel('特征2')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/linear_vs_mlp.png', dpi=300, bbox_inches='tight')
print("💾 决策边界图已保存: linear_vs_mlp.png")
plt.show()

# ============= 5. 为什么你的模型是27维输出？ =============
print("\n\n📌 5. 为什么输出是27维？")
print("-"*70)

num_tissues = 9
num_class_per_tissue = 3
total_output = num_tissues * num_class_per_tissue

print(f"任务: 预测9个组织的表达水平（Low/Medium/High）")
print(f"\n方法1: Flatten输出（原始代码使用）")
print(f"  输出维度: {num_tissues} × {num_class_per_tissue} = {total_output}")
print(f"  输出形状: [batch, {total_output}]")
print(f"  然后reshape: [batch, {total_output}] → [batch, {num_tissues}, {num_class_per_tissue}]")
print(f"\n示例:")
print(f"  Linear输出: [2, 27]")
print(f"  Reshape后: [2, 9, 3]")
print(f"    ├─ batch维度: 2个样本")
print(f"    ├─ tissue维度: 9个组织")
print(f"    └─ class维度: 3个类别（Low/Medium/High）")

print(f"\n对于每个组织，应用Softmax得到3个类别的概率:")
print(f"  logits[i, j, :] → softmax → [P(Low), P(Medium), P(High)]")

# ============= 6. 代码实现对比 =============
print("\n\n📌 6. 实际代码对比")
print("-"*70)

print("\n原始代码（单层Linear）:")
print("""
class OriginalModel:
    def __init__(self, hidden_size=768):
        self.classifier = nn.Linear(hidden_size, 27)  # 单层
    
    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)  # [batch, 27]
        logits = logits.view(batch_size, 9, 3)   # reshape
        return logits
""")

print("\n改进代码（多层MLP）:")
print("""
class ImprovedModel:
    def __init__(self, hidden_size=768):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 768 → 384
            nn.ReLU(),                                 # 非线性激活
            nn.Dropout(0.4),                          # 正则化
            nn.Linear(hidden_size // 2, 27)           # 384 → 27
        )
    
    def forward(self, pooled_output):
        logits = self.classifier(pooled_output)  # [batch, 27]
        logits = logits.view(batch_size, 9, 3)   # reshape
        return logits
""")

print("\n优势:")
print("  ✅ MLP有更强的表达能力（可以学习非线性关系）")
print("  ✅ 中间层可以学习更好的特征表示")
print("  ✅ Dropout帮助防止过拟合")

print("\n" + "="*70)
print("✅ 详解完成！")



