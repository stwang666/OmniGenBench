# -*- coding: utf-8 -*-
# 正则化和Dropout详解

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("="*70)
print("🛡️ 正则化（Regularization）和 Dropout 详解")
print("="*70)

# ============= 1. 什么是正则化？ =============
print("\n📌 1. 什么是正则化（Regularization）？")
print("-"*70)

print("""
正则化的核心目的：防止过拟合

过拟合问题：
  训练集：99% ✅
  测试集：40% ❌
  
  模型把训练数据"记住"了，而不是学到规律

正则化的思想：
  给模型添加约束，让它不要太"复杂"
  
类比：
  学生做题：
    无约束：死记硬背每道题的答案（过拟合）
    有约束：理解原理，学会举一反三（泛化）
""")

# ============= 2. L2正则化（Weight Decay） =============
print("\n📌 2. L2正则化（Weight Decay）")
print("-"*70)

print("""
原理：惩罚大的权重值

损失函数变化：
  原始损失：
    Loss = CrossEntropyLoss(predictions, labels)
  
  添加L2正则化：
    Loss = CrossEntropyLoss(predictions, labels) + λ * Σ(w²)
           └─────── 任务损失 ──────┘   └── 正则化项 ──┘
  
  其中：
    - λ (lambda): 正则化强度（如0.01）
    - w: 模型的权重
    - Σ(w²): 所有权重的平方和

效果：
  ✅ 让权重变小
  ✅ 模型更"平滑"
  ✅ 降低过拟合风险
  
PyTorch实现：
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=1e-5,
      weight_decay=0.01  # 🔑 这就是L2正则化！
  )
""")

# 可视化L2正则化效果
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 生成简单数据
np.random.seed(42)
x = np.linspace(-3, 3, 100)
y_true = 2 * x + 1  # 真实关系

# 添加噪声
y_noisy = y_true + np.random.randn(100) * 2

# 三种模型：无正则化、轻度正则化、重度正则化
for ax, decay, title in zip(axes, [0, 0.01, 0.1], 
                            ['无正则化（过拟合）', 'L2=0.01（适度）', 'L2=0.1（欠拟合）']):
    # 模拟不同正则化强度下的拟合
    if decay == 0:
        # 过拟合：使用高次多项式
        coeffs = np.polyfit(x, y_noisy, 9)
        y_pred = np.polyval(coeffs, x)
        color = 'red'
    elif decay == 0.01:
        # 适度：简单线性
        coeffs = np.polyfit(x, y_noisy, 1)
        y_pred = np.polyval(coeffs, x)
        color = 'green'
    else:
        # 过度正则化：接近常数
        y_pred = np.ones_like(x) * np.mean(y_noisy)
        color = 'orange'
    
    ax.scatter(x, y_noisy, alpha=0.3, s=20, label='数据点')
    ax.plot(x, y_true, 'b--', linewidth=2, label='真实关系', alpha=0.7)
    ax.plot(x, y_pred, color=color, linewidth=3, label='模型拟合')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/l2_regularization.png', 
           dpi=300, bbox_inches='tight')
print("💾 L2正则化效果图已保存: l2_regularization.png\n")

# ============= 3. Dropout =============
print("\n📌 3. Dropout")
print("-"*70)

print("""
原理：训练时随机"关闭"一些神经元

工作机制：
  训练阶段（Training）：
    1. 对每个神经元，以概率p随机将其输出设为0
    2. 其余神经元的输出 × (1/(1-p)) 来保持期望不变
    3. 每个batch都重新随机
  
  测试阶段（Inference）：
    - 所有神经元都工作
    - 不进行dropout操作

示例（Dropout=0.4）：
  原始神经元输出:
    [0.5, 0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.1, 0.5]
  
  训练时应用Dropout（随机关闭40%）:
    [0.0, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 0.7, 0.2, 0.0]
    └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘
     ❌   ✅   ❌   ✅   ✅   ❌   ❌   ✅   ✅   ❌
  
  测试时（无Dropout）:
    [0.5, 0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.1, 0.5]
    └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘  └─┘
     ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅

为什么有效？
  1. 防止神经元之间的"合谋"（co-adaptation）
  2. 类似于训练多个模型的集成
  3. 强制网络学习更鲁棒的特征
  
类比：
  足球队训练：
    无Dropout: 固定11人组合训练
      → 队员之间配合好，但依赖性强
      → 缺一个人就玩不转
    
    有Dropout: 每次训练随机缺人
      → 每个人学会独立能力
      → 任何组合都能发挥
""")

# 可视化Dropout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 创建一个简单的神经网络层
layer_size = 100
dropout_rate = 0.4

# 原始激活
original_activation = np.random.randn(layer_size)

# 绘制6次不同的dropout mask
for idx, ax in enumerate(axes.flat):
    # 生成dropout mask
    mask = np.random.binomial(1, 1-dropout_rate, layer_size)
    dropped_activation = original_activation * mask / (1 - dropout_rate)
    
    # 可视化
    ax.bar(range(layer_size), original_activation, alpha=0.3, color='blue', label='原始')
    ax.bar(range(layer_size), dropped_activation, alpha=0.7, color='red', label='Dropout后')
    ax.set_title(f'Dropout示例 {idx+1}\n（红色=0表示被关闭）', fontsize=10, fontweight='bold')
    ax.set_xlabel('神经元索引')
    ax.set_ylabel('激活值')
    ax.set_ylim(-3, 3)
    if idx == 0:
        ax.legend()
    
    # 统计被dropout的数量
    n_dropped = np.sum(mask == 0)
    ax.text(0.5, 0.95, f'关闭: {n_dropped}/{layer_size} ({n_dropped/layer_size*100:.1f}%)',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
           fontsize=9)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/dropout_visualization.png', 
           dpi=300, bbox_inches='tight')
print("💾 Dropout可视化已保存: dropout_visualization.png\n")

# ============= 4. PyTorch代码示例 =============
print("\n📌 4. PyTorch代码示例")
print("-"*70)

print("""
示例1: 添加Dropout层

class ModelWithoutDropout(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 27)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # ❌ 容易过拟合
        return x

class ModelWithDropout(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(768, 384)
        self.dropout = nn.Dropout(0.4)  # ✅ 添加Dropout
        self.fc2 = nn.Linear(384, 27)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # ✅ 训练时随机dropout
        x = self.fc2(x)
        return x

示例2: 使用Weight Decay

# 无正则化
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 有L2正则化（Weight Decay）
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-5,
    weight_decay=0.01  # ✅ λ=0.01
)

示例3: 训练/测试模式切换

# 训练模式（Dropout生效）
model.train()
for batch in train_loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 测试模式（Dropout关闭）
model.eval()
with torch.no_grad():
    for batch in test_loader:
        predictions = model(batch)
""")

# ============= 5. 实际训练对比 =============
print("\n📌 5. 有无正则化的对比")
print("-"*70)

# 模拟训练曲线
epochs = np.arange(1, 51)

# 无正则化：快速过拟合
train_loss_no_reg = 1.0 * np.exp(-epochs / 5) + 0.05
test_loss_no_reg = 0.8 * np.exp(-epochs / 10) + 0.4 + 0.1 * epochs / 50

# 有正则化：稳定收敛
train_loss_reg = 1.0 * np.exp(-epochs / 8) + 0.15
test_loss_reg = 0.8 * np.exp(-epochs / 8) + 0.2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1 = axes[0]
ax1.plot(epochs, train_loss_no_reg, 'r-', linewidth=2, label='训练集（无正则化）')
ax1.plot(epochs, test_loss_no_reg, 'r--', linewidth=2, label='测试集（无正则化）')
ax1.plot(epochs, train_loss_reg, 'g-', linewidth=2, label='训练集（有正则化）')
ax1.plot(epochs, test_loss_reg, 'g--', linewidth=2, label='测试集（有正则化）')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('训练曲线对比\n（绿色=有正则化，红色=无正则化）', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# 标注过拟合区域
ax1.axvspan(15, 50, alpha=0.2, color='red')
ax1.text(32, 0.9, '无正则化：\n严重过拟合', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax1.text(32, 0.3, '有正则化：\n泛化良好', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

# 泛化gap
ax2 = axes[1]
gap_no_reg = test_loss_no_reg - train_loss_no_reg
gap_reg = test_loss_reg - train_loss_reg

ax2.plot(epochs, gap_no_reg, 'r-', linewidth=3, label='无正则化')
ax2.plot(epochs, gap_reg, 'g-', linewidth=3, label='有正则化')
ax2.axhline(y=0.1, color='orange', linestyle='--', label='可接受阈值')
ax2.fill_between(epochs, 0, gap_no_reg, alpha=0.3, color='red')
ax2.fill_between(epochs, 0, gap_reg, alpha=0.3, color='green')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('泛化Gap（测试Loss - 训练Loss）', fontsize=11)
ax2.set_title('泛化能力对比\n（Gap越小越好）', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/sw1136/OmniGenBench/examples/dingling_te/regularization_comparison.png', 
           dpi=300, bbox_inches='tight')
print("💾 正则化对比图已保存: regularization_comparison.png\n")

# ============= 6. 其他正则化技术 =============
print("\n📌 6. 其他正则化技术")
print("-"*70)

print("""
1️⃣ L1正则化（Lasso）
   Loss = Task_Loss + λ * Σ|w|
   
   特点：
   - 倾向于产生稀疏权重（很多权重=0）
   - 可用于特征选择
   - PyTorch中需要手动实现

2️⃣ Label Smoothing
   原始标签: [0, 0, 1]  # one-hot
   平滑后:    [0.05, 0.05, 0.9]  # 更软的标签
   
   效果：
   - 防止模型过度自信
   - 提高泛化能力
   
   代码：
   loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

3️⃣ Early Stopping
   监控验证集性能，如果不再提升就停止训练
   
   实现：
   best_val_loss = float('inf')
   patience = 5
   counter = 0
   
   for epoch in range(max_epochs):
       val_loss = validate(model)
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           save_model(model)
           counter = 0
       else:
           counter += 1
           if counter >= patience:
               print("Early stopping!")
               break

4️⃣ Batch Normalization
   标准化每层的输入
   
   class ModelWithBN(nn.Module):
       def __init__(self):
           self.fc1 = nn.Linear(768, 384)
           self.bn = nn.BatchNorm1d(384)  # ✅ 批归一化
           self.fc2 = nn.Linear(384, 27)
       
       def forward(self, x):
           x = self.fc1(x)
           x = self.bn(x)  # 归一化
           x = F.relu(x)
           x = self.fc2(x)
           return x

5️⃣ Data Augmentation（数据增强）
   对DNA序列：
   - 反向互补
   - 随机突变（慎用）
   - 滑动窗口
   
   效果：
   - 增加数据多样性
   - 提高模型鲁棒性

6️⃣ 冻结层（Layer Freezing）
   只训练部分层，冻结其他层
   
   # 冻结前6层
   for i, layer in enumerate(model.encoder.layer):
       if i < 6:
           for param in layer.parameters():
               param.requires_grad = False
""")

# ============= 7. 如何选择正则化强度？ =============
print("\n📌 7. 如何选择正则化强度？")
print("-"*70)

print("""
Weight Decay（λ）的选择：

太小（λ < 0.0001）:
  ❌ 正则化效果不明显
  ❌ 仍然容易过拟合

适中（λ = 0.001 - 0.1）:
  ✅ 平衡拟合和泛化
  ✅ 大多数情况的推荐值

太大（λ > 0.5）:
  ❌ 欠拟合
  ❌ 模型学不到东西

常用值：
  - 图像分类: 0.0001 - 0.001
  - NLP: 0.01 - 0.1
  - 基因组学: 0.01（你的任务推荐值）

Dropout Rate的选择：

太小（< 0.1）:
  ❌ 效果不明显

适中（0.3 - 0.5）:
  ✅ 推荐范围
  ✅ 0.4是常用值（你的模型使用）

太大（> 0.7）:
  ❌ 扔掉太多信息
  ❌ 训练不稳定

经验法则：
  - 全连接层: 0.5
  - RNN: 0.2 - 0.5
  - Transformer: 0.1 - 0.3
  - 任务头: 0.3 - 0.5（你的任务）

调参建议：
  1. 从小开始（λ=0.01, dropout=0.1）
  2. 观察训练/验证曲线
  3. 如果过拟合严重，增大正则化
  4. 如果欠拟合，减小正则化
  5. 使用网格搜索或贝叶斯优化
""")

plt.show()

print("\n" + "="*70)
print("✅ 正则化和Dropout详解完成！")
print("="*70)

print("\n💡 关键要点：")
print("  1. 正则化 = 防止过拟合的技术")
print("  2. Weight Decay = 惩罚大权重")
print("  3. Dropout = 随机关闭神经元")
print("  4. 两者可以同时使用")
print("  5. 合理选择强度很重要")



