# -*- coding: utf-8 -*-
# file: visualize_performance.py
# 可视化训练和测试性能，诊断过拟合

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from omnigenbench import ModelHub, OmniTokenizer
from triclass_te import TriClassTEDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

model_name_or_path = "yangheng/OmniGenome-52M"
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

print("="*70)
print("📊 模型性能可视化分析")
print("="*70)

# 加载数据
datasets = TriClassTEDataset.from_hub(
    dataset_name_or_path="examples/dingling_te/",
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

# 加载模型
model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
print(f"\n🔄 加载模型: {model_path}")
model = ModelHub.load(model_path)

tissue_names = [
    'root', 'seedling', 'leaf', 'FMI', 'FOD',
    'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
]
label_names = ['Low', 'Medium', 'High']


def evaluate_dataset(model, dataset, dataset_name):
    """评估数据集并返回准确率、预测和真实标签"""
    print(f"\n📈 评估 {dataset_name} 集...")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for item in dataset.examples:
            sequence = item['sequence']
            outputs = model.inference(sequence)
            
            predictions = outputs['predictions'].cpu().numpy()
            confidence = outputs['confidence'].cpu().numpy()
            
            # 获取真实标签
            true_labels = []
            for tissue in tissue_names:
                col_name = f"{tissue}_TE_label"
                label_str = str(item.get(col_name, 'nan'))
                if label_str in ['Low', 'Medium', 'High']:
                    label_idx = label_names.index(label_str)
                    true_labels.append(label_idx)
                else:
                    true_labels.append(-100)  # 忽略
            
            all_predictions.append(predictions)
            all_labels.append(true_labels)
            all_confidences.append(confidence)
    
    all_predictions = np.array(all_predictions)  # [N, 9]
    all_labels = np.array(all_labels)  # [N, 9]
    all_confidences = np.array(all_confidences)  # [N, 9]
    
    # 计算准确率（忽略-100）
    mask = all_labels != -100
    correct = (all_predictions == all_labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    print(f"✅ {dataset_name} 准确率: {accuracy:.4f} ({correct.sum()}/{mask.sum()})")
    
    return accuracy, all_predictions, all_labels, all_confidences, mask


# 评估各个数据集
train_acc, train_preds, train_labels, train_conf, train_mask = evaluate_dataset(
    model, datasets['train'], 'Train'
)
val_acc, val_preds, val_labels, val_conf, val_mask = evaluate_dataset(
    model, datasets['valid'], 'Valid'
)
test_acc, test_preds, test_labels, test_conf, test_mask = evaluate_dataset(
    model, datasets['test'], 'Test'
)

# 创建可视化
fig = plt.figure(figsize=(20, 12))

# 1. 准确率对比柱状图
ax1 = plt.subplot(2, 3, 1)
datasets_names = ['Train', 'Valid', 'Test']
accuracies = [train_acc, val_acc, test_acc]
colors = ['#2ECC71', '#F39C12', '#E74C3C']

bars = ax1.bar(datasets_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('各数据集准确率对比\n(显示过拟合程度)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.333, color='red', linestyle='--', alpha=0.5, label='随机猜测 (33.3%)')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# 在柱子上标注数值
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2%}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# 添加过拟合程度标注
overfitting = train_acc - test_acc
ax1.text(0.5, 0.95, f'过拟合程度: {overfitting:.2%}',
        transform=ax1.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        fontsize=12, fontweight='bold')

# 2. 每个组织的准确率
ax2 = plt.subplot(2, 3, 2)
tissue_accs = {'train': [], 'valid': [], 'test': []}

for tissue_idx in range(9):
    for split_name, preds, labels, mask in [
        ('train', train_preds, train_labels, train_mask),
        ('valid', val_preds, val_labels, val_mask),
        ('test', test_preds, test_labels, test_mask)
    ]:
        tissue_mask = mask[:, tissue_idx]
        tissue_correct = ((preds[:, tissue_idx] == labels[:, tissue_idx]) & tissue_mask).sum()
        tissue_total = tissue_mask.sum()
        tissue_acc = tissue_correct / tissue_total if tissue_total > 0 else 0
        tissue_accs[split_name].append(tissue_acc)

x = np.arange(len(tissue_names))
width = 0.25

ax2.bar(x - width, tissue_accs['train'], width, label='Train', color='#2ECC71', alpha=0.7)
ax2.bar(x, tissue_accs['valid'], width, label='Valid', color='#F39C12', alpha=0.7)
ax2.bar(x + width, tissue_accs['test'], width, label='Test', color='#E74C3C', alpha=0.7)

ax2.set_ylabel('准确率', fontsize=10)
ax2.set_title('各组织的准确率对比', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(tissue_names, rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.0)

# 3. 置信度分布对比
ax3 = plt.subplot(2, 3, 3)

train_conf_flat = train_conf[train_mask].flatten()
test_conf_flat = test_conf[test_mask].flatten()

ax3.hist(train_conf_flat, bins=50, alpha=0.5, label=f'Train (mean={train_conf_flat.mean():.3f})', 
        color='green', density=True)
ax3.hist(test_conf_flat, bins=50, alpha=0.5, label=f'Test (mean={test_conf_flat.mean():.3f})', 
        color='red', density=True)
ax3.set_xlabel('预测置信度', fontsize=10)
ax3.set_ylabel('密度', fontsize=10)
ax3.set_title('置信度分布对比\n(过拟合模型在训练集上更自信)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. 测试集混淆矩阵
ax4 = plt.subplot(2, 3, 4)

test_preds_flat = test_preds[test_mask].flatten()
test_labels_flat = test_labels[test_mask].flatten()
cm = confusion_matrix(test_labels_flat, test_preds_flat, labels=[0, 1, 2])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
           xticklabels=label_names, yticklabels=label_names)
ax4.set_ylabel('真实标签', fontsize=10)
ax4.set_xlabel('预测标签', fontsize=10)
ax4.set_title('测试集混淆矩阵', fontsize=12, fontweight='bold')

# 5. 训练集混淆矩阵
ax5 = plt.subplot(2, 3, 5)

train_preds_flat = train_preds[train_mask].flatten()
train_labels_flat = train_labels[train_mask].flatten()
cm_train = confusion_matrix(train_labels_flat, train_preds_flat, labels=[0, 1, 2])

sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=ax5,
           xticklabels=label_names, yticklabels=label_names)
ax5.set_ylabel('真实标签', fontsize=10)
ax5.set_xlabel('预测标签', fontsize=10)
ax5.set_title('训练集混淆矩阵', fontsize=12, fontweight='bold')

# 6. 类别分布对比
ax6 = plt.subplot(2, 3, 6)

train_label_dist = Counter(train_labels_flat)
test_label_dist = Counter(test_labels_flat)

x_labels = label_names
train_counts = [train_label_dist[i] for i in range(3)]
test_counts = [test_label_dist[i] for i in range(3)]

x_pos = np.arange(len(x_labels))
ax6.bar(x_pos - 0.2, train_counts, 0.4, label='Train', color='green', alpha=0.7)
ax6.bar(x_pos + 0.2, test_counts, 0.4, label='Test', color='red', alpha=0.7)

ax6.set_ylabel('样本数量', fontsize=10)
ax6.set_title('类别分布对比\n(检查训练测试分布是否一致)', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(x_labels)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = '/home/sw1136/OmniGenBench/examples/dingling_te/overfitting_analysis.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n💾 可视化已保存: {save_path}")
plt.show()

# 打印详细分析报告
print("\n" + "="*70)
print("📋 详细分析报告")
print("="*70)

print(f"\n1️⃣ 整体性能:")
print(f"   训练集准确率: {train_acc:.4f}")
print(f"   验证集准确率: {val_acc:.4f}")
print(f"   测试集准确率: {test_acc:.4f}")
print(f"   过拟合程度: {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.30:
    print("   ❌ 严重过拟合！")
elif train_acc - test_acc > 0.15:
    print("   ⚠️  中度过拟合")
elif train_acc - test_acc > 0.05:
    print("   ⚠️  轻微过拟合")
else:
    print("   ✅ 泛化良好")

print(f"\n2️⃣ 置信度分析:")
print(f"   训练集平均置信度: {train_conf_flat.mean():.4f}")
print(f"   测试集平均置信度: {test_conf_flat.mean():.4f}")

if train_conf_flat.mean() - test_conf_flat.mean() > 0.15:
    print("   ❌ 模型在训练集上过度自信，典型的过拟合表现！")

print(f"\n3️⃣ 测试集分类报告:")
print(classification_report(test_labels_flat, test_preds_flat, 
                          target_names=label_names, digits=3))

print("\n4️⃣ 建议:")
if train_acc - test_acc > 0.20:
    print("   🔥 紧急建议:")
    print("      1. 运行 diagnose_overfitting.py 检查数据质量")
    print("      2. 使用 triclass_te_improved.py 重新训练")
    print("      3. 检查是否有数据泄漏")
elif train_acc - test_acc > 0.10:
    print("   📌 改进建议:")
    print("      1. 增加正则化（Dropout, Weight Decay）")
    print("      2. 减少训练epoch或添加early stopping")
    print("      3. 使用数据增强")
else:
    print("   ✅ 模型泛化能力良好，可以继续优化超参数提升整体性能")

print("\n" + "="*70)



