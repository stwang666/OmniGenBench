# -*- coding: utf-8 -*-
# file: train_with_kfold.py
# K-Fold交叉验证训练，更准确地评估泛化能力

import torch
import numpy as np
from sklearn.model_selection import KFold
from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    OmniTokenizer,
)
from triclass_te_improved import TriClassTEDataset, ImprovedOmniModelForTriClassTE

model_name_or_path = "yangheng/OmniGenome-52M"
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

print("="*70)
print("🔄 K-Fold 交叉验证训练")
print("="*70)

# 加载数据
datasets = TriClassTEDataset.from_hub(
    dataset_name_or_path="examples/dingling_te/",
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

# 合并训练集和验证集进行K-Fold
all_train_data = datasets['train'].examples + datasets['valid'].examples
test_data = datasets['test'].examples

print(f"\n📊 数据统计:")
print(f"  总训练数据: {len(all_train_data)}")
print(f"  测试数据: {len(test_data)}")

# K-Fold设置
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_results = []

print(f"\n🔁 开始 {n_folds}-Fold 交叉验证...")

for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_data), 1):
    print(f"\n{'='*70}")
    print(f"📁 Fold {fold}/{n_folds}")
    print(f"{'='*70}")
    
    # 创建fold的训练集和验证集
    fold_train_data = [all_train_data[i] for i in train_idx]
    fold_val_data = [all_train_data[i] for i in val_idx]
    
    print(f"训练集: {len(fold_train_data)} 样本")
    print(f"验证集: {len(fold_val_data)} 样本")
    
    # 创建数据集对象
    fold_train_dataset = TriClassTEDataset(
        examples=fold_train_data,
        tokenizer=tokenizer,
        max_length=512,
        augment=True
    )
    
    fold_val_dataset = TriClassTEDataset(
        examples=fold_val_data,
        tokenizer=tokenizer,
        max_length=512,
        augment=False
    )
    
    # 初始化模型（每个fold重新初始化）
    model = ImprovedOmniModelForTriClassTE(
        model_name_or_path,
        tokenizer,
        num_labels=9,
        num_classes=3,
        dropout_rate=0.4,
        freeze_layers=6,
        trust_remote_code=True,
        dataset_class=TriClassTEDataset
    )
    
    # 定义metrics
    metric_functions = [
        ClassificationMetric(ignore_y=-100).accuracy_score,
        ClassificationMetric(ignore_y=-100, average='macro').f1_score,
    ]
    
    # 训练器
    trainer = AccelerateTrainer(
        model=model,
        epochs=20,
        learning_rate=1e-5,
        batch_size=8,
        train_dataset=fold_train_dataset,
        eval_dataset=fold_val_dataset,
        test_dataset=None,  # 每个fold不测试
        compute_metrics=metric_functions,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        warmup_steps=100,
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_score",
        greater_is_better=True,
    )
    
    # 训练
    metrics = trainer.train(
        path_to_save=f"ogb_te_kfold_{fold}",
        dataset_class=TriClassTEDataset
    )
    
    fold_results.append({
        'fold': fold,
        'train_accuracy': metrics.get('train_accuracy_score', 0),
        'val_accuracy': metrics.get('eval_accuracy_score', 0),
        'train_f1': metrics.get('train_f1_score', 0),
        'val_f1': metrics.get('eval_f1_score', 0),
    })
    
    print(f"\n✅ Fold {fold} 完成:")
    print(f"  训练准确率: {fold_results[-1]['train_accuracy']:.4f}")
    print(f"  验证准确率: {fold_results[-1]['val_accuracy']:.4f}")
    print(f"  过拟合程度: {fold_results[-1]['train_accuracy'] - fold_results[-1]['val_accuracy']:.4f}")

# 汇总结果
print(f"\n{'='*70}")
print("📊 K-Fold 交叉验证结果汇总")
print(f"{'='*70}")

train_accs = [r['train_accuracy'] for r in fold_results]
val_accs = [r['val_accuracy'] for r in fold_results]
overfitting_gaps = [t - v for t, v in zip(train_accs, val_accs)]

print(f"\n训练集准确率: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
print(f"验证集准确率: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
print(f"平均过拟合程度: {np.mean(overfitting_gaps):.4f} ± {np.std(overfitting_gaps):.4f}")

if np.mean(overfitting_gaps) > 0.20:
    print("\n❌ 警告: 平均过拟合程度 > 0.20，模型泛化能力不足！")
    print("建议:")
    print("  1. 进一步增加正则化（增大dropout, weight_decay）")
    print("  2. 冻结更多层")
    print("  3. 减少训练epoch")
    print("  4. 收集更多数据")
elif np.mean(overfitting_gaps) > 0.10:
    print("\n⚠️  注意: 有轻微过拟合，建议适当增加正则化")
else:
    print("\n✅ 过拟合程度可接受")

print("\n各fold详细结果:")
for r in fold_results:
    print(f"  Fold {r['fold']}: Train={r['train_accuracy']:.4f}, "
          f"Val={r['val_accuracy']:.4f}, Gap={r['train_accuracy']-r['val_accuracy']:.4f}")



