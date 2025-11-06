#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算train.csv中各类别的比例
"""

import pandas as pd
import numpy as np

# 读取train.csv文件
train_file = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_02/original_data/train.csv"
df = pd.read_csv(train_file)

# 定义9个组织
tissue_names = [
    'root_TE_label',
    'seedling_TE_label',
    'leaf_TE_label',
    'FMI_TE_label',
    'FOD_TE_label',
    'Prophase-I-pollen_TE_label',
    'Tricellular-pollen_TE_label',
    'flag_TE_label',
    'grain_TE_label'
]

print("=" * 80)
print("📊 Train数据集类别统计")
print("=" * 80)
print(f"\n总样本数: {len(df)}")
print("\n" + "-" * 80)

# 统计每个组织的类别分布
results = []

for tissue in tissue_names:
    # 获取该组织的标签列
    labels = df[tissue].copy()
    
    # 将标签转换为数值，处理空值
    def normalize_label(val):
        if pd.isna(val) or val == '' or str(val).strip() == '':
            return None
        val_str = str(val).strip()
        if val_str in ['0.0', '0']:
            return 0
        elif val_str in ['1.0', '1']:
            return 1
        elif val_str in ['2.0', '2']:
            return 2
        elif val_str.lower() in ['nan', 'na', 'null']:
            return None
        else:
            return None
    
    labels_normalized = labels.apply(normalize_label)
    
    # 统计各类别数量（排除空值）
    valid_labels = labels_normalized.dropna()
    total_valid = len(valid_labels)
    
    count_0 = (valid_labels == 0).sum()
    count_1 = (valid_labels == 1).sum()
    count_2 = (valid_labels == 2).sum()
    count_na = len(labels) - total_valid
    
    # 计算比例
    if total_valid > 0:
        ratio_0 = count_0 / total_valid * 100
        ratio_1 = count_1 / total_valid * 100
        ratio_2 = count_2 / total_valid * 100
    else:
        ratio_0 = ratio_1 = ratio_2 = 0
    
    ratio_na = count_na / len(labels) * 100
    
    results.append({
        'tissue': tissue.replace('_TE_label', ''),
        'total': len(labels),
        'valid': total_valid,
        'count_0': count_0,
        'count_1': count_1,
        'count_2': count_2,
        'count_na': count_na,
        'ratio_0': ratio_0,
        'ratio_1': ratio_1,
        'ratio_2': ratio_2,
        'ratio_na': ratio_na
    })
    
    print(f"\n🔬 {tissue.replace('_TE_label', ''):30s}")
    print(f"  总样本数: {len(labels):6d} | 有效标签: {total_valid:6d} | 空值: {count_na:6d} ({ratio_na:.2f}%)")
    print(f"  类别 0: {count_0:6d} ({ratio_0:6.2f}%) | 类别 1: {count_1:6d} ({ratio_1:6.2f}%) | 类别 2: {count_2:6d} ({ratio_2:6.2f}%)")

# 总体统计
print("\n" + "=" * 80)
print("📈 总体统计（所有组织合并）")
print("=" * 80)

total_all = 0
total_valid_all = 0
total_0_all = 0
total_1_all = 0
total_2_all = 0
total_na_all = 0

for r in results:
    total_all += r['total']
    total_valid_all += r['valid']
    total_0_all += r['count_0']
    total_1_all += r['count_1']
    total_2_all += r['count_2']
    total_na_all += r['count_na']

if total_valid_all > 0:
    overall_ratio_0 = total_0_all / total_valid_all * 100
    overall_ratio_1 = total_1_all / total_valid_all * 100
    overall_ratio_2 = total_2_all / total_valid_all * 100
else:
    overall_ratio_0 = overall_ratio_1 = overall_ratio_2 = 0

overall_ratio_na = total_na_all / total_all * 100

print(f"\n总标签数（所有组织）: {total_all:,}")
print(f"有效标签数: {total_valid_all:,} | 空值数: {total_na_all:,} ({overall_ratio_na:.2f}%)")
print(f"\n类别分布:")
print(f"  类别 0: {total_0_all:8,} ({overall_ratio_0:6.2f}%)")
print(f"  类别 1: {total_1_all:8,} ({overall_ratio_1:6.2f}%)")
print(f"  类别 2: {total_2_all:8,} ({overall_ratio_2:6.2f}%)")

# 计算类别权重建议（用于损失函数）
if total_valid_all > 0:
    # 计算逆频率权重
    n_samples = total_valid_all
    n_classes = 3
    class_counts = [total_0_all, total_1_all, total_2_all]
    
    # 避免除零
    class_counts_safe = [max(c, 1) for c in class_counts]
    
    # 计算权重（标准化使得最小权重为1.0）
    weights = [n_samples / (n_classes * count) for count in class_counts_safe]
    min_weight = min(weights)
    normalized_weights = [w / min_weight for w in weights]
    
    print(f"\n💡 建议的类别权重（用于CrossEntropyLoss）:")
    print(f"  class_weights = torch.tensor({normalized_weights}, dtype=torch.float32)")
    print(f"  # 说明: 类别0权重={normalized_weights[0]:.3f}, 类别1权重={normalized_weights[1]:.3f}, 类别2权重={normalized_weights[2]:.3f}")

print("\n" + "=" * 80)
print("✅ 统计完成！")
print("=" * 80)

