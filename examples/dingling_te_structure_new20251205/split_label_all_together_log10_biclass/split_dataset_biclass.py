#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理合并后的数据：移除label为1的样本，只保留label为0和2的样本，并划分为train、valid、test数据集
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 数据所在目录和文件
data_dir = '/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205/split_label_all_together_log10_biclass'
input_file = os.path.join(data_dir, '9tissue_structure_te_hc_deseq2_tp_split_log10_triclass_mean_std.csv')

# 读取合并后的文件
print("正在读取合并文件...")
merged_df = pd.read_csv(input_file)

print("="*60)
print("数据预处理与划分（二分类：只保留label=0和label=2）")
print("="*60)

# 移除label为空的样本
print(f"\n原始数据总数: {len(merged_df)}")
print(f"label为空的样本数: {merged_df['label'].isna().sum()}")

# 只保留label不为空的样本
valid_df = merged_df.dropna(subset=['label']).copy()
print(f"移除空label后的有效样本数: {len(valid_df)}")

# 显示原始label分布
print("\n原始label分布：")
label_counts = valid_df['label'].value_counts()
for label, count in label_counts.items():
    print(f"  label={int(label)}: {count} ({count/len(valid_df)*100:.2f}%)")

# 只保留label为0和2的样本（移除label=1的样本）
valid_df = valid_df[valid_df['label'].isin([0, 2])].copy()
print(f"\n移除label=1后的样本数: {len(valid_df)}")

# 显示有效样本中各组织的分布
print("\n有效样本中各组织的分布：")
tissue_counts = valid_df['tissue'].value_counts()
for tissue, count in tissue_counts.items():
    print(f"  {tissue}: {count}")

print("\n有效样本中label的分布：")
label_counts = valid_df['label'].value_counts()
for label, count in label_counts.items():
    print(f"  label={int(label)}: {count} ({count/len(valid_df)*100:.2f}%)")

# 设置随机种子以确保可重复性
random_seed = 42

# 第一步：先分出80%的训练集和20%的临时集
train_df, temp_df = train_test_split(
    valid_df, 
    test_size=0.2, 
    random_state=random_seed,
    stratify=valid_df['label']  # 按照label的比例分层抽样
)

# 第二步：将临时集平分为验证集和测试集（各10%）
valid_df_split, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=random_seed,
    stratify=temp_df['label']
)

print(f"\n数据集划分结果：")
print(f"训练集 (80%): {len(train_df)} 样本")
print(f"验证集 (10%): {len(valid_df_split)} 样本")
print(f"测试集 (10%): {len(test_df)} 样本")
print(f"总计: {len(train_df) + len(valid_df_split) + len(test_df)} 样本")

# 显示各数据集中label的分布
print("\n训练集label分布：")
for label in sorted(train_df['label'].unique()):
    count = (train_df['label']==label).sum()
    print(f"  label={int(label)}: {count} ({count/len(train_df)*100:.2f}%)")

print("\n验证集label分布：")
for label in sorted(valid_df_split['label'].unique()):
    count = (valid_df_split['label']==label).sum()
    print(f"  label={int(label)}: {count} ({count/len(valid_df_split)*100:.2f}%)")

print("\n测试集label分布：")
for label in sorted(test_df['label'].unique()):
    count = (test_df['label']==label).sum()
    print(f"  label={int(label)}: {count} ({count/len(test_df)*100:.2f}%)")

# 显示各数据集中tissue的分布
print("\n训练集各组织分布：")
for tissue, count in train_df['tissue'].value_counts().items():
    print(f"  {tissue}: {count}")

print("\n验证集各组织分布：")
for tissue, count in valid_df_split['tissue'].value_counts().items():
    print(f"  {tissue}: {count}")

print("\n测试集各组织分布：")
for tissue, count in test_df['tissue'].value_counts().items():
    print(f"  {tissue}: {count}")

# 保存划分后的数据集
train_file = os.path.join(data_dir, 'train.csv')
valid_file = os.path.join(data_dir, 'valid.csv')
test_file = os.path.join(data_dir, 'test.csv')

print("\n正在保存数据集...")
train_df.to_csv(train_file, index=False)
valid_df_split.to_csv(valid_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\n数据集已保存：")
print(f"  - 训练集: {train_file}")
print(f"  - 验证集: {valid_file}")
print(f"  - 测试集: {test_file}")

print("\n" + "="*60)
print("处理完成！")
print("="*60)
