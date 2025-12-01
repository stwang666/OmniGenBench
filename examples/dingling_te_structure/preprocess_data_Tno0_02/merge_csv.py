#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并9个组织的TE CSV文件
"""

import pandas as pd
import os

# 定义文件名和对应的组织名称
file_tissue_mapping = {
    'root_TE.csv': 'root',
    'seedling_TE.csv': 'seedling',
    'leaf_TE.csv': 'leaf',
    'FMI_TE.csv': 'FMI',
    'FOD_TE.csv': 'FOD',
    'Prophase-I-pollen_TE.csv': 'Prophase-I-pollen',
    'Tricellular-pollen_TE.csv': 'Tricellular-pollen',
    'flag_TE.csv': 'flag',
    'grain_TE.csv': 'grain'
}

# 存储所有数据的列表
all_data = []

# 数据所在目录
data_dir = '/home/sw1136/OmniGenBench/examples/dingling_te_structure/preprocess_data_Tno0_02'

# 遍历每个文件
for filename, tissue in file_tissue_mapping.items():
    filepath = os.path.join(data_dir, filename)
    print(f"正在处理: {filename} (tissue: {tissue})")
    
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 提取需要的列：ID, Seq, label
    # 添加tissue列
    df_subset = pd.DataFrame({
        'ID': df['ID'],
        'sequence': df['Seq'],
        'tissue': tissue,
        'label': df['label']
    })
    
    all_data.append(df_subset)
    print(f"  - 读取了 {len(df_subset)} 条记录")

# 合并所有数据
merged_df = pd.concat(all_data, ignore_index=True)

print(f"\n总共合并了 {len(merged_df)} 条记录")
print(f"各组织的样本数量：")
print(merged_df['tissue'].value_counts())

# 保存合并后的文件
output_file = os.path.join(data_dir, 'merged_all_tissues_TE.csv')
merged_df.to_csv(output_file, index=False)
print(f"\n合并后的文件已保存到: {output_file}")

# 显示前几行
print("\n前5行数据预览：")
print(merged_df.head())

# 显示数据统计信息
print("\n数据统计信息：")
print(merged_df.info())
print("\nlabel分布：")
print(merged_df['label'].value_counts())

# ========== 处理有效样本并划分数据集 ==========
print("\n" + "="*60)
print("开始处理有效样本并划分数据集...")
print("="*60)

# 移除label为空的样本
print(f"\n原始数据总数: {len(merged_df)}")
print(f"label为空的样本数: {merged_df['label'].isna().sum()}")

# 只保留label不为空的样本
valid_df = merged_df.dropna(subset=['label']).copy()
print(f"移除空label后的有效样本数: {len(valid_df)}")

# 显示有效样本中各组织的分布
print("\n有效样本中各组织的分布：")
print(valid_df['tissue'].value_counts())
print("\n有效样本中label的分布：")
print(valid_df['label'].value_counts())

# 打乱数据
from sklearn.model_selection import train_test_split

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

# 显示各数据集中label的分布
print("\n训练集label分布：")
print(train_df['label'].value_counts())
print(f"  - label=0: {(train_df['label']==0).sum()} ({(train_df['label']==0).sum()/len(train_df)*100:.2f}%)")
print(f"  - label=1: {(train_df['label']==1).sum()} ({(train_df['label']==1).sum()/len(train_df)*100:.2f}%)")

print("\n验证集label分布：")
print(valid_df_split['label'].value_counts())
print(f"  - label=0: {(valid_df_split['label']==0).sum()} ({(valid_df_split['label']==0).sum()/len(valid_df_split)*100:.2f}%)")
print(f"  - label=1: {(valid_df_split['label']==1).sum()} ({(valid_df_split['label']==1).sum()/len(valid_df_split)*100:.2f}%)")

print("\n测试集label分布：")
print(test_df['label'].value_counts())
print(f"  - label=0: {(test_df['label']==0).sum()} ({(test_df['label']==0).sum()/len(test_df)*100:.2f}%)")
print(f"  - label=1: {(test_df['label']==1).sum()} ({(test_df['label']==1).sum()/len(test_df)*100:.2f}%)")

# 保存划分后的数据集
train_file = os.path.join(data_dir, 'train.csv')
valid_file = os.path.join(data_dir, 'valid.csv')
test_file = os.path.join(data_dir, 'test.csv')

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

