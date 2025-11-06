#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析train.csv中标签类别的分布情况
"""

import pandas as pd
import numpy as np

# 读取CSV文件
print("正在读取train.csv文件...")
df = pd.read_csv('train.csv')

print(f"总样本数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n列名: {list(df.columns)}\n")

# 获取所有标签列（从第3列开始，排除ID和sequence）
label_columns = df.columns[2:].tolist()
print(f"标签列数量: {len(label_columns)}")
print(f"标签列: {label_columns}\n")

# 分析每个标签列的分布
print("=" * 80)
print("各标签列的类别分布情况")
print("=" * 80)

for col in label_columns:
    print(f"\n【{col}】")
    print("-" * 60)
    
    # 统计各值的数量
    value_counts = df[col].value_counts(dropna=False)
    total = len(df[col])
    
    # 计算比例
    print(f"总样本数: {total}")
    print(f"\n类别分布:")
    for val, count in value_counts.items():
        percentage = (count / total) * 100
        if pd.isna(val):
            print(f"  空值 (NA): {count:>8} ({percentage:>6.2f}%)")
        else:
            print(f"  {val:>4}: {count:>8} ({percentage:>6.2f}%)")
    
    # 统计非空值
    non_null_count = df[col].notna().sum()
    null_count = df[col].isna().sum()
    print(f"\n非空值数量: {non_null_count}")
    print(f"空值数量: {null_count}")

# 整体统计
print("\n" + "=" * 80)
print("整体统计信息")
print("=" * 80)

# 统计所有标签列的类别分布
all_labels = []
for col in label_columns:
    all_labels.extend(df[col].dropna().tolist())

if all_labels:
    all_labels = pd.Series(all_labels)
    print(f"\n所有标签列（合并）的类别分布:")
    value_counts = all_labels.value_counts()
    total = len(all_labels)
    for val, count in value_counts.items():
        percentage = (count / total) * 100
        print(f"  {val:>4}: {count:>10} ({percentage:>6.2f}%)")
    print(f"\n总标签数（非空）: {total}")

# 统计每行有多少个非空标签
print(f"\n每行非空标签数量分布:")
non_null_counts = df[label_columns].notna().sum(axis=1)
non_null_counts_dist = non_null_counts.value_counts().sort_index()
for count, freq in non_null_counts_dist.items():
    percentage = (freq / len(df)) * 100
    print(f"  {count:>2} 个非空标签: {freq:>8} 行 ({percentage:>6.2f}%)")

# 统计完全为空的行
completely_empty = (df[label_columns].isna().all(axis=1)).sum()
print(f"\n完全为空的行数: {completely_empty} ({completely_empty/len(df)*100:.2f}%)")

print("\n分析完成！")

















