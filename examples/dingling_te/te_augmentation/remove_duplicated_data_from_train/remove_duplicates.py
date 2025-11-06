#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从train.csv中删除与valid.csv中ID和seq都相同的样本
"""

import pandas as pd
import os

# 文件路径
valid_file = 'valid.csv'
train_file = 'train.csv'
output_file = 'train.csv'  # 直接覆盖原文件，如果需要备份可以修改

print("正在读取valid.csv...")
valid_df = pd.read_csv(valid_file)

print(f"valid.csv共有 {len(valid_df)} 行数据")

print("正在读取train.csv...")
train_df = pd.read_csv(train_file)

print(f"train.csv原始共有 {len(train_df)} 行数据")

# 创建valid中所有(ID, seq)组合的集合
valid_id_seq_set = set(zip(valid_df['ID'], valid_df['seq']))
print(f"valid.csv中有 {len(valid_id_seq_set)} 个唯一的(ID, seq)组合")

# 过滤train.csv：保留那些(ID, seq)不在valid中的行
train_id_seq_set = set(zip(train_df['ID'], train_df['seq']))
duplicates = train_id_seq_set & valid_id_seq_set
print(f"找到 {len(duplicates)} 个重复的(ID, seq)组合")

# 创建一个布尔掩码：当(ID, seq)不在valid中时为True
mask = ~train_df.apply(lambda row: (row['ID'], row['seq']) in valid_id_seq_set, axis=1)

# 应用掩码过滤数据
train_filtered = train_df[mask]

print(f"删除重复后，train.csv剩余 {len(train_filtered)} 行数据")
print(f"共删除 {len(train_df) - len(train_filtered)} 行数据")

# 保存结果（覆盖原文件）
print(f"正在保存到 {output_file}...")
train_filtered.to_csv(output_file, index=False)

print("完成！")

