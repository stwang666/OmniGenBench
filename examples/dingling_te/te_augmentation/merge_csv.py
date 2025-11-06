#!/usr/bin/env python3
"""
合并两个CSV文件
"""
import pandas as pd

# 读取两个CSV文件
file1 = "train_original.csv"
file2 = "train_filtered_3_augmented.csv"
output = "train.csv"

print(f"正在读取 {file1}...")
df1 = pd.read_csv(file1)

print(f"正在读取 {file2}...")
df2 = pd.read_csv(file2)

print(f"\n文件1行数: {len(df1)}")
print(f"文件2行数: {len(df2)}")

# 合并两个数据框
print("\n正在合并...")
df_merged = pd.concat([df1, df2], ignore_index=True)

print(f"合并后总行数: {len(df_merged)}")
print(f"\n正在保存到 {output}...")
df_merged.to_csv(output, index=False)

print(f"✅ 合并完成！输出文件: {output}")

