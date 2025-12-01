#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计9个CSV文件中标签0和1的数量
"""

import pandas as pd
import os

# 定义要统计的9个CSV文件
csv_files = [
    "Tricellular-pollen_TE.csv",
    "seedling_TE.csv",
    "root_TE.csv",
    "Prophase-I-pollen_TE.csv",
    "leaf_TE.csv",
    "grain_TE.csv",
    "FOD_TE.csv",
    "FMI_TE.csv",
    "flag_TE.csv"
]

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("统计各CSV文件中标签0和1的数量")
print("=" * 60)
print()

results = []

for csv_file in csv_files:
    file_path = os.path.join(script_dir, csv_file)
    
    if not os.path.exists(file_path):
        print(f"警告: 文件 {csv_file} 不存在，跳过")
        continue
    
    try:
        # 读取CSV文件
        print(f"正在处理: {csv_file}...")
        df = pd.read_csv(file_path)
        
        # 检查是否有label列
        if 'label' not in df.columns:
            print(f"  警告: {csv_file} 中没有找到 'label' 列")
            continue
        
        # 统计标签数量
        label_counts = df['label'].value_counts()
        
        # 获取标签0和1的数量（如果存在）
        count_0 = label_counts.get(0, 0)
        count_1 = label_counts.get(1, 0)
        total = len(df)
        
        # 显示所有标签的统计（用于调试）
        print(f"  总行数: {total}")
        print(f"  标签0的数量: {count_0}")
        print(f"  标签1的数量: {count_1}")
        print(f"  其他标签: {label_counts.drop([0, 1], errors='ignore').sum() if 0 in label_counts.index or 1 in label_counts.index else label_counts.sum()}")
        print(f"  标签分布: {dict(label_counts)}")
        print()
        
        results.append({
            '文件名': csv_file,
            '总行数': total,
            '标签0': count_0,
            '标签1': count_1
        })
        
    except Exception as e:
        print(f"  错误: 处理 {csv_file} 时出错: {str(e)}")
        print()

# 打印汇总结果
print("=" * 60)
print("汇总结果")
print("=" * 60)
print(f"{'文件名':<35} {'总行数':<10} {'标签0':<10} {'标签1':<10}")
print("-" * 60)

total_rows = 0
total_label_0 = 0
total_label_1 = 0

for result in results:
    print(f"{result['文件名']:<35} {result['总行数']:<10} {result['标签0']:<10} {result['标签1']:<10}")
    total_rows += result['总行数']
    total_label_0 += result['标签0']
    total_label_1 += result['标签1']

print("-" * 60)
print(f"{'总计':<35} {total_rows:<10} {total_label_0:<10} {total_label_1:<10}")
print("=" * 60)

