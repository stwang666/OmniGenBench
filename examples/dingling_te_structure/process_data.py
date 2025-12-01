#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理9tissue_dot.txt文件：
1. 去除没有标签的样本（LABLE列为空）
2. 按照8:1:1的比例划分为train、valid和test
"""

import random
import os

# 设置随机种子以确保可重复性
random.seed(42)

input_file = '9tissue_dot.txt'
output_train = 'train.csv'
output_valid = 'valid.csv'
output_test = 'test.csv'

# 读取文件
print("正在读取文件...")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 保存表头
header = lines[0]
print(f"表头: {header.strip()}")

# 处理数据行，过滤掉LABLE列为空的行
data_lines = []
no_label_count = 0

for i, line in enumerate(lines[1:], start=2):
    parts = line.strip().split('\t')
    if len(parts) >= 6:
        label = parts[5]  # label列是第6列（索引5）
        if label.strip():  # 如果标签不为空
            data_lines.append(line)
        else:
            no_label_count += 1
    else:
        # 如果列数不够，也视为无标签
        no_label_count += 1

print(f"总行数（不含表头）: {len(lines) - 1}")
print(f"有标签的样本数: {len(data_lines)}")
print(f"无标签的样本数: {no_label_count}")

# 随机打乱数据
random.shuffle(data_lines)

# 计算划分点
total = len(data_lines)
train_size = int(total * 0.8)
valid_size = int(total * 0.1)
# test_size = total - train_size - valid_size

print(f"\n数据集划分:")
print(f"训练集: {train_size} ({train_size/total*100:.1f}%)")
print(f"验证集: {valid_size} ({valid_size/total*100:.1f}%)")
print(f"测试集: {total - train_size - valid_size} ({(total - train_size - valid_size)/total*100:.1f}%)")

# 划分数据集
train_data = data_lines[:train_size]
valid_data = data_lines[train_size:train_size + valid_size]
test_data = data_lines[train_size + valid_size:]

# 将制表符分隔转换为逗号分隔（CSV格式）
def convert_to_csv(line):
    return line.replace('\t', ',')

# 写入文件
print(f"\n正在写入文件...")
with open(output_train, 'w', encoding='utf-8') as f:
    f.write(convert_to_csv(header))
    for line in train_data:
        f.write(convert_to_csv(line))

with open(output_valid, 'w', encoding='utf-8') as f:
    f.write(convert_to_csv(header))
    for line in valid_data:
        f.write(convert_to_csv(line))

with open(output_test, 'w', encoding='utf-8') as f:
    f.write(convert_to_csv(header))
    for line in test_data:
        f.write(convert_to_csv(line))

print(f"✓ 训练集已保存到: {output_train}")
print(f"✓ 验证集已保存到: {output_valid}")
print(f"✓ 测试集已保存到: {output_test}")
print("\n处理完成！")

