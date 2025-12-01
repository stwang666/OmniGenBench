#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将数据集中的label从浮点数转换为整数
"""

import pandas as pd
import os

# 数据所在目录
data_dir = '/home/sw1136/OmniGenBench/examples/dingling_te_structure/preprocess_data_Tno0_02'

# 需要转换的文件列表
files = ['train.csv', 'valid.csv', 'test.csv']

for filename in files:
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        continue
    
    print(f"\n处理文件: {filename}")
    
    # 读取数据
    df = pd.read_csv(filepath)
    
    print(f"  原始label类型: {df['label'].dtype}")
    print(f"  原始label唯一值: {df['label'].unique()}")
    
    # 将label转换为整数
    df['label'] = df['label'].astype(int)
    
    print(f"  转换后label类型: {df['label'].dtype}")
    print(f"  转换后label唯一值: {df['label'].unique()}")
    
    # 保存
    df.to_csv(filepath, index=False)
    print(f"  ✅ 已保存")

print("\n" + "="*60)
print("所有文件处理完成！")
print("="*60)


