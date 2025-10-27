#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
划分FOD_TE数据为训练集、验证集和测试集
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_fod_te_data():
    """划分FOD_TE数据为训练集、验证集和测试集"""
    
    print("=" * 60)
    print("划分FOD_TE数据为训练集、验证集和测试集")
    print("=" * 60)
    
    # 读取FOD_TE数据
    input_file = '/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE.csv'
    output_dir = '/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE'
    
    print(f"📂 读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"📊 原始数据信息:")
    print(f"   样本数: {len(df):,}")
    print(f"   特征数: {df.shape[1]}")
    print(f"   列名: {list(df.columns)}")
    
    # 检查数据质量
    na_count = df.isna().sum().sum()
    na_ratio = na_count / (df.shape[0] * df.shape[1])
    print(f"   NA值数量: {na_count:,}")
    print(f"   NA比率: {na_ratio:.4f} ({na_ratio*100:.2f}%)")
    
    # 数据划分
    print(f"\n🔄 开始数据划分...")
    
    # 第一次划分：训练集70%，临时集30%
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42,
        shuffle=True
    )
    
    # 第二次划分：临时集分为验证集和测试集各50%
    valid_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        shuffle=True
    )
    
    print(f"✅ 数据划分完成!")
    print(f"   训练集: {len(train_df):,} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   验证集: {len(valid_df):,} 样本 ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"   测试集: {len(test_df):,} 样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 保存数据
    print(f"\n💾 保存数据到 {output_dir}...")
    
    train_file = os.path.join(output_dir, 'train.csv')
    valid_file = os.path.join(output_dir, 'valid.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"   ✅ 训练集保存到: {train_file}")
    print(f"   ✅ 验证集保存到: {valid_file}")
    print(f"   ✅ 测试集保存到: {test_file}")
    
    # 生成划分报告
    report_file = os.path.join(output_dir, 'data_split_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FOD_TE数据划分报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"原始数据信息:\n")
        f.write(f"  样本数: {len(df):,}\n")
        f.write(f"  特征数: {df.shape[1]}\n")
        f.write(f"  NA值数量: {na_count:,}\n")
        f.write(f"  NA比率: {na_ratio:.4f} ({na_ratio*100:.2f}%)\n\n")
        
        f.write(f"数据划分结果:\n")
        f.write(f"  训练集: {len(train_df):,} 样本 ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"  验证集: {len(valid_df):,} 样本 ({len(valid_df)/len(df)*100:.1f}%)\n")
        f.write(f"  测试集: {len(test_df):,} 样本 ({len(test_df)/len(df)*100:.1f}%)\n\n")
        
        f.write(f"文件保存位置:\n")
        f.write(f"  训练集: {train_file}\n")
        f.write(f"  验证集: {valid_file}\n")
        f.write(f"  测试集: {test_file}\n")
        
        f.write(f"\n划分参数:\n")
        f.write(f"  随机种子: 42\n")
        f.write(f"  训练集比例: 70%\n")
        f.write(f"  验证集比例: 15%\n")
        f.write(f"  测试集比例: 15%\n")
    
    print(f"📋 划分报告保存到: {report_file}")
    
    # 验证保存的文件
    print(f"\n🔍 验证保存的文件...")
    for file_path, name in [(train_file, "训练集"), (valid_file, "验证集"), (test_file, "测试集")]:
        if os.path.exists(file_path):
            df_check = pd.read_csv(file_path)
            print(f"   ✅ {name}: {len(df_check):,} 样本, {df_check.shape[1]} 特征")
        else:
            print(f"   ❌ {name}: 文件不存在")
    
    print(f"\n🎉 FOD_TE数据划分完成!")
    print(f"📁 所有文件已保存到: {output_dir}")

if __name__ == "__main__":
    split_fod_te_data()
