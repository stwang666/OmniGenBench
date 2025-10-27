#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去除FOD_TE数据中标签为NA的样本，然后重新划分数据集
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def remove_na_and_split_fod_te():
    """去除NA样本并重新划分数据集"""
    
    print("=" * 60)
    print("去除FOD_TE数据中标签为NA的样本并重新划分数据集")
    print("=" * 60)
    
    # 文件路径
    input_file = '/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE.csv'
    output_dir = '/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE_remove_na'
    
    print(f"📂 读取原始数据: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"📊 原始数据信息:")
    print(f"   样本数: {len(df):,}")
    print(f"   特征数: {df.shape[1]}")
    print(f"   列名: {list(df.columns)}")
    
    # 检查NA情况
    na_in_label = df['label'].isna().sum()
    na_in_seq = df['Seq'].isna().sum()
    total_na = na_in_label + na_in_seq
    
    print(f"\n🔍 NA值检查:")
    print(f"   label列NA数: {na_in_label:,}")
    print(f"   Seq列NA数: {na_in_seq:,}")
    print(f"   总NA数: {total_na:,}")
    print(f"   NA比例: {total_na/(len(df)*df.shape[1])*100:.2f}%")
    
    # 去除NA样本
    print(f"\n🧹 去除NA样本...")
    df_clean = df.dropna(subset=['label', 'Seq'])
    removed_count = len(df) - len(df_clean)
    
    print(f"   原始样本数: {len(df):,}")
    print(f"   去除NA后样本数: {len(df_clean):,}")
    print(f"   去除的样本数: {removed_count:,}")
    print(f"   保留比例: {len(df_clean)/len(df)*100:.2f}%")
    
    # 检查清洗后的标签分布
    if len(df_clean) > 0:
        label_counts = df_clean['label'].value_counts()
        print(f"\n📊 清洗后标签分布:")
        for label, count in label_counts.items():
            print(f"   {label}: {count:,} ({count/len(df_clean)*100:.2f}%)")
    
    # 数据划分
    print(f"\n🔄 开始数据划分...")
    
    # 第一次划分：训练集70%，临时集30%
    train_df, temp_df = train_test_split(
        df_clean, 
        test_size=0.3, 
        random_state=42,
        shuffle=True,
        stratify=df_clean['label']  # 使用分层抽样保持标签比例
    )
    
    # 第二次划分：临时集分为验证集和测试集各50%
    valid_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42,
        shuffle=True,
        stratify=temp_df['label']  # 使用分层抽样保持标签比例
    )
    
    print(f"✅ 数据划分完成!")
    print(f"   训练集: {len(train_df):,} 样本 ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"   验证集: {len(valid_df):,} 样本 ({len(valid_df)/len(df_clean)*100:.1f}%)")
    print(f"   测试集: {len(test_df):,} 样本 ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    # 检查各数据集的标签分布
    print(f"\n📊 各数据集标签分布:")
    for name, data in [("训练集", train_df), ("验证集", valid_df), ("测试集", test_df)]:
        label_counts = data['label'].value_counts()
        print(f"   {name}:")
        for label, count in label_counts.items():
            print(f"     {label}: {count:,} ({count/len(data)*100:.2f}%)")
    
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
    
    # 生成详细报告
    report_file = os.path.join(output_dir, 'data_cleaning_and_split_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FOD_TE数据清洗和划分报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"原始数据信息:\n")
        f.write(f"  样本数: {len(df):,}\n")
        f.write(f"  特征数: {df.shape[1]}\n")
        f.write(f"  label列NA数: {na_in_label:,}\n")
        f.write(f"  Seq列NA数: {na_in_seq:,}\n")
        f.write(f"  总NA数: {total_na:,}\n\n")
        
        f.write(f"数据清洗结果:\n")
        f.write(f"  原始样本数: {len(df):,}\n")
        f.write(f"  清洗后样本数: {len(df_clean):,}\n")
        f.write(f"  去除样本数: {removed_count:,}\n")
        f.write(f"  保留比例: {len(df_clean)/len(df)*100:.2f}%\n\n")
        
        f.write(f"清洗后标签分布:\n")
        for label, count in df_clean['label'].value_counts().items():
            f.write(f"  {label}: {count:,} ({count/len(df_clean)*100:.2f}%)\n")
        f.write(f"\n")
        
        f.write(f"数据划分结果:\n")
        f.write(f"  训练集: {len(train_df):,} 样本 ({len(train_df)/len(df_clean)*100:.1f}%)\n")
        f.write(f"  验证集: {len(valid_df):,} 样本 ({len(valid_df)/len(df_clean)*100:.1f}%)\n")
        f.write(f"  测试集: {len(test_df):,} 样本 ({len(test_df)/len(df_clean)*100:.1f}%)\n\n")
        
        f.write(f"各数据集标签分布:\n")
        for name, data in [("训练集", train_df), ("验证集", valid_df), ("测试集", test_df)]:
            f.write(f"  {name}:\n")
            for label, count in data['label'].value_counts().items():
                f.write(f"    {label}: {count:,} ({count/len(data)*100:.2f}%)\n")
            f.write(f"\n")
        
        f.write(f"文件保存位置:\n")
        f.write(f"  训练集: {train_file}\n")
        f.write(f"  验证集: {valid_file}\n")
        f.write(f"  测试集: {test_file}\n")
        
        f.write(f"\n划分参数:\n")
        f.write(f"  随机种子: 42\n")
        f.write(f"  训练集比例: 70%\n")
        f.write(f"  验证集比例: 15%\n")
        f.write(f"  测试集比例: 15%\n")
        f.write(f"  分层抽样: 是\n")
    
    print(f"📋 详细报告保存到: {report_file}")
    
    # 验证保存的文件
    print(f"\n🔍 验证保存的文件...")
    for file_path, name in [(train_file, "训练集"), (valid_file, "验证集"), (test_file, "测试集")]:
        if os.path.exists(file_path):
            df_check = pd.read_csv(file_path)
            print(f"   ✅ {name}: {len(df_check):,} 样本, {df_check.shape[1]} 特征")
            
            # 检查是否有NA值
            na_count = df_check.isna().sum().sum()
            if na_count == 0:
                print(f"      ✅ 无NA值")
            else:
                print(f"      ⚠️  仍有{na_count}个NA值")
        else:
            print(f"   ❌ {name}: 文件不存在")
    
    print(f"\n🎉 FOD_TE数据清洗和划分完成!")
    print(f"📁 所有文件已保存到: {output_dir}")
    
    # 最终统计
    print(f"\n📈 最终统计:")
    print(f"   原始数据: {len(df):,} 样本")
    print(f"   清洗后数据: {len(df_clean):,} 样本")
    print(f"   训练集: {len(train_df):,} 样本")
    print(f"   验证集: {len(valid_df):,} 样本")
    print(f"   测试集: {len(test_df):,} 样本")
    print(f"   数据利用率: {len(df_clean)/len(df)*100:.2f}%")

if __name__ == "__main__":
    remove_na_and_split_fod_te()
