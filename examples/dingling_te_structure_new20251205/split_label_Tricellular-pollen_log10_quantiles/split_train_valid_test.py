#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Tricellular-pollen三分类数据按照8:1:1的比例划分为train、valid和test数据集
使用分层抽样保持各类别比例一致
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(input_file, output_dir=None, random_seed=42):
    """
    将数据按照8:1:1的比例划分为train、valid和test
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_dir : str, optional
        输出目录，如果为None则使用输入文件所在目录
    random_seed : int
        随机种子，默认42
    """
    print("="*70)
    print("数据集划分：8:1:1 (train:valid:test)")
    print("="*70)
    
    # 读取数据
    print(f"\n读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据维度: {df.shape}")
    
    # 检查是否有label列
    if 'label' not in df.columns:
        print("错误: 数据文件中没有'label'列！")
        return
    
    # 显示原始数据标签分布
    print("\n原始数据标签分布:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  label={int(label)}: {count} ({count/len(df)*100:.2f}%)")
    
    # 第一步：先分出80%的训练集和20%的临时集
    print("\n第一步: 划分训练集(80%)和临时集(20%)...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=random_seed,
        stratify=df['label']  # 按照label的比例分层抽样
    )
    
    print(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"临时集: {len(temp_df)} 样本 ({len(temp_df)/len(df)*100:.1f}%)")
    
    # 第二步：将临时集平分为验证集和测试集（各10%）
    print("\n第二步: 将临时集划分为验证集(10%)和测试集(10%)...")
    valid_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=random_seed,
        stratify=temp_df['label']
    )
    
    print(f"验证集: {len(valid_df)} 样本 ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")
    
    # 显示各数据集中标签的分布
    print("\n" + "="*70)
    print("各数据集标签分布")
    print("="*70)
    
    print("\n训练集标签分布:")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        print(f"  label={int(label)}: {count:5d} ({count/len(train_df)*100:.2f}%)")
    
    print("\n验证集标签分布:")
    valid_label_counts = valid_df['label'].value_counts().sort_index()
    for label, count in valid_label_counts.items():
        print(f"  label={int(label)}: {count:5d} ({count/len(valid_df)*100:.2f}%)")
    
    print("\n测试集标签分布:")
    test_label_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        print(f"  label={int(label)}: {count:5d} ({count/len(test_df)*100:.2f}%)")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path(input_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存划分后的数据集
    train_file = output_dir / 'train.csv'
    valid_file = output_dir / 'valid.csv'
    test_file = output_dir / 'test.csv'
    
    print("\n" + "="*70)
    print("保存数据集")
    print("="*70)
    
    train_df.to_csv(train_file, index=False)
    print(f"✓ 训练集已保存: {train_file}")
    print(f"  样本数: {len(train_df)}")
    
    valid_df.to_csv(valid_file, index=False)
    print(f"✓ 验证集已保存: {valid_file}")
    print(f"  样本数: {len(valid_df)}")
    
    test_df.to_csv(test_file, index=False)
    print(f"✓ 测试集已保存: {test_file}")
    print(f"  样本数: {len(test_df)}")
    
    # 统计信息汇总
    print("\n" + "="*70)
    print("划分结果汇总")
    print("="*70)
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"验证集: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"总计: {len(train_df) + len(valid_df) + len(test_df)}")
    
    return train_df, valid_df, test_df


def main():
    """主函数"""
    # 输入文件路径
    input_file = '9tissue_structure_te_hc_deseq2_tp_split_log10_Tricellular-pollen_triclass.csv'
    
    # 如果文件不在当前目录，尝试使用完整路径
    if not Path(input_file).exists():
        input_file = Path(__file__).parent / input_file
    
    if not Path(input_file).exists():
        print(f"错误: 找不到输入文件: {input_file}")
        print("请确保文件路径正确")
        return
    
    # 执行数据划分
    split_dataset(
        input_file=str(input_file),
        output_dir=None,  # 保存到输入文件所在目录
        random_seed=42
    )
    
    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
