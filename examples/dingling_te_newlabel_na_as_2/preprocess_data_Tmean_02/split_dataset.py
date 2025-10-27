#!/usr/bin/env python3
"""
数据划分脚本：将合并后的tissue标签数据划分为训练集、验证集和测试集
- 支持分层抽样，保持每个tissue的标签分布
- 处理缺失标签的情况
- 生成符合OmniGenBench期望格式的文件
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def analyze_data_distribution(df):
    """
    分析数据分布情况
    
    Args:
        df: 数据框
    
    Returns:
        dict: 统计信息
    """
    print("=" * 60)
    print("数据分布分析")
    print("=" * 60)
    
    # 基本信息
    print(f"总样本数: {len(df)}")
    print(f"特征数: {len(df.columns)}")
    
    # 获取所有tissue标签列
    label_columns = [col for col in df.columns if col.endswith('_TE_label')]
    print(f"Tissue数量: {len(label_columns)}")
    print(f"Tissue列: {label_columns}")
    
    # 分析每个tissue的标签分布
    tissue_stats = {}
    for col in label_columns:
        if col in df.columns:
            # 统计有效标签（非缺失值）
            valid_data = df[col].dropna()
            total_valid = len(valid_data)
            missing_count = df[col].isnull().sum()
            
            if total_valid > 0:
                label_counts = valid_data.value_counts().sort_index()
                label_0 = label_counts.get(0.0, 0)
                label_1 = label_counts.get(1.0, 0)
                label_2 = label_counts.get(2.0, 0)
                
                tissue_stats[col] = {
                    'total_valid': total_valid,
                    'missing': missing_count,
                    'label_0': label_0,
                    'label_1': label_1,
                    'label_2': label_2,
                    'label_distribution': {
                        0: label_0 / total_valid,
                        1: label_1 / total_valid,
                        2: label_2 / total_valid
                    }
                }
                
                print(f"\n{col}:")
                print(f"  有效样本: {total_valid}, 缺失: {missing_count}")
                print(f"  标签分布: 0={label_0}({label_0/total_valid:.3f}), 1={label_1}({label_1/total_valid:.3f}), 2={label_2}({label_2/total_valid:.3f})")
            else:
                print(f"\n{col}: 无有效数据")
                tissue_stats[col] = {
                    'total_valid': 0,
                    'missing': len(df),
                    'label_0': 0,
                    'label_1': 0,
                    'label_2': 0,
                    'label_distribution': {0: 0, 1: 0, 2: 0}
                }
    
    return tissue_stats

def create_stratified_splits(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    创建分层划分的数据集
    
    Args:
        df: 数据框
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        random_state: 随机种子
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"\n开始数据划分...")
    print(f"测试集比例: {test_size}")
    print(f"验证集比例: {val_size}")
    print(f"训练集比例: {1 - test_size - val_size}")
    
    # 获取所有tissue标签列
    label_columns = [col for col in df.columns if col.endswith('_TE_label')]
    
    # 创建分层标签：使用所有tissue标签的组合作为分层依据
    # 对于缺失值，我们用-1表示
    print("创建分层标签...")
    
    # 方法1：使用主要tissue的标签进行分层
    # 选择有最多有效数据的tissue作为主要分层依据
    tissue_valid_counts = {}
    for col in label_columns:
        if col in df.columns:
            valid_count = df[col].dropna().shape[0]
            tissue_valid_counts[col] = valid_count
    
    # 找到有效数据最多的tissue
    main_tissue = max(tissue_valid_counts, key=tissue_valid_counts.get)
    print(f"使用 {main_tissue} 作为主要分层依据 (有效样本: {tissue_valid_counts[main_tissue]})")
    
    # 创建分层标签
    stratify_labels = df[main_tissue].fillna(-1)  # 用-1表示缺失值
    
    # 检查分层标签的分布
    label_dist = stratify_labels.value_counts().sort_index()
    print(f"分层标签分布: {label_dist.to_dict()}")
    
    # 检查是否有足够的样本进行分层
    min_samples = label_dist.min()
    if min_samples < 2:
        print("警告: 某些类别的样本数太少，无法进行分层抽样，将使用随机抽样")
        stratify_labels = None
    
    # 第一次划分：训练+验证 vs 测试
    print("\n第一次划分: 训练+验证 vs 测试")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_labels if stratify_labels is not None else None
    )
    
    print(f"训练+验证集: {len(train_val_df)} 样本")
    print(f"测试集: {len(test_df)} 样本")
    
    # 第二次划分：训练 vs 验证
    print("\n第二次划分: 训练 vs 验证")
    
    # 为验证集创建分层标签
    if stratify_labels is not None:
        val_stratify_labels = train_val_df[main_tissue].fillna(-1)
        val_label_dist = val_stratify_labels.value_counts().sort_index()
        print(f"验证集分层标签分布: {val_label_dist.to_dict()}")
        
        # 检查是否有足够的样本进行分层
        val_min_samples = val_label_dist.min()
        if val_min_samples < 2:
            print("警告: 验证集某些类别的样本数太少，将使用随机抽样")
            val_stratify_labels = None
    else:
        val_stratify_labels = None
    
    # 计算验证集的实际大小
    val_size_actual = val_size / (1 - test_size)  # 从训练+验证集中划分验证集的比例
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_actual,
        random_state=random_state,
        stratify=val_stratify_labels
    )
    
    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")
    
    return train_df, val_df, test_df

def analyze_split_distribution(train_df, val_df, test_df):
    """
    分析划分后各数据集的分布
    
    Args:
        train_df, val_df, test_df: 划分后的数据框
    """
    print("\n" + "=" * 60)
    print("数据划分结果分析")
    print("=" * 60)
    
    datasets = {
        '训练集': train_df,
        '验证集': val_df,
        '测试集': test_df
    }
    
    # 获取所有tissue标签列
    label_columns = [col for col in train_df.columns if col.endswith('_TE_label')]
    
    for dataset_name, df in datasets.items():
        print(f"\n{dataset_name} ({len(df)} 样本):")
        
        for col in label_columns:
            if col in df.columns:
                valid_data = df[col].dropna()
                missing_count = df[col].isnull().sum()
                
                if len(valid_data) > 0:
                    label_counts = valid_data.value_counts().sort_index()
                    label_0 = label_counts.get(0.0, 0)
                    label_1 = label_counts.get(1.0, 0)
                    label_2 = label_counts.get(2.0, 0)
                    total_valid = len(valid_data)
                    
                    print(f"  {col}: 有效={total_valid}, 缺失={missing_count}, "
                          f"0={label_0}({label_0/total_valid:.3f}), "
                          f"1={label_1}({label_1/total_valid:.3f}), "
                          f"2={label_2}({label_2/total_valid:.3f})")
                else:
                    print(f"  {col}: 无有效数据")

def save_datasets(train_df, val_df, test_df, output_dir):
    """
    保存划分后的数据集，生成符合OmniGenBench期望格式的文件
    
    Args:
        train_df, val_df, test_df: 划分后的数据框
        output_dir: 输出目录
    """
    print(f"\n保存数据集到: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据集 - 使用OmniGenBench期望的文件名
    train_file = os.path.join(output_dir, "train.csv")
    val_file = os.path.join(output_dir, "valid.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"训练集: {train_file} ({len(train_df)} 样本)")
    print(f"验证集: {val_file} ({len(val_df)} 样本)")
    print(f"测试集: {test_file} ({len(test_df)} 样本)")
    
    # 保存划分信息
    split_info = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
        'val_ratio': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
        'test_ratio': len(test_df) / (len(train_df) + len(val_df) + len(test_df))
    }
    
    info_file = os.path.join(output_dir, "split_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("数据划分信息\n")
        f.write("=" * 40 + "\n")
        f.write(f"总样本数: {split_info['total_samples']}\n")
        f.write(f"训练集: {split_info['train_samples']} 样本 ({split_info['train_ratio']:.3f})\n")
        f.write(f"验证集: {split_info['val_samples']} 样本 ({split_info['val_ratio']:.3f})\n")
        f.write(f"测试集: {split_info['test_samples']} 样本 ({split_info['test_ratio']:.3f})\n")
    
    print(f"划分信息: {info_file}")

def main():
    """主函数"""
    # 设置路径
    data_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tmean_02"
    input_file = os.path.join(data_dir, "merged_tissue_labels.csv")
    output_dir = data_dir  # 直接保存到数据目录，这样from_hub可以找到
    
    print("=" * 60)
    print("数据划分脚本")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 读取数据
    print(f"读取数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"数据形状: {df.shape}")
    
    # 分析数据分布
    tissue_stats = analyze_data_distribution(df)
    
    # 创建数据划分
    train_df, val_df, test_df = create_stratified_splits(
        df, 
        test_size=0.2,  # 20% 测试集
        val_size=0.2,   # 20% 验证集
        random_state=42
    )
    
    # 分析划分结果
    analyze_split_distribution(train_df, val_df, test_df)
    
    # 保存数据集
    save_datasets(train_df, val_df, test_df, output_dir)
    
    print(f"\n数据划分完成!")
    print(f"输出目录: {output_dir}")
    print(f"生成的文件: train.csv, valid.csv, test.csv")

if __name__ == "__main__":
    main()






