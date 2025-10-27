#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: split_a_data.py
# time: 2025-01-27
# author: 划分A亚基因组数据为训练集、验证集和测试集

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def load_a_data(data_file):
    """
    加载A亚基因组过滤后的数据
    
    参数:
        data_file: A亚基因组数据文件路径
    
    返回:
        DataFrame: 加载的数据
    """
    print(f"📖 读取A亚基因组数据: {data_file}")
    
    if not Path(data_file).exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"   ✅ 加载 {len(df)} 个样本")
    
    return df

def analyze_data_distribution(df):
    """
    分析A亚基因组数据分布情况
    
    参数:
        df: A亚基因组数据DataFrame
    """
    print("\n📊 A亚基因组数据分布分析")
    print("=" * 60)
    
    print(f"总样本数: {len(df)}")
    
    # 分析标签分布
    label_cols = [col for col in df.columns if col.endswith('_TE_label')]
    print(f"包含的组织类型: {len(label_cols)}")
    print(f"组织类型: {[col.replace('_TE_label', '') for col in label_cols]}")
    
    # 统计每个样本的标签数量
    label_counts = df[label_cols].notna().sum(axis=1)
    print(f"平均每个样本的标签数: {label_counts.mean():.1f}")
    print(f"标签数分布:")
    for i in range(1, len(label_cols)+1):
        count = (label_counts == i).sum()
        if count > 0:
            print(f"  {i}个标签: {count} 样本 ({count/len(df)*100:.1f}%)")
    
    # 分析每个组织类型的标签分布
    print(f"\n各组织类型标签分布:")
    for col in label_cols:
        tissue_name = col.replace('_TE_label', '')
        non_null_count = df[col].notna().sum()
        print(f"  {tissue_name}: {non_null_count} 样本有标签 ({non_null_count/len(df)*100:.1f}%)")

def split_a_data(df, strategy='stratified', test_size=0.2, valid_size=0.2, random_state=42):
    """
    划分A亚基因组数据
    
    参数:
        df: A亚基因组数据DataFrame
        strategy: 划分策略 ('random', 'stratified', 'balanced')
        test_size: 测试集比例
        valid_size: 验证集比例（相对于剩余数据）
        random_state: 随机种子
    
    返回:
        训练集、验证集、测试集的DataFrame
    """
    print(f"\n🔀 划分A亚基因组数据 (策略: {strategy})")
    print("=" * 60)
    
    # 计算实际比例
    total_size = len(df)
    test_count = int(total_size * test_size)
    remaining_size = total_size - test_count
    valid_count = int(remaining_size * valid_size)
    train_count = remaining_size - valid_count
    
    print(f"📋 划分方案:")
    print(f"  总样本数: {total_size}")
    print(f"  训练集: {train_count} 样本 ({train_count/total_size*100:.1f}%)")
    print(f"  验证集: {valid_count} 样本 ({valid_count/total_size*100:.1f}%)")
    print(f"  测试集: {test_count} 样本 ({test_count/total_size*100:.1f}%)")
    
    if strategy == 'random':
        # 随机划分
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        train_df = df_shuffled[:train_count].copy()
        valid_df = df_shuffled[train_count:train_count+valid_count].copy()
        test_df = df_shuffled[train_count+valid_count:].copy()
        
    elif strategy == 'stratified':
        # 分层划分 - 基于标签数量进行分层
        label_cols = [col for col in df.columns if col.endswith('_TE_label')]
        label_counts = df[label_cols].notna().sum(axis=1)
        
        # 创建分层标签（基于标签数量）
        df['label_count'] = label_counts
        
        # 先划分训练集和临时集
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_size + valid_size, 
            stratify=df['label_count'], 
            random_state=random_state
        )
        
        # 再从临时集中划分验证集和测试集
        valid_ratio = valid_count / (test_count + valid_count)
        valid_df, test_df = train_test_split(
            temp_df, 
            test_size=1-valid_ratio, 
            stratify=temp_df['label_count'], 
            random_state=random_state
        )
        
        # 删除临时列
        for df_split in [train_df, valid_df, test_df]:
            if 'label_count' in df_split.columns:
                df_split.drop('label_count', axis=1, inplace=True)
    
    elif strategy == 'balanced':
        # 平衡划分 - 确保每个组织类型在各集合中都有代表性
        label_cols = [col for col in df.columns if col.endswith('_TE_label')]
        
        # 为每个组织类型创建分层
        train_dfs = []
        valid_dfs = []
        test_dfs = []
        
        for col in label_cols:
            tissue_name = col.replace('_TE_label', '')
            tissue_data = df[df[col].notna()].copy()
            
            if len(tissue_data) < 3:  # 如果样本太少，随机分配
                if len(tissue_data) == 1:
                    train_dfs.append(tissue_data)
                elif len(tissue_data) == 2:
                    train_dfs.append(tissue_data.iloc[:1])
                    valid_dfs.append(tissue_data.iloc[1:2])
                continue
            
            # 按7:2:1划分
            tissue_train, tissue_temp = train_test_split(
                tissue_data, 
                test_size=0.3, 
                random_state=random_state
            )
            tissue_valid, tissue_test = train_test_split(
                tissue_temp, 
                test_size=0.33, 
                random_state=random_state
            )
            
            train_dfs.append(tissue_train)
            valid_dfs.append(tissue_valid)
            test_dfs.append(tissue_test)
            
            print(f"  {tissue_name}: 训练{len(tissue_train)}, 验证{len(tissue_valid)}, 测试{len(tissue_test)}")
        
        # 合并所有组织类型的数据
        train_df = pd.concat(train_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
        valid_df = pd.concat(valid_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
        
        # 如果还有未分配的样本，随机分配
        all_assigned = set(train_df.index) | set(valid_df.index) | set(test_df.index)
        remaining_indices = set(df.index) - all_assigned
        
        if remaining_indices:
            remaining_df = df.loc[list(remaining_indices)]
            remaining_train, remaining_temp = train_test_split(
                remaining_df, 
                test_size=0.3, 
                random_state=random_state
            )
            remaining_valid, remaining_test = train_test_split(
                remaining_temp, 
                test_size=0.33, 
                random_state=random_state
            )
            
            train_df = pd.concat([train_df, remaining_train], ignore_index=True)
            valid_df = pd.concat([valid_df, remaining_valid], ignore_index=True)
            test_df = pd.concat([test_df, remaining_test], ignore_index=True)
    
    print(f"\n📋 最终划分结果:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(valid_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    
    return train_df, valid_df, test_df

def save_datasets(train_df, valid_df, test_df, output_dir):
    """
    保存划分后的数据集
    
    参数:
        train_df: 训练集DataFrame
        valid_df: 验证集DataFrame
        test_df: 测试集DataFrame
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 保存数据集到 {output_dir}")
    
    # 保存数据文件
    train_file = output_dir / "train.csv"
    valid_file = output_dir / "valid.csv"
    test_file = output_dir / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"   ✅ 训练集: {train_file} ({len(train_df)} 样本)")
    print(f"   ✅ 验证集: {valid_file} ({len(valid_df)} 样本)")
    print(f"   ✅ 测试集: {test_file} ({len(test_df)} 样本)")
    
    # 生成统计报告
    report_file = output_dir / "data_split_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("A亚基因组数据划分报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(train_df) + len(valid_df) + len(test_df)}\n\n")
        
        f.write("训练集:\n")
        f.write(f"  样本数: {len(train_df)}\n")
        f.write(f"  比例: {len(train_df)/(len(train_df) + len(valid_df) + len(test_df))*100:.1f}%\n")
        
        # 分析训练集标签分布
        label_cols = [col for col in train_df.columns if col.endswith('_TE_label')]
        f.write(f"  各组织类型标签数:\n")
        for col in label_cols:
            tissue_name = col.replace('_TE_label', '')
            count = train_df[col].notna().sum()
            f.write(f"    {tissue_name}: {count}\n")
        
        f.write("\n验证集:\n")
        f.write(f"  样本数: {len(valid_df)}\n")
        f.write(f"  比例: {len(valid_df)/(len(train_df) + len(valid_df) + len(test_df))*100:.1f}%\n")
        
        f.write(f"  各组织类型标签数:\n")
        for col in label_cols:
            tissue_name = col.replace('_TE_label', '')
            count = valid_df[col].notna().sum()
            f.write(f"    {tissue_name}: {count}\n")
        
        f.write("\n测试集:\n")
        f.write(f"  样本数: {len(test_df)}\n")
        f.write(f"  比例: {len(test_df)/(len(train_df) + len(valid_df) + len(test_df))*100:.1f}%\n")
        
        f.write(f"  各组织类型标签数:\n")
        for col in label_cols:
            tissue_name = col.replace('_TE_label', '')
            count = test_df[col].notna().sum()
            f.write(f"    {tissue_name}: {count}\n")
    
    print(f"   ✅ 统计报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='划分A亚基因组数据为训练集、验证集和测试集')
    parser.add_argument('--data_file', type=str, 
                        default='merged_A_tissue_data_filtered.csv',
                        help='A亚基因组数据文件路径')
    parser.add_argument('--strategy', type=str, 
                        choices=['random', 'stratified', 'balanced'], 
                        default='stratified', help='数据划分策略')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='验证集比例（相对于剩余数据）')
    parser.add_argument('--output_dir', type=str, default='split_a',
                        help='输出目录')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    print("🧬 A亚基因组数据划分")
    print("=" * 60)
    
    try:
        # 加载数据
        df = load_a_data(args.data_file)
        
        # 分析数据分布
        analyze_data_distribution(df)
        
        # 划分数据
        train_df, valid_df, test_df = split_a_data(
            df, 
            strategy=args.strategy,
            test_size=args.test_size,
            valid_size=args.valid_size,
            random_state=args.random_state
        )
        
        # 保存数据集
        save_datasets(train_df, valid_df, test_df, args.output_dir)
        
        print("\n🎉 A亚基因组数据划分完成！")
        print("\n💡 建议:")
        print("  - 使用训练集训练模型")
        print("  - 使用验证集调参和模型选择")
        print("  - 使用测试集进行最终评估")
        print("  - 确保数据划分的随机性和代表性")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
