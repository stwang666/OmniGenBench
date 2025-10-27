#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: split_data_by_subgenome.py
# time: 10:00 22/10/2025
# author: 按亚基因组划分训练集、验证集和测试集，避免数据泄漏

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_filtered_data(base_dir):
    """
    加载过滤后的A、B、D亚基因组数据
    
    参数:
        base_dir: 数据目录
    
    返回:
        包含A、B、D数据的字典
    """
    base_dir = Path(base_dir)
    data = {}
    
    subgenomes = ['A', 'B', 'D']
    
    for subgenome in subgenomes:
        file_path = base_dir / f"merged_{subgenome}_tissue_data_filtered.csv"
        
        if not file_path.exists():
            print(f"⚠️  警告: 找不到文件 {file_path}")
            continue
        
        print(f"📖 读取{subgenome}亚基因组数据...")
        df = pd.read_csv(file_path)
        
        # 添加亚基因组标识符
        df['subgenome'] = subgenome
        
        data[subgenome] = df
        print(f"   ✅ 加载 {len(df)} 个样本")
    
    return data

def analyze_data_distribution(data):
    """
    分析数据分布情况
    
    参数:
        data: 包含各亚基因组数据的字典
    """
    print("\n📊 数据分布分析")
    print("=" * 60)
    
    total_samples = sum(len(df) for df in data.values())
    print(f"总样本数: {total_samples}")
    
    for subgenome, df in data.items():
        print(f"\n{subgenome}亚基因组:")
        print(f"  样本数: {len(df)} ({len(df)/total_samples*100:.1f}%)")
        
        # 分析标签分布
        label_cols = [col for col in df.columns if col.endswith('_TE_label')]
        print(f"  包含tissue: {len(label_cols)}")
        
        # 统计每个样本的标签数量
        label_counts = df[label_cols].notna().sum(axis=1)
        print(f"  平均每个样本的标签数: {label_counts.mean():.1f}")
        print(f"  标签数分布:")
        for i in range(1, len(label_cols)+1):
            count = (label_counts == i).sum()
            if count > 0:
                print(f"    {i}个标签: {count} 样本 ({count/len(df)*100:.1f}%)")

def split_by_subgenome(data, strategy='balanced'):
    """
    按亚基因组划分数据
    
    参数:
        data: 包含各亚基因组数据的字典
        strategy: 划分策略 ('balanced', 'size_based', 'mixed', 'ab_train_d_split', 'ad_train_b_split')
    
    返回:
        训练集、验证集、测试集的DataFrame
    """
    print(f"\n🔀 按亚基因组划分数据 (策略: {strategy})")
    print("=" * 60)
    
    if strategy == 'balanced':
        # 方案1: 平衡划分 - 每个亚基因组一个用途
        train_df = data['A'].copy()
        valid_df = data['B'].copy()
        test_df = data['D'].copy()
        
        print("📋 划分方案:")
        print(f"  训练集: A亚基因组 ({len(train_df)} 样本)")
        print(f"  验证集: B亚基因组 ({len(valid_df)} 样本)")
        print(f"  测试集: D亚基因组 ({len(test_df)} 样本)")
        
    elif strategy == 'ab_train_d_split':
        # 方案2: A+B作为训练集，D分为验证集和测试集
        train_df = pd.concat([data['A'], data['B']], ignore_index=True)
        
        # 将D亚基因组按7:3划分为验证集和测试集
        d_data = data['D'].sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(len(d_data) * 0.7)
        
        valid_df = d_data[:valid_size].copy()
        test_df = d_data[valid_size:].copy()
        
        print("📋 划分方案:")
        print(f"  训练集: A+B亚基因组 ({len(train_df)} 样本)")
        print(f"  验证集: D亚基因组70% ({len(valid_df)} 样本)")
        print(f"  测试集: D亚基因组30% ({len(test_df)} 样本)")
        
    elif strategy == 'ad_train_b_split':
        # 方案3: A+D作为训练集，B分为验证集和测试集
        train_df = pd.concat([data['A'], data['D']], ignore_index=True)
        
        # 将B亚基因组按7:3划分为验证集和测试集
        b_data = data['B'].sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(len(b_data) * 0.7)
        
        valid_df = b_data[:valid_size].copy()
        test_df = b_data[valid_size:].copy()
        
        print("📋 划分方案:")
        print(f"  训练集: A+D亚基因组 ({len(train_df)} 样本)")
        print(f"  验证集: B亚基因组70% ({len(valid_df)} 样本)")
        print(f"  测试集: B亚基因组30% ({len(test_df)} 样本)")
        
    elif strategy == 'size_based':
        # 方案4: 基于样本大小的划分
        # 训练集用最大的两个亚基因组，测试集用最小的
        train_df = pd.concat([data['A'], data['B']], ignore_index=True)
        valid_df = data['D'].copy()
        test_df = None
        
        print("📋 划分方案:")
        print(f"  训练集: A+B亚基因组 ({len(train_df)} 样本)")
        print(f"  验证集: D亚基因组 ({len(valid_df)} 样本)")
        print(f"  测试集: 从训练集随机抽取10%")
        
        # 从训练集中随机抽取10%作为测试集
        test_size = int(len(train_df) * 0.1)
        test_indices = np.random.choice(len(train_df), test_size, replace=False)
        test_df = train_df.iloc[test_indices].copy()
        train_df = train_df.drop(test_indices).reset_index(drop=True)
        
        print(f"  调整后训练集: {len(train_df)} 样本")
        print(f"  测试集: {len(test_df)} 样本")
        
    elif strategy == 'mixed':
        # 方案5: 混合划分 - 每个亚基因组都分配到不同用途
        # 将每个亚基因组按7:2:1划分
        train_dfs = []
        valid_dfs = []
        test_dfs = []
        
        for subgenome, df in data.items():
            # 随机打乱数据
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # 按7:2:1划分
            train_size = int(len(df_shuffled) * 0.7)
            valid_size = int(len(df_shuffled) * 0.2)
            
            train_part = df_shuffled[:train_size]
            valid_part = df_shuffled[train_size:train_size+valid_size]
            test_part = df_shuffled[train_size+valid_size:]
            
            train_dfs.append(train_part)
            valid_dfs.append(valid_part)
            test_dfs.append(test_part)
            
            print(f"  {subgenome}亚基因组: 训练{train_size}, 验证{len(valid_part)}, 测试{len(test_part)}")
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        valid_df = pd.concat(valid_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"\n📋 最终划分:")
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
        f.write("数据划分报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {len(train_df) + len(valid_df) + len(test_df)}\n\n")
        
        f.write("训练集:\n")
        f.write(f"  样本数: {len(train_df)}\n")
        if 'subgenome' in train_df.columns:
            subgenome_counts = train_df['subgenome'].value_counts()
            f.write(f"  亚基因组分布: {dict(subgenome_counts)}\n")
        
        f.write("\n验证集:\n")
        f.write(f"  样本数: {len(valid_df)}\n")
        if 'subgenome' in valid_df.columns:
            subgenome_counts = valid_df['subgenome'].value_counts()
            f.write(f"  亚基因组分布: {dict(subgenome_counts)}\n")
        
        f.write("\n测试集:\n")
        f.write(f"  样本数: {len(test_df)}\n")
        if 'subgenome' in test_df.columns:
            subgenome_counts = test_df['subgenome'].value_counts()
            f.write(f"  亚基因组分布: {dict(subgenome_counts)}\n")
    
    print(f"   ✅ 统计报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='按亚基因组划分训练集、验证集和测试集')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='包含过滤后数据文件的目录')
    parser.add_argument('--strategy', type=str, 
                        choices=['balanced', 'size_based', 'mixed', 'ab_train_d_split', 'ad_train_b_split'], 
                        default='balanced', help='数据划分策略')
    parser.add_argument('--output_dir', type=str, default='split_data',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("🧬 按亚基因组划分数据，避免数据泄漏")
    print("=" * 60)
    
    # 加载数据
    data = load_filtered_data(args.data_dir)
    
    if not data:
        print("❌ 没有找到任何数据文件")
        return
    
    # 分析数据分布
    analyze_data_distribution(data)
    
    # 划分数据
    train_df, valid_df, test_df = split_by_subgenome(data, args.strategy)
    
    # 保存数据集
    save_datasets(train_df, valid_df, test_df, args.output_dir)
    
    print("\n🎉 数据划分完成！")
    print("\n💡 建议:")
    print("  - 使用训练集训练模型")
    print("  - 使用验证集调参和模型选择")
    print("  - 使用测试集进行最终评估")
    print("  - 确保验证集和测试集来自不同亚基因组，避免数据泄漏")

if __name__ == '__main__':
    main()
