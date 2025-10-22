# -*- coding: utf-8 -*-
# file: merge_tissue_data.py
# time: 10:00 22/10/2025
# author: Merge script for 9 tissue CSV files
# 合并9个tissue的CSV文件到一个文件

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def read_tissue_data(file_path, tissue_name):
    """
    读取单个tissue的CSV文件
    
    参数:
        file_path: CSV文件路径
        tissue_name: tissue名称
    
    返回:
        DataFrame with ID, Seq, label columns
    """
    print(f"📖 读取 {tissue_name}_TE.csv...")
    
    try:
        df = pd.read_csv(file_path)
        
        # 检查必需的列
        if 'ID' not in df.columns:
            print(f"   ❌ 错误: 文件中没有'ID'列")
            return None
        
        if 'Seq' not in df.columns:
            print(f"   ❌ 错误: 文件中没有'Seq'列")
            return None
            
        if 'label' not in df.columns:
            print(f"   ❌ 错误: 文件中没有'label'列")
            return None
        
        # 只保留需要的列
        tissue_df = df[['ID', 'Seq', 'label']].copy()
        
        # 重命名label列为tissue特定的名称
        tissue_df.rename(columns={'label': f'{tissue_name}_TE_label'}, inplace=True)
        
        # 统计信息
        total_samples = len(tissue_df)
        label_counts = tissue_df[f'{tissue_name}_TE_label'].value_counts(dropna=False)
        
        print(f"   ✅ 读取 {total_samples} 个样本")
        print(f"   标签分布:")
        for label, count in label_counts.items():
            if pd.isna(label):
                print(f"      NA: {count}")
            else:
                print(f"      {label}: {count}")
        
        return tissue_df
        
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return None


def merge_tissue_dataframes(tissue_dfs, tissue_names):
    """
    合并多个tissue的DataFrame
    
    参数:
        tissue_dfs: tissue DataFrame列表
        tissue_names: tissue名称列表
    
    返回:
        合并后的DataFrame
    """
    print(f"\n🔗 开始合并数据...")
    
    if not tissue_dfs:
        print("❌ 没有数据可以合并")
        return None
    
    # 从第一个DataFrame开始
    merged_df = tissue_dfs[0].copy()
    print(f"   基础数据: {tissue_names[0]} ({len(merged_df)} 个样本)")
    
    # 依次合并其他DataFrame
    for i, (df, tissue_name) in enumerate(zip(tissue_dfs[1:], tissue_names[1:]), 1):
        print(f"   合并 {tissue_name}...")
        
        # 使用outer join保留所有ID
        before_count = len(merged_df)
        merged_df = pd.merge(
            merged_df,
            df,
            on=['ID', 'Seq'],
            how='outer',
            suffixes=('', f'_{tissue_name}')
        )
        after_count = len(merged_df)
        
        if after_count > before_count:
            print(f"      新增 {after_count - before_count} 个样本")
    
    print(f"   ✅ 合并完成，总共 {len(merged_df)} 个样本")
    
    return merged_df


def check_sequence_consistency(merged_df):
    """
    检查ID相同的样本是否有不同的序列
    
    参数:
        merged_df: 合并后的DataFrame
    """
    print(f"\n🔍 检查序列一致性...")
    
    # 按ID分组，检查是否有多个不同的序列
    grouped = merged_df.groupby('ID')['Seq'].nunique()
    inconsistent_ids = grouped[grouped > 1]
    
    if len(inconsistent_ids) > 0:
        print(f"   ⚠️  警告: 发现 {len(inconsistent_ids)} 个ID有不同的序列")
        print(f"   示例 (前5个):")
        for id_name in inconsistent_ids.head(5).index:
            seqs = merged_df[merged_df['ID'] == id_name]['Seq'].unique()
            print(f"      {id_name}: {len(seqs)} 个不同序列")
    else:
        print(f"   ✅ 所有ID的序列一致")


def generate_statistics(merged_df, tissue_names):
    """
    生成数据统计报告
    
    参数:
        merged_df: 合并后的DataFrame
        tissue_names: tissue名称列表
    """
    print(f"\n📊 数据统计报告")
    print("=" * 60)
    
    # 总体统计
    print(f"\n总样本数: {len(merged_df)}")
    
    # 每个tissue的标签统计
    print(f"\n各tissue标签分布:")
    for tissue in tissue_names:
        label_col = f'{tissue}_TE_label'
        if label_col in merged_df.columns:
            total = len(merged_df)
            non_na = merged_df[label_col].notna().sum()
            label_0 = (merged_df[label_col] == 0).sum()
            label_1 = (merged_df[label_col] == 1).sum()
            na_count = merged_df[label_col].isna().sum()
            
            print(f"\n{tissue:25s}:")
            print(f"   有效标签: {non_na}/{total} ({non_na/total*100:.1f}%)")
            print(f"   标签=0: {label_0} ({label_0/total*100:.1f}%)")
            print(f"   标签=1: {label_1} ({label_1/total*100:.1f}%)")
            print(f"   标签=NA: {na_count} ({na_count/total*100:.1f}%)")
    
    # 每个样本有多少个tissue有标签
    label_cols = [f'{tissue}_TE_label' for tissue in tissue_names if f'{tissue}_TE_label' in merged_df.columns]
    merged_df['num_tissues_with_label'] = merged_df[label_cols].notna().sum(axis=1)
    
    print(f"\n每个样本的tissue标签数量分布:")
    label_count_dist = merged_df['num_tissues_with_label'].value_counts().sort_index()
    for num_labels, count in label_count_dist.items():
        print(f"   {int(num_labels)} 个tissue: {count} 个样本 ({count/len(merged_df)*100:.1f}%)")
    
    # 删除临时列
    merged_df.drop('num_tissues_with_label', axis=1, inplace=True)


def merge_tissue_files(data_dir, output_file=None, generate_report=True):
    """
    合并9个tissue的CSV文件
    
    参数:
        data_dir: 包含9个tissue CSV文件的目录
        output_file: 输出文件路径（默认为data_dir/merged_tissue_data.csv）
        generate_report: 是否生成统计报告
    
    返回:
        合并后的DataFrame
    """
    data_dir = Path(data_dir)
    
    # 9个tissue的名称（按照指定顺序）
    tissue_names = [
        'root', 'seedling', 'leaf', 'FMI', 'FOD',
        'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
    ]
    
    print("=" * 60)
    print("合并9个Tissue数据文件")
    print("=" * 60)
    print(f"\n📁 数据目录: {data_dir}")
    
    # 读取所有tissue数据
    tissue_dfs = []
    successful_tissues = []
    
    for tissue in tissue_names:
        csv_file = data_dir / f"{tissue}_TE.csv"
        
        if not csv_file.exists():
            print(f"\n⚠️  警告: 找不到文件 {csv_file}")
            continue
        
        df = read_tissue_data(csv_file, tissue)
        
        if df is not None:
            tissue_dfs.append(df)
            successful_tissues.append(tissue)
    
    if not tissue_dfs:
        print("\n❌ 错误: 没有成功读取任何tissue数据")
        return None
    
    # 合并数据
    merged_df = merge_tissue_dataframes(tissue_dfs, successful_tissues)
    
    if merged_df is None:
        return None
    
    # 检查序列一致性
    check_sequence_consistency(merged_df)
    
    # 确保列的顺序正确
    label_cols = [f'{tissue}_TE_label' for tissue in tissue_names if f'{tissue}_TE_label' in merged_df.columns]
    column_order = ['ID', 'Seq'] + label_cols
    
    # 重命名Seq为sequence（更统一）
    merged_df.rename(columns={'Seq': 'sequence'}, inplace=True)
    column_order = ['ID', 'sequence'] + label_cols
    
    # 只保留指定的列
    merged_df = merged_df[column_order]
    
    # 生成统计报告
    if generate_report:
        generate_statistics(merged_df, successful_tissues)
    
    # 保存文件
    if output_file is None:
        output_file = data_dir / "merged_tissue_data.csv"
    else:
        output_file = Path(output_file)
    
    print(f"\n💾 保存合并后的数据...")
    merged_df.to_csv(output_file, index=False)
    print(f"   ✅ 已保存到: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 保存统计报告到文本文件
    if generate_report:
        report_file = output_file.parent / "merge_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("合并9个Tissue数据文件 - 统计报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总样本数: {len(merged_df)}\n")
            f.write(f"输出文件: {output_file}\n")
            f.write(f"\n成功合并的tissue: {', '.join(successful_tissues)}\n")
            
            # 各tissue标签统计
            f.write(f"\n各tissue标签分布:\n")
            for tissue in successful_tissues:
                label_col = f'{tissue}_TE_label'
                if label_col in merged_df.columns:
                    total = len(merged_df)
                    non_na = merged_df[label_col].notna().sum()
                    label_0 = (merged_df[label_col] == 0).sum()
                    label_1 = (merged_df[label_col] == 1).sum()
                    na_count = merged_df[label_col].isna().sum()
                    
                    f.write(f"\n{tissue}:\n")
                    f.write(f"  有效标签: {non_na}/{total} ({non_na/total*100:.1f}%)\n")
                    f.write(f"  标签=0: {label_0} ({label_0/total*100:.1f}%)\n")
                    f.write(f"  标签=1: {label_1} ({label_1/total*100:.1f}%)\n")
                    f.write(f"  标签=NA: {na_count} ({na_count/total*100:.1f}%)\n")
        
        print(f"   ✅ 统计报告已保存到: {report_file}")
    
    print("\n🎉 合并完成！")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(description='合并9个tissue的CSV文件')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='包含9个tissue CSV文件的目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径（默认为data_dir/merged_tissue_data.csv）')
    parser.add_argument('--no_report', action='store_true',
                        help='不生成统计报告')
    
    args = parser.parse_args()
    
    merge_tissue_files(
        data_dir=args.data_dir,
        output_file=args.output,
        generate_report=not args.no_report
    )


if __name__ == '__main__':
    main()

