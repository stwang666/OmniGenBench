#!/usr/bin/env python3
"""
合并9个tissue文件的label脚本
- 通过ID和Seq比对合并数据
- 将NA值转换为2，0和1保持不变
- 重命名label列为文件名_TE_label格式
- 处理缺失的tissue标签
- 按照指定顺序排列tissue列
- 使用外连接保存所有表中的ID和序列
- 合并后的缺失值表示该tissue在该样本中没有标签数据
"""

import pandas as pd
import os
import glob
from pathlib import Path

def merge_tissue_labels(data_dir):
    """
    合并9个tissue文件的label
    
    Args:
        data_dir: 包含tissue文件的目录路径
    
    Returns:
        pd.DataFrame: 合并后的数据框
    """
    
    # 定义tissue的顺序（根据图片中的顺序）
    tissue_order = [
        'root',
        'seedling',
        'leaf',
        'FMI',
        'FOD',
        'Prophase-I-pollen',
        'Tricellular-pollen',
        'flag',
        'grain'
    ]
    
    # 获取所有tissue文件
    tissue_files = glob.glob(os.path.join(data_dir, "*_TE.csv"))
    print(f"找到 {len(tissue_files)} 个tissue文件:")
    for file in tissue_files:
        print(f"  - {os.path.basename(file)}")
    
    # 存储所有数据框
    tissue_data = {}
    
    # 读取每个tissue文件
    for file_path in tissue_files:
        filename = os.path.basename(file_path)
        tissue_name = filename.replace('_TE.csv', '')
        
        print(f"\n正在处理 {filename}...")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"  原始数据形状: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        
        # 检查必要的列是否存在
        if 'ID' not in df.columns or 'Seq' not in df.columns or 'label' not in df.columns:
            print(f"  警告: {filename} 缺少必要的列 (ID, Seq, label)")
            continue
        
        # 选择需要的列：ID, Seq, label
        df_subset = df[['ID', 'Seq', 'label']].copy()
        
        # 处理label列：将NA转换为2，0和1保持不变
        # 注意：这里只处理原始文件中的NA值，合并后产生的缺失值保持为NaN
        df_subset['label'] = df_subset['label'].fillna(2)
        
        # 重命名label列为 tissue_name_TE_label格式
        df_subset = df_subset.rename(columns={'label': f'{tissue_name}_TE_label'})
        
        # 存储数据
        tissue_data[tissue_name] = df_subset
        
        print(f"  处理后数据形状: {df_subset.shape}")
        print(f"  label值分布: {df_subset[f'{tissue_name}_TE_label'].value_counts().to_dict()}")
    
    if not tissue_data:
        print("错误: 没有找到有效的tissue文件")
        return None
    
    # 合并所有tissue数据
    print(f"\n开始合并 {len(tissue_data)} 个tissue文件...")
    
    # 从第一个tissue开始合并
    tissue_names = list(tissue_data.keys())
    merged_df = tissue_data[tissue_names[0]].copy()
    
    print(f"  基准文件: {tissue_names[0]} (形状: {merged_df.shape})")
    
    # 逐个合并其他tissue，使用外连接保存所有表中的ID和序列
    for tissue_name in tissue_names[1:]:
        print(f"  合并 {tissue_name}...")
        
        # 使用ID和Seq进行外连接
        tissue_df = tissue_data[tissue_name]
        
        # 外连接，保留所有数据
        merged_df = pd.merge(
            merged_df, 
            tissue_df, 
            on=['ID', 'Seq'], 
            how='outer'
        )
        
        print(f"    合并后形状: {merged_df.shape}")
    
    # 将合并后的Seq列重命名为sequence
    merged_df = merged_df.rename(columns={'Seq': 'sequence'})
    
    # 重新排列列的顺序
    # 将ID和sequence放在前面，然后按照指定顺序排列tissue的label列
    label_columns = []
    for tissue in tissue_order:
        col_name = f'{tissue}_TE_label'
        if col_name in merged_df.columns:
            label_columns.append(col_name)
    
    # 添加任何不在预定义顺序中的label列（作为安全措施）
    # 正常情况下这个循环不应该添加任何列，因为所有tissue都应该在tissue_order中
    for col in merged_df.columns:
        if col.endswith('_TE_label') and col not in label_columns:
            label_columns.append(col)
    
    other_columns = ['ID', 'sequence']
    final_columns = other_columns + label_columns
    merged_df = merged_df[final_columns]
    
    print(f"\n最终合并结果:")
    print(f"  数据形状: {merged_df.shape}")
    print(f"  列名: {list(merged_df.columns)}")
    
    # 统计每个tissue的label分布
    print(f"\n各tissue的label分布:")
    for col in label_columns:
        if col in merged_df.columns:
            value_counts = merged_df[col].value_counts(dropna=False)
            print(f"  {col}: {value_counts.to_dict()}")
    
    # 统计缺失值
    print(f"\n缺失值统计（表示该tissue在该样本中没有标签数据）:")
    missing_stats = merged_df[label_columns].isnull().sum()
    for col, missing_count in missing_stats.items():
        print(f"  {col}: {missing_count} 个缺失值")
    
    # 统计有效标签值的分布
    print(f"\n有效标签值统计（0, 1, 2）:")
    for col in label_columns:
        if col in merged_df.columns:
            valid_labels = merged_df[col].dropna()
            total_valid = len(valid_labels)
            label_0 = (valid_labels == 0.0).sum()
            label_1 = (valid_labels == 1.0).sum()
            label_2 = (valid_labels == 2.0).sum()
            print(f"  {col}: 总计={total_valid}, 0={label_0}, 1={label_1}, 2(NA)={label_2}")
    
    return merged_df

def main():
    """主函数"""
    # 设置数据目录
    data_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_02"
    
    print("=" * 60)
    print("合并9个tissue文件的label")
    print("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在: {data_dir}")
        return
    
    # 合并数据
    merged_df = merge_tissue_labels(data_dir)
    
    if merged_df is not None:
        # 保存合并后的数据
        output_file = os.path.join(data_dir, "merged_tissue_labels.csv")
        merged_df.to_csv(output_file, index=False)
        
        print(f"\n合并完成!")
        print(f"输出文件: {output_file}")
        print(f"数据形状: {merged_df.shape}")
        
        # 显示前几行数据
        print(f"\n前5行数据预览:")
        print(merged_df.head())
        
        # 显示数据统计信息
        print(f"\n数据统计信息:")
        print(f"总样本数: {len(merged_df)}")
        print(f"tissue数量: {len([col for col in merged_df.columns if col.endswith('_TE_label')])}")
        
    else:
        print("合并失败!")

if __name__ == "__main__":
    main()
