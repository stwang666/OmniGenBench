#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查9个tissue的_TE文件中样本的大小以及NA的比率
"""

import pandas as pd
import os
import numpy as np

def check_tissue_files():
    """检查tissue文件的样本大小和NA比率"""
    
    # 定义9个tissue文件
    tissue_files = [
        'FMI_TE.csv',
        'FOD_TE.csv', 
        'Tricellular-pollen_TE.csv',
        'Prophase-I-pollen_TE.csv',
        'seedling_TE.csv',
        'root_TE.csv',
        'leaf_TE.csv',
        'grain_TE.csv',
        'flag_TE.csv'
    ]
    
    results = []
    
    print("=" * 80)
    print("检查9个tissue的_TE文件样本大小和NA比率")
    print("=" * 80)
    
    for file_name in tissue_files:
        file_path = os.path.join('/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02', file_name)
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_name}")
            continue
            
        try:
            # 读取文件
            df = pd.read_csv(file_path)
            
            # 基本信息
            num_rows, num_cols = df.shape
            total_cells = num_rows * num_cols
            
            # 计算NA值
            na_count = df.isna().sum().sum()
            na_ratio = na_count / total_cells if total_cells > 0 else 0
            
            # 按列统计NA
            na_by_column = df.isna().sum()
            columns_with_na = (na_by_column > 0).sum()
            columns_with_all_na = (na_by_column == num_rows).sum()
            
            # 按行统计NA
            na_by_row = df.isna().sum(axis=1)
            rows_with_na = (na_by_row > 0).sum()
            rows_with_all_na = (na_by_row == num_cols).sum()
            
            result = {
                'file_name': file_name,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'total_cells': total_cells,
                'na_count': na_count,
                'na_ratio': na_ratio,
                'columns_with_na': columns_with_na,
                'columns_with_all_na': columns_with_all_na,
                'rows_with_na': rows_with_na,
                'rows_with_all_na': rows_with_all_na
            }
            
            results.append(result)
            
            print(f"\n📊 {file_name}")
            print(f"   样本数 (行数): {num_rows:,}")
            print(f"   特征数 (列数): {num_cols:,}")
            print(f"   总单元格数: {total_cells:,}")
            print(f"   NA值数量: {na_count:,}")
            print(f"   NA比率: {na_ratio:.4f} ({na_ratio*100:.2f}%)")
            print(f"   有NA的列数: {columns_with_na}/{num_cols}")
            print(f"   全部为NA的列数: {columns_with_all_na}")
            print(f"   有NA的行数: {rows_with_na}/{num_rows}")
            print(f"   全部为NA的行数: {rows_with_all_na}")
            
        except Exception as e:
            print(f"❌ 读取文件 {file_name} 时出错: {str(e)}")
            continue
    
    # 生成汇总报告
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)
    
    if results:
        # 创建汇总DataFrame
        summary_df = pd.DataFrame(results)
        
        print(f"\n📈 总体统计:")
        print(f"   总文件数: {len(results)}")
        print(f"   总样本数: {summary_df['num_rows'].sum():,}")
        print(f"   总特征数: {summary_df['num_cols'].sum():,}")
        print(f"   总NA值: {summary_df['na_count'].sum():,}")
        print(f"   平均NA比率: {summary_df['na_ratio'].mean():.4f} ({summary_df['na_ratio'].mean()*100:.2f}%)")
        
        print(f"\n📋 各文件详细信息:")
        print(summary_df[['file_name', 'num_rows', 'num_cols', 'na_ratio']].to_string(index=False))
        
        # 保存详细报告
        summary_df.to_csv('tissue_files_analysis.csv', index=False)
        print(f"\n💾 详细分析结果已保存到: tissue_files_analysis.csv")
        
        # 检查是否有问题
        print(f"\n🔍 问题检查:")
        high_na_files = summary_df[summary_df['na_ratio'] > 0.1]
        if len(high_na_files) > 0:
            print(f"   ⚠️  NA比率超过10%的文件:")
            for _, row in high_na_files.iterrows():
                print(f"      {row['file_name']}: {row['na_ratio']:.4f} ({row['na_ratio']*100:.2f}%)")
        else:
            print(f"   ✅ 所有文件的NA比率都在10%以下")
            
        empty_files = summary_df[summary_df['num_rows'] == 0]
        if len(empty_files) > 0:
            print(f"   ⚠️  空文件:")
            for _, row in empty_files.iterrows():
                print(f"      {row['file_name']}")
        else:
            print(f"   ✅ 没有空文件")

if __name__ == "__main__":
    check_tissue_files()
