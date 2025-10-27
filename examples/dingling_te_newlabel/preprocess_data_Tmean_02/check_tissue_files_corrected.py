#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查9个tissue的_TE文件中样本的大小以及NA的比率
NA比例相对于样本数（行数）计算
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
    print("NA比例相对于样本数（行数）计算")
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
            na_ratio_by_cells = na_count / total_cells if total_cells > 0 else 0
            
            # 按行统计NA - 这是关键：计算有NA值的样本比例
            na_by_row = df.isna().sum(axis=1)
            rows_with_na = (na_by_row > 0).sum()
            rows_with_all_na = (na_by_row == num_cols).sum()
            
            # NA比例相对于样本数（行数）
            na_ratio_by_samples = rows_with_na / num_rows if num_rows > 0 else 0
            
            # 按列统计NA
            na_by_column = df.isna().sum()
            columns_with_na = (na_by_column > 0).sum()
            columns_with_all_na = (na_by_column == num_rows).sum()
            
            # 计算平均每行NA数量
            avg_na_per_row = na_count / num_rows if num_rows > 0 else 0
            
            result = {
                'file_name': file_name,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'total_cells': total_cells,
                'na_count': na_count,
                'na_ratio_by_cells': na_ratio_by_cells,
                'na_ratio_by_samples': na_ratio_by_samples,
                'rows_with_na': rows_with_na,
                'rows_with_all_na': rows_with_all_na,
                'avg_na_per_row': avg_na_per_row,
                'columns_with_na': columns_with_na,
                'columns_with_all_na': columns_with_all_na
            }
            
            results.append(result)
            
            print(f"\n📊 {file_name}")
            print(f"   样本数 (行数): {num_rows:,}")
            print(f"   特征数 (列数): {num_cols:,}")
            print(f"   总单元格数: {total_cells:,}")
            print(f"   NA值数量: {na_count:,}")
            print(f"   NA比率 (相对于总单元格): {na_ratio_by_cells:.4f} ({na_ratio_by_cells*100:.2f}%)")
            print(f"   NA比率 (相对于样本数): {na_ratio_by_samples:.4f} ({na_ratio_by_samples*100:.2f}%)")
            print(f"   有NA的样本数: {rows_with_na:,}/{num_rows:,}")
            print(f"   全部为NA的样本数: {rows_with_all_na:,}")
            print(f"   平均每样本NA数: {avg_na_per_row:.2f}")
            print(f"   有NA的列数: {columns_with_na}/{num_cols}")
            print(f"   全部为NA的列数: {columns_with_all_na}")
            
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
        print(f"   平均NA比率 (相对于总单元格): {summary_df['na_ratio_by_cells'].mean():.4f} ({summary_df['na_ratio_by_cells'].mean()*100:.2f}%)")
        print(f"   平均NA比率 (相对于样本数): {summary_df['na_ratio_by_samples'].mean():.4f} ({summary_df['na_ratio_by_samples'].mean()*100:.2f}%)")
        
        print(f"\n📋 各文件详细信息 (NA比率相对于样本数):")
        display_df = summary_df[['file_name', 'num_rows', 'num_cols', 'na_ratio_by_samples', 'rows_with_na']].copy()
        display_df.columns = ['文件名', '样本数', '特征数', 'NA样本比例', '有NA的样本数']
        print(display_df.to_string(index=False))
        
        # 保存详细报告
        summary_df.to_csv('tissue_files_analysis_corrected.csv', index=False)
        print(f"\n💾 详细分析结果已保存到: tissue_files_analysis_corrected.csv")
        
        # 检查是否有问题
        print(f"\n🔍 问题检查:")
        high_na_files = summary_df[summary_df['na_ratio_by_samples'] > 0.8]
        if len(high_na_files) > 0:
            print(f"   ⚠️  NA样本比例超过80%的文件:")
            for _, row in high_na_files.iterrows():
                print(f"      {row['file_name']}: {row['na_ratio_by_samples']:.4f} ({row['na_ratio_by_samples']*100:.2f}%)")
        else:
            print(f"   ✅ 所有文件的NA样本比例都在80%以下")
            
        empty_files = summary_df[summary_df['num_rows'] == 0]
        if len(empty_files) > 0:
            print(f"   ⚠️  空文件:")
            for _, row in empty_files.iterrows():
                print(f"      {row['file_name']}")
        else:
            print(f"   ✅ 没有空文件")
            
        # 显示NA分布统计
        print(f"\n📊 NA分布统计:")
        print(f"   平均每样本NA数: {summary_df['avg_na_per_row'].mean():.2f}")
        print(f"   最大每样本NA数: {summary_df['avg_na_per_row'].max():.2f}")
        print(f"   最小每样本NA数: {summary_df['avg_na_per_row'].min():.2f}")

if __name__ == "__main__":
    check_tissue_files()
