#!/usr/bin/env python3
"""
处理CSV文件：
1. 对TE列取log10
2. 修改TISSUE列中的组织名称
   - Prophase.I.pollen -> Prophase-I-pollen
   - Tricellular.pollen -> Tricellular-pollen
"""

import pandas as pd
import numpy as np

def process_te_data(input_file, output_file=None):
    """
    处理翻译效率数据
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str, optional
        输出CSV文件路径，如果为None则覆盖原文件
    """
    print("="*70)
    print("开始处理文件...")
    print("="*70)
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    print(f"\n原始数据维度: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 显示原始数据的前几行
    print("\n原始数据样例（前3行）:")
    print(df.head(3))
    
    # ========================================
    # 步骤1: 对TE列取log10
    # ========================================
    print("\n" + "="*70)
    print("步骤1: 对TE列取log10")
    print("="*70)
    
    # 检查TE列的值
    print(f"\nTE列的统计信息（转换前）:")
    print(f"  最小值: {df['TE'].min():.6f}")
    print(f"  最大值: {df['TE'].max():.6f}")
    print(f"  平均值: {df['TE'].mean():.6f}")
    print(f"  中位数: {df['TE'].median():.6f}")
    print(f"  是否有负值: {(df['TE'] < 0).any()}")
    print(f"  是否有零值: {(df['TE'] == 0).any()}")
    
    # 对TE取log10
    df['TE'] = np.log10(df['TE'])
    
    print(f"\nTE列的统计信息（转换后，log10）:")
    print(f"  最小值: {df['TE'].min():.6f}")
    print(f"  最大值: {df['TE'].max():.6f}")
    print(f"  平均值: {df['TE'].mean():.6f}")
    print(f"  中位数: {df['TE'].median():.6f}")
    print(f"  标准差: {df['TE'].std():.6f}")
    
    # ========================================
    # 步骤2: 修改TISSUE列的名称
    # ========================================
    print("\n" + "="*70)
    print("步骤2: 修改TISSUE列的组织名称")
    print("="*70)
    
    # 显示原始组织分布
    print("\n原始TISSUE分布:")
    tissue_counts = df['TISSUE'].value_counts()
    for tissue, count in tissue_counts.items():
        print(f"  {tissue}: {count}")
    
    # 替换组织名称
    df['TISSUE'] = df['TISSUE'].str.replace('Prophase.I.pollen', 'Prophase-I-pollen', regex=False)
    df['TISSUE'] = df['TISSUE'].str.replace('Tricellular.pollen', 'Tricellular-pollen', regex=False)
    
    print("\n修改后TISSUE分布:")
    tissue_counts = df['TISSUE'].value_counts()
    for tissue, count in tissue_counts.items():
        print(f"  {tissue}: {count}")
    
    # 验证是否还有旧名称
    old_names = ['Prophase.I.pollen', 'Tricellular.pollen']
    has_old = df['TISSUE'].isin(old_names).any()
    if has_old:
        print("\n⚠ 警告: 仍然存在旧的组织名称!")
    else:
        print("\n✓ 确认: 所有旧名称已成功替换")
    
    # 显示修改后的数据样例
    print("\n" + "="*70)
    print("修改后的数据样例（前5行）:")
    print("="*70)
    print(df.head(5)[['GeneID', 'TE', 'TISSUE']])
    
    # ========================================
    # 保存结果
    # ========================================
    print("\n" + "="*70)
    print("保存文件...")
    print("="*70)
    
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"✓ 文件已保存: {output_file}")
    print(f"✓ 处理完成！共处理 {len(df)} 行数据")
    
    return df


if __name__ == "__main__":
    # 使用示例
    input_file = '9tissue_structure_te_hc_deseq2_tp_split.csv'
    
    # 处理文件（覆盖原文件）
    df = process_te_data(input_file)
    
    # 或者保存到新文件
    # df = process_te_data(input_file, output_file='9tissue_structure_te_hc_deseq2_tp_split_log10.csv')
