#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从数据文件中提取Tricellular-pollen数据，并按照三分位点划分标签为0, 1, 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def extract_tricellular_pollen_and_label(input_file, output_file=None):
    """
    提取Tricellular-pollen数据并按三分位点划分标签
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str, optional
        输出CSV文件路径
    """
    # 读取数据
    print("="*70)
    print("提取Tricellular-pollen数据并划分标签")
    print("="*70)
    print(f"\n读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据维度: {df.shape}")
    
    # 筛选Tricellular-pollen数据
    print("\n筛选Tricellular-pollen数据...")
    df_tricellular = df[df['tissue'] == 'Tricellular-pollen'].copy()
    print(f"Tricellular-pollen数据维度: {df_tricellular.shape}")
    
    if len(df_tricellular) == 0:
        print("错误: 未找到Tricellular-pollen数据！")
        return None
    
    # 分析TE值分布
    te_values = df_tricellular['TE'].values
    mean = np.mean(te_values)
    std = np.std(te_values)
    median = np.median(te_values)
    
    print("\n" + "="*70)
    print("Tricellular-pollen的TE值分布统计")
    print("="*70)
    print(f"样本数: {len(te_values)}")
    print(f"均值: {mean:.6f}")
    print(f"标准差: {std:.6f}")
    print(f"中位数: {median:.6f}")
    print(f"最小值: {np.min(te_values):.6f}")
    print(f"最大值: {np.max(te_values):.6f}")
    print(f"第25分位数: {np.percentile(te_values, 25):.6f}")
    print(f"第75分位数: {np.percentile(te_values, 75):.6f}")
    
    # 计算三分位点 (33.33% 和 66.67%)
    threshold_low = np.percentile(te_values, 100/3)   # 33.33%分位点
    threshold_high = np.percentile(te_values, 200/3)  # 66.67%分位点
    
    print(f"\n三分位点阈值:")
    print(f"  低阈值 (33.33%): {threshold_low:.6f}")
    print(f"  高阈值 (66.67%): {threshold_high:.6f}")
    
    # 根据三分位点划分标签
    labels = np.zeros(len(te_values), dtype=int)
    labels[te_values >= threshold_low] = 1
    labels[te_values >= threshold_high] = 2
    
    # 添加标签列
    df_tricellular['label'] = labels
    
    # 统计各类别数量
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    count_2 = np.sum(labels == 2)
    total = len(labels)
    
    print("\n" + "="*70)
    print("标签划分结果")
    print("="*70)
    print(f"  类别0: {count_0:5d} ({count_0/total*100:.1f}%)  [TE < {threshold_low:.6f}]")
    print(f"  类别1: {count_1:5d} ({count_1/total*100:.1f}%)  [{threshold_low:.6f} ≤ TE < {threshold_high:.6f}]")
    print(f"  类别2: {count_2:5d} ({count_2/total*100:.1f}%)  [TE ≥ {threshold_high:.6f}]")
    
    # 保存结果
    if output_file is None:
        output_file = input_file.replace('.csv', '_Tricellular-pollen_triclass.csv')
        # 如果输入文件在另一个目录，确保输出到当前目录
        if Path(input_file).parent != Path('.'):
            output_file = Path(input_file).parent / Path(output_file).name
    
    df_tricellular.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 可视化
    plot_distribution(df_tricellular, threshold_low, threshold_high, output_file)
    
    return df_tricellular


def plot_distribution(df, threshold_low, threshold_high, output_file):
    """绘制数据分布和分类结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图
    ax1 = axes[0, 0]
    ax1.hist(df['TE'], bins=100, alpha=0.7, edgecolor='black')
    ax1.axvline(threshold_low, color='r', linestyle='--', linewidth=2, label=f'Low: {threshold_low:.3f}')
    ax1.axvline(threshold_high, color='r', linestyle='--', linewidth=2, label=f'High: {threshold_high:.3f}')
    ax1.set_xlabel('TE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of TE (Tricellular-pollen)\nTertile Classification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按类别着色的直方图
    ax2 = axes[0, 1]
    for label in [0, 1, 2]:
        data = df[df['label'] == label]['TE']
        ax2.hist(data, bins=50, alpha=0.6, label=f'Class {label} (n={len(data)})')
    ax2.set_xlabel('TE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax3 = axes[1, 0]
    data_by_label = [df[df['label'] == i]['TE'].values for i in [0, 1, 2]]
    bp = ax3.boxplot(data_by_label)
    ax3.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    ax3.set_ylabel('TE')
    ax3.set_title('Box Plot by Class')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 类别比例饼图
    ax4 = axes[1, 1]
    label_counts = df['label'].value_counts().sort_index()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax4.pie(label_counts, labels=[f'Class {i}\n({count} samples)\n{count/len(df)*100:.1f}%' 
                                    for i, count in enumerate(label_counts)],
            autopct='', colors=colors, startangle=90)
    ax4.set_title('Class Distribution')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = Path(output_file).parent / 'tricellular_pollen_triclass_distribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {plot_file}")
    plt.close()


def main():
    """主函数"""
    # 输入文件路径
    input_file = '9tissue_structure_te_hc_deseq2_tp_split_log10.csv'
    
    # 如果文件不在当前目录，尝试使用完整路径
    if not Path(input_file).exists():
        input_file = Path(__file__).parent / input_file
    
    if not Path(input_file).exists():
        print(f"错误: 找不到输入文件: {input_file}")
        print("请确保文件路径正确")
        return
    
    # 执行提取和标签划分
    df_result = extract_tricellular_pollen_and_label(
        input_file=str(input_file),
        output_file=None
    )
    
    if df_result is not None:
        print("\n" + "="*70)
        print("处理完成！")
        print("="*70)


if __name__ == "__main__":
    main()
