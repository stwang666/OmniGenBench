#!/usr/bin/env python3
"""
将log10(TE)数据划分为三类 (0, 1, 2)
提供三种划分方案：
1. mean ± std
2. mean ± 0.5*std  
3. 等分位数 (33.3%/66.7%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_distribution(te_values):
    """分析log10(TE)的数据分布"""
    mean = np.mean(te_values)
    std = np.std(te_values)
    median = np.median(te_values)
    
    print("="*70)
    print("log10(TE) 数据分布统计")
    print("="*70)
    print(f"\n样本数: {len(te_values)}")
    print(f"均值: {mean:.6f}")
    print(f"标准差: {std:.6f}")
    print(f"中位数: {median:.6f}")
    print(f"最小值: {np.min(te_values):.6f}")
    print(f"最大值: {np.max(te_values):.6f}")
    print(f"第25分位数: {np.percentile(te_values, 25):.6f}")
    print(f"第75分位数: {np.percentile(te_values, 75):.6f}")
    
    return mean, std, median


def method1_mean_std(te_values):
    """方案1: mean ± std"""
    mean = np.mean(te_values)
    std = np.std(te_values)
    
    threshold_low = mean - std
    threshold_high = mean + std
    
    labels = np.zeros(len(te_values), dtype=int)
    labels[te_values >= threshold_low] = 1
    labels[te_values >= threshold_high] = 2
    
    return labels, threshold_low, threshold_high


def method2_mean_half_std(te_values):
    """方案2: mean ± 0.5*std"""
    mean = np.mean(te_values)
    std = np.std(te_values)
    
    threshold_low = mean - 0.5 * std
    threshold_high = mean + 0.5 * std
    
    labels = np.zeros(len(te_values), dtype=int)
    labels[te_values >= threshold_low] = 1
    labels[te_values >= threshold_high] = 2
    
    return labels, threshold_low, threshold_high


def method3_quantile(te_values):
    """方案3: 等分位数 (33.3%/66.7%)"""
    threshold_low = np.percentile(te_values, 100/3)
    threshold_high = np.percentile(te_values, 200/3)
    
    labels = np.zeros(len(te_values), dtype=int)
    labels[te_values >= threshold_low] = 1
    labels[te_values >= threshold_high] = 2
    
    return labels, threshold_low, threshold_high


def print_classification_stats(labels, te_values, threshold_low, threshold_high, method_name):
    """打印分类统计信息"""
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    count_2 = np.sum(labels == 2)
    total = len(labels)
    
    print(f"\n{method_name}:")
    print(f"  阈值: [{threshold_low:.6f}, {threshold_high:.6f}]")
    print(f"  类别0: {count_0:5d} ({count_0/total*100:.1f}%)  [log10(TE) < {threshold_low:.3f}]")
    print(f"  类别1: {count_1:5d} ({count_1/total*100:.1f}%)  [{threshold_low:.3f} ≤ log10(TE) < {threshold_high:.3f}]")
    print(f"  类别2: {count_2:5d} ({count_2/total*100:.1f}%)  [log10(TE) ≥ {threshold_high:.3f}]")
    
    # 计算均衡度（方差越小越均衡）
    variance = np.var([count_0, count_1, count_2])
    print(f"  均衡度（方差）: {variance:.0f}")


def split_triclass(input_file, output_file=None, method='quantile', save_plot=True):
    """
    将数据划分为三类并保存
    
    Parameters:
    -----------
    input_file : str
        输入CSV文件路径（已经过log10转换的数据）
    output_file : str, optional
        输出CSV文件路径
    method : str
        划分方法: 'mean_std', 'mean_half_std', 或 'quantile'
    save_plot : bool
        是否保存可视化图表
    """
    # 读取数据
    print("读取数据...")
    df = pd.read_csv(input_file)
    print(f"数据维度: {df.shape}")
    
    # 分析分布
    mean, std, median = analyze_distribution(df['TE'].values)
    
    # 应用不同的分类方法
    print("\n" + "="*70)
    print("三类划分方案比较")
    print("="*70)
    
    methods = {
        'mean_std': method1_mean_std,
        'mean_half_std': method2_mean_half_std,
        'quantile': method3_quantile
    }
    
    method_names = {
        'mean_std': 'mean ± std',
        'mean_half_std': 'mean ± 0.5*std',
        'quantile': ' 等分位数'
    }
    
    # 生成所有方案的标签用于比较
    all_results = {}
    for method_key, method_func in methods.items():
        labels, threshold_low, threshold_high = method_func(df['TE'].values)
        all_results[method_key] = {
            'labels': labels,
            'threshold_low': threshold_low,
            'threshold_high': threshold_high
        }
        print_classification_stats(
            labels, df['TE'].values, 
            threshold_low, threshold_high,
            method_names[method_key]
        )
    
    # 使用选定的方法
    print("\n" + "="*70)
    print(f"使用方法: {method_names[method]}")
    print("="*70)
    
    result = all_results[method]
    df['label'] = result['labels']
    
    # 保存结果
    if output_file is None:
        output_file = input_file.replace('.csv', f'_triclass_{method}.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\n✓ 分类结果已保存到: {output_file}")
    
    # 可视化
    if save_plot:
        plot_distribution(df, result['threshold_low'], result['threshold_high'], method_names[method])
    
    return df


def plot_distribution(df, threshold_low, threshold_high, method_name):
    """绘制数据分布和分类结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图
    ax1 = axes[0, 0]
    ax1.hist(df['TE'], bins=100, alpha=0.7, edgecolor='black')
    ax1.axvline(threshold_low, color='r', linestyle='--', linewidth=2, label=f'Low: {threshold_low:.3f}')
    ax1.axvline(threshold_high, color='r', linestyle='--', linewidth=2, label=f'High: {threshold_high:.3f}')
    ax1.set_xlabel('log10(TE)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of log10(TE)\n{method_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 按类别着色的直方图
    ax2 = axes[0, 1]
    for label in [0, 1, 2]:
        data = df[df['label'] == label]['TE']
        ax2.hist(data, bins=50, alpha=0.6, label=f'Class {label} (n={len(data)})')
    ax2.set_xlabel('log10(TE)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax3 = axes[1, 0]
    data_by_label = [df[df['label'] == i]['TE'].values for i in [0, 1, 2]]
    ax3.boxplot(data_by_label, labels=['Class 0', 'Class 1', 'Class 2'])
    ax3.set_ylabel('log10(TE)')
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
    plot_file = 'triclass_distribution.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化图表已保存: {plot_file}")
    plt.close()


def main():
    """主函数"""
    # 输入文件
    input_file = '9tissue_structure_te_hc_deseq2_tp_split_log10.csv'
    
    print("="*70)
    print("三类划分脚本")
    print("="*70)
    print("\n可用方法:")
    print("  1. 'mean_std'      - 使用 mean ± std")
    print("  2. 'mean_half_std' - 使用 mean ± 0.5*std")
    print("  3. 'quantile'      - 使用等分位数 (推荐)")
    
    # 使用方案2（mean ± 0.5*std）作为默认推荐
    method = 'mean_std'
    
    print(f"\n使用方法: {method}")
    
    # 执行分类
    df = split_triclass(
        input_file=input_file,
        output_file=None,
        method=method,
        save_plot=True
    )
    
    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
