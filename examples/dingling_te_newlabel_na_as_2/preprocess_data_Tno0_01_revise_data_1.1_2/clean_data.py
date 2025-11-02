#!/usr/bin/env python3
"""
剔除合并后文件中不含0或1的样本
- 只保留至少有一个tissue的label为0或1的样本
- 剔除所有tissue的label都是2或NaN的样本
"""

import pandas as pd
import os


def filter_samples(input_file, output_file):
    """
    过滤样本，只保留至少有一个tissue的label为0或1的样本
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    
    Returns:
        tuple: (原始样本数, 过滤后样本数)
    """
    print(f"\n正在处理文件: {os.path.basename(input_file)}")
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"  原始数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    
    # 获取所有label列
    label_cols = [col for col in df.columns if col.endswith('_TE_label')]
    print(f"  找到 {len(label_cols)} 个label列")
    
    # 统计原始数据的label分布
    print(f"\n  原始数据label分布:")
    for col in label_cols:
        value_counts = df[col].value_counts(dropna=False)
        print(f"    {col}: {value_counts.to_dict()}")
    
    # 创建一个布尔掩码：对于每一行，检查是否至少有一个label列的值为0或1
    # 方法：对于每一行，检查label列中是否存在0或1
    mask = df[label_cols].isin([0, 1, 0.0, 1.0]).any(axis=1)
    
    # 应用过滤
    df_filtered = df[mask].copy()
    
    original_count = len(df)
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    print(f"\n  过滤结果:")
    print(f"    原始样本数: {original_count}")
    print(f"    保留样本数: {filtered_count}")
    print(f"    剔除样本数: {removed_count}")
    print(f"    保留比例: {filtered_count/original_count*100:.2f}%")
    
    # 统计过滤后的label分布
    print(f"\n  过滤后label分布:")
    for col in label_cols:
        value_counts = df_filtered[col].value_counts(dropna=False)
        print(f"    {col}: {value_counts.to_dict()}")
    
    # 保存过滤后的数据
    df_filtered.to_csv(output_file, index=False)
    print(f"\n  已保存到: {output_file}")
    
    return original_count, filtered_count


def main():
    """主函数"""
    # 设置基础目录
    base_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_01_revise_data_1.1_2"
    
    print("=" * 80)
    print("剔除合并后文件中不含0或1的样本")
    print("=" * 80)
    
    # 设置输入和输出文件路径
    input_file = os.path.join(base_dir, "merged_tissue_labels.csv")
    output_file = os.path.join(base_dir, "merged_tissue_labels_filtered.csv")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 过滤数据
    original_count, filtered_count = filter_samples(input_file, output_file)
    
    # 打印总结
    print(f"\n{'=' * 80}")
    print("处理完成 - 总结")
    print(f"{'=' * 80}")
    
    removed_count = original_count - filtered_count
    print(f"\n原始样本数: {original_count:,}")
    print(f"保留样本数: {filtered_count:,}")
    print(f"剔除样本数: {removed_count:,}")
    print(f"保留比例: {filtered_count/original_count*100:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("文件处理完成!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
