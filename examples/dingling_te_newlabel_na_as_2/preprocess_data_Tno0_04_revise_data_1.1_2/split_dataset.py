#!/usr/bin/env python3
"""
数据集划分脚本
- 将剔除后的A和B合并作为训练集
- 将D划分为验证集和测试集
- 使用随机划分（random_state=42保证可复现）
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_filtered_data(base_dir, groups):
    """
    加载过滤后的数据
    
    Args:
        base_dir: 基础目录
        groups: 要加载的组列表
    
    Returns:
        dict: {group: dataframe}
    """
    data_dict = {}
    
    for group in groups:
        data_dir = os.path.join(base_dir, group)
        input_file = os.path.join(data_dir, f"merged_tissue_labels_{group}_filtered.csv")
        
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在: {input_file}")
            continue
        
        print(f"正在加载 {group} 组数据...")
        df = pd.read_csv(input_file)
        print(f"  {group} 组样本数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        data_dict[group] = df
    
    return data_dict


def merge_train_data(data_dict, train_groups):
    """
    合并训练集数据
    
    Args:
        data_dict: 数据字典
        train_groups: 要合并为训练集的组列表
    
    Returns:
        pd.DataFrame: 合并后的训练集
    """
    train_dfs = []
    
    for group in train_groups:
        if group in data_dict:
            train_dfs.append(data_dict[group])
            print(f"  添加 {group} 组到训练集，样本数: {len(data_dict[group])}")
    
    if not train_dfs:
        print("错误: 没有可用的训练集数据")
        return None
    
    # 合并所有训练集数据
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\n  合并后训练集样本数: {len(train_df)}")
    
    return train_df


def split_test_data(df, test_size=0.5, random_state=42):
    """
    将数据划分为验证集和测试集
    
    Args:
        df: 要划分的数据框
        test_size: 测试集占比（默认0.5，即验证集和测试集各占50%）
        random_state: 随机种子
    
    Returns:
        tuple: (验证集, 测试集)
    """
    valid_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"  验证集样本数: {len(valid_df)}")
    print(f"  测试集样本数: {len(test_df)}")
    
    return valid_df, test_df


def analyze_label_distribution(df, dataset_name):
    """
    分析数据集的label分布
    
    Args:
        df: 数据框
        dataset_name: 数据集名称
    """
    print(f"\n{dataset_name}的label分布:")
    
    # 获取所有label列
    label_cols = [col for col in df.columns if col.endswith('_TE_label')]
    
    for col in label_cols:
        value_counts = df[col].value_counts(dropna=False)
        total_valid = df[col].notna().sum()
        label_0 = (df[col] == 0.0).sum()
        label_1 = (df[col] == 1.0).sum()
        label_2 = (df[col] == 2.0).sum()
        label_nan = df[col].isna().sum()
        
        print(f"  {col}:")
        print(f"    有效值: {total_valid}, NaN: {label_nan}")
        print(f"    0: {label_0} ({label_0/len(df)*100:.2f}%)")
        print(f"    1: {label_1} ({label_1/len(df)*100:.2f}%)")
        print(f"    2: {label_2} ({label_2/len(df)*100:.2f}%)")


def main():
    """主函数"""
    # 设置基础目录
    base_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_04_revise_data_3.1_2"
    
    # 输出目录
    output_dir = base_dir
    
    print("=" * 80)
    print("数据集划分")
    print("=" * 80)
    print("\n策略:")
    print("  - 训练集: A + B")
    print("  - 验证集: D的50%")
    print("  - 测试集: D的50%")
    print("=" * 80)
    
    # 加载所有过滤后的数据
    print("\n步骤1: 加载过滤后的数据")
    print("-" * 80)
    data_dict = load_filtered_data(base_dir, ['A', 'B', 'D'])
    
    if 'A' not in data_dict or 'B' not in data_dict or 'D' not in data_dict:
        print("\n错误: 缺少必要的数据文件!")
        return
    
    # 合并A和B作为训练集
    print("\n步骤2: 合并A和B作为训练集")
    print("-" * 80)
    train_df = merge_train_data(data_dict, ['A', 'B'])
    
    if train_df is None:
        return
    
    # 将D划分为验证集和测试集
    print("\n步骤3: 将D划分为验证集和测试集")
    print("-" * 80)
    print(f"  D组样本数: {len(data_dict['D'])}")
    valid_df, test_df = split_test_data(data_dict['D'], test_size=0.5, random_state=42)
    
    # 保存数据集
    print("\n步骤4: 保存数据集")
    print("-" * 80)
    
    train_file = os.path.join(output_dir, "train.csv")
    valid_file = os.path.join(output_dir, "valid.csv")
    test_file = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    print(f"  训练集已保存: {train_file}")
    print(f"    样本数: {len(train_df)}")
    
    valid_df.to_csv(valid_file, index=False)
    print(f"  验证集已保存: {valid_file}")
    print(f"    样本数: {len(valid_df)}")
    
    test_df.to_csv(test_file, index=False)
    print(f"  测试集已保存: {test_file}")
    print(f"    样本数: {len(test_df)}")
    
    # 分析label分布
    print("\n步骤5: 分析数据集label分布")
    print("-" * 80)
    
    analyze_label_distribution(train_df, "训练集")
    analyze_label_distribution(valid_df, "验证集")
    analyze_label_distribution(test_df, "测试集")
    
    # 总结
    print("\n" + "=" * 80)
    print("数据集划分完成!")
    print("=" * 80)
    
    total_samples = len(train_df) + len(valid_df) + len(test_df)
    print(f"\n总样本数: {total_samples:,}")
    print(f"  训练集: {len(train_df):,} ({len(train_df)/total_samples*100:.2f}%)")
    print(f"  验证集: {len(valid_df):,} ({len(valid_df)/total_samples*100:.2f}%)")
    print(f"  测试集: {len(test_df):,} ({len(test_df)/total_samples*100:.2f}%)")
    
    print("\n数据来源:")
    print(f"  训练集: A组 + B组")
    print(f"    A组: {len(data_dict['A']):,} 样本")
    print(f"    B组: {len(data_dict['B']):,} 样本")
    print(f"  验证集: D组的50%")
    print(f"  测试集: D组的50%")
    print(f"    D组总计: {len(data_dict['D']):,} 样本")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

