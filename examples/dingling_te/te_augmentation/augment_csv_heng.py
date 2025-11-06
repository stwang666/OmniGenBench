#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对CSV文件中的序列进行数据增强
基于OmniGenBench的OmniModelForAugmentation进行序列增强
使用augment_sequences函数进行批量增强
"""

import pandas as pd
import json
from tqdm import tqdm
from omnigenbench import OmniModelForAugmentation


def augment_csv_file(
    input_csv_path,
    output_csv_path,
    sequence_column="sequence",
    model_name="yangheng/OmniGenome-186M",
    noise_ratio=0.2,
    instance_num=3,
    max_length=512,
    batch_size=8,
    keep_original=True,
    min_sequence_length=20,
):
    """
    对CSV文件中的序列进行数据增强（使用augment_sequences批量增强）

    参数:
        input_csv_path (str): 输入CSV文件路径
        output_csv_path (str): 输出CSV文件路径
        sequence_column (str): 序列列名，默认为"sequence"
        model_name (str): 模型名称，默认为"yangheng/OmniGenome-52M"
        noise_ratio (float): 掩码比例，默认为0.2（20%的token将被掩码）
        instance_num (int): 每个序列生成的增强实例数，默认为3
        max_length (int): 最大序列长度，默认为512
        batch_size (int): 批处理大小，默认为8
        keep_original (bool): 是否保留原始数据，默认为True
        min_sequence_length (int): 最小序列长度，低于此长度的序列不进行增强，默认为20
    """
    print("=" * 60)
    print(" 基因组数据增强程序 (批量增强)")
    print("=" * 60)
    print(f" 输入文件: {input_csv_path}")
    print(f" 输出文件: {output_csv_path}")
    print(f" 模型: {model_name}")
    print(f" 增强参数:")
    print(f"   - 掩码比例: {noise_ratio:.1%}")
    print(f"   - 每个序列生成实例数: {instance_num}")
    print(f"   - 最大序列长度: {max_length}")
    print(f"   - 最小序列长度: {min_sequence_length}")
    print(f"   - 批处理大小: {batch_size}")
    print(f"   - 保留原始数据: {keep_original}")
    print("=" * 60)

    # 1. 加载CSV文件
    print("\n 步骤1: 加载CSV文件...")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"成功加载 {len(df)} 条记录")
        print(f"列名: {list(df.columns)}")

        if sequence_column not in df.columns:
            raise ValueError(f"错误: CSV文件中未找到序列列 '{sequence_column}'")

    except Exception as e:
        print(f"加载CSV文件失败: {str(e)}")
        raise

    # 2. 初始化增强模型
    print("\n步骤2: 初始化增强模型...")
    try:
        augmentation_model = OmniModelForAugmentation(
            config_or_model=model_name,
            noise_ratio=noise_ratio,
            max_length=max_length,
            instance_num=instance_num,
            batch_size=batch_size,
        )
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        raise

     # print(f"DEBUG: 模型内部的 instance_num 被设置为: {augmentation_model.instance_num}")
    # 3. 提取序列数据并过滤短序列
    print("\n步骤3: 开始批量序列增强...")
    
    # 准备输出数据列表
    augmented_rows = []
    
    # 分离需要增强的序列和不需要增强的序列
    sequences_to_augment = []
    indices_to_augment = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        sequence = row[sequence_column]
        seq_length = len(str(sequence))
        
        # 如果保留原始数据，添加原始行
        if keep_original:
            augmented_rows.append(row.to_dict())
        
        # 检查序列长度
        if seq_length < min_sequence_length:
            skipped_count += 1
            # 短序列不进行增强，如果不保留原始数据，这里也需要添加
            if not keep_original:
                augmented_rows.append(row.to_dict())
        else:
            sequences_to_augment.append(sequence)
            indices_to_augment.append(idx)
    
    print(f" 总序列数: {len(df)}")
    print(f" 需要增强的序列数: {len(sequences_to_augment)}")
    print(f" 跳过的短序列数 (长度 < {min_sequence_length}): {skipped_count}")

    # 4. 使用augment_sequences进行批量增强
    if len(sequences_to_augment) > 0:
        print(f" 处理 {len(sequences_to_augment)} 条序列，每条生成 {instance_num} 个增强实例...")
        print(f" 使用augment_sequences批量处理，批处理大小: {batch_size}")

        try:
            # augment_sequences 会自动按batch_size批量处理所有序列
            # 返回: List[str], 一个扁平列表，长度为 len(sequences) * instance_num
            # 格式: [seq0_aug1, seq0_aug2, ..., seq1_aug1, seq1_aug2, ..., ...]
            all_augmented_sequences = augmentation_model.augment_sequences(sequences_to_augment)

            print(f" 批量增强完成，开始处理结果...")
            print(f" 返回的增强序列总数: {len(all_augmented_sequences)} (应该是 {len(sequences_to_augment)} * {instance_num} = {len(sequences_to_augment) * instance_num})")

            # 使用tqdm显示处理进度
            for list_idx, original_idx in tqdm(enumerate(indices_to_augment), total=len(indices_to_augment), desc="处理增强结果", unit="seq"):
                row = df.iloc[original_idx]
                
                # 从扁平列表中提取当前序列对应的所有增强结果
                # 每个序列对应 instance_num 个增强结果，从 list_idx * instance_num 位置开始
                start_idx = list_idx * instance_num
                end_idx = start_idx + instance_num
                augmented_sequences = all_augmented_sequences[start_idx:end_idx]

                # 为每个增强序列创建新行，并去除每个碱基之间的空格
                for aug_seq in augmented_sequences:
                    aug_seq_nospace = aug_seq.replace(" ", "") if isinstance(aug_seq, str) else aug_seq
                    new_row = row.copy()
                    # 更新序列列为增强后的序列（去除空格）
                    new_row[sequence_column] = aug_seq_nospace
                    # 可以修改ID以标识这是增强数据
                    if "ID" in new_row:
                        original_id = str(new_row["ID"])
                        # 为增强数据添加后缀标识
                        new_row["ID"] = f"{original_id}_aug"
                    augmented_rows.append(new_row)

        except Exception as e:
            print(f"\n 批量增强失败: {str(e)}")
            raise
    else:
        print(" 没有符合条件的序列需要增强")

    # 5. 保存增强后的数据
    print("\n 步骤4: 保存增强后的数据...")
    try:
        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.to_csv(output_csv_path, index=False)

        original_count = len(df)
        augmented_count = len(augmented_df)
        new_count = augmented_count - (original_count if keep_original else 0)

        print(f" 数据保存成功!")
        print(f" 统计信息:")
        print(f"   - 原始记录数: {original_count}")
        print(f"   - 跳过的短序列数: {skipped_count}")
        if keep_original:
            print(f"   - 增强记录数: {new_count}")
            print(f"   - 总记录数: {augmented_count}")
            print(f"   - 数据集扩展: {augmented_count / original_count:.1f}x")
        else:
            print(f"   - 总记录数: {augmented_count}")
            print(f"   - 数据集扩展: {augmented_count / original_count:.1f}x")
        print(f" 输出文件: {output_csv_path}")

    except Exception as e:
        print(f" 保存数据失败: {str(e)}")
        raise

    print("\n" + "=" * 60)
    print(" 数据增强完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    config = {
        "input_csv_path": "/home/sw1136/OmniGenBench/examples/dingling_te/te_augmentation/remove_duplicated_data_from_train/train_original_some_seq_too_short.csv",
        "output_csv_path": "train_augmented_5instances_5noise_20length.csv",
        "sequence_column": "seq",
        "model_name": "yangheng/OmniGenome-186M",  # 可以使用 "yangheng/OmniGenome-186M" 获得更好的效果
        "noise_ratio": 0.5,          # 20%的token将被掩码
        "instance_num": 5,           # 每个序列生成3个增强实例
       # "max_length": 512,            # 最大序列长度
        "batch_size": 32,              # 批处理大小
        "keep_original": True,        # 保留原始数据
        "min_sequence_length": 20,    # 最小序列长度
    }

    # 运行增强程序
    augment_csv_file(**config)
