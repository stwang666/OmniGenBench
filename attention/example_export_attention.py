#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出Attention信息的示例脚本

展示如何使用export_attention.py中的函数
"""

from export_attention import export_attention_to_file

# 示例1: 从数据集导出，使用最后一层
if __name__ == "__main__":
    # 配置参数
    model_path = "/path/to/your/model"  # 替换为你的模型路径
    dataset_path = "/path/to/your/dataset"  # 替换为你的数据集路径
    output_file = "attention_export.atten"
    
    # 方式1: 使用最后一层，平均所有头
    print("=" * 60)
    print("示例1: 使用最后一层，平均所有头")
    print("=" * 60)
    export_attention_to_file(
        model_path=model_path,
        dataset_path=dataset_path,
        output_file="attention_last_layer.atten",
        layer_indices=-1,  # 最后一层
        head_aggregation="mean",  # 平均所有头
        position_aggregation="row_mean",  # 行平均（该位置关注其他位置的平均）
        batch_size=4,
        max_length=512,
    )
    
    # 方式2: 使用所有层平均，最大头聚合
    print("\n" + "=" * 60)
    print("示例2: 使用所有层平均，最大头聚合")
    print("=" * 60)
    export_attention_to_file(
        model_path=model_path,
        dataset_path=dataset_path,
        output_file="attention_all_layers_max.atten",
        layer_indices=None,  # 所有层
        head_aggregation="max",  # 最大头
        position_aggregation="col_mean",  # 列平均（其他位置关注该位置的平均）
        batch_size=4,
        max_length=512,
    )
    
    # 方式3: 使用多个指定层（如第0, 5, 10层）
    print("\n" + "=" * 60)
    print("示例3: 使用多个指定层")
    print("=" * 60)
    export_attention_to_file(
        model_path=model_path,
        dataset_path=dataset_path,
        output_file="attention_multiple_layers.atten",
        layer_indices=[0, 5, 10],  # 指定多个层
        head_aggregation="mean",
        position_aggregation="row_mean",
        batch_size=4,
        max_length=512,
    )
    
    # 方式4: 直接从序列列表导出（不依赖数据集）
    print("\n" + "=" * 60)
    print("示例4: 直接从序列列表导出")
    print("=" * 60)
    sequences = [
        "CTGTCAATCGCCGAATGGCACCCTGCCGCACGGCAGAGGATCCGCGTCCACAAAACCCAACCCCCACGCACCGCGCGCAGCCGTTTTACCCAGCATGGCAAGAGGCGCATCCCGAGCATCACTCGTCGACTGACGAGGACCTGAGGCGGAGGCCCTAGAGGGAGCAAAGCAAAAGGTGCTGGTGAGTGGTGACTATAGCTACCATGACAAGTTGAGAGAGAGAGGGAGAGACCCACACTAGGCTGTCGCACCAACGGGTCAGGAGGGAGGGAGAGAGAGAGAGAGACGGGGGTTTTAATTTGAAGGGCGAAGCCTGGTCACTCGTGTGTGTGCGTGTGTTTCCCCCCGTCGTCGCCACTCTCCCTCTTTCCCCTCTCTCCCTCGAGAGAGGAGAGGAGAGGGGGGCCAGCAAGCAGGCAAGAGCAGTGTTCCACCTCCACTTCCACGACCAGCATCGCCAGCGCACCACGAGCTAGCTAGGGCCGACGACAGGCGCCGCA",
        "CGAAGCCAAAGATGGCATGAATAGGCAGCGGGGCGCAGCCTTTTCACCATACGTACTTGCTCCCCGGCATGCACGGGCATACAAAACCCAACCCCTCCCGCCCGCCCGCACCCCCACGCACCACGCGCAGCCGTTTTACCCAGCATGGCAAGGGGCGCATCCCTCCCCTCTCTCATCGACTGACGAGGACTCAAAGCAAAAGGTGCTGGTGAGTGGTGACTATAGCTACCATGACAAGTTGAGAGACCCACACTAGGCTGTCGCACCAACGGGTCAGGAGGGAGAGAGAGGGATGGGGGTTTTAATTTGAAGGGTGAAGCCTGGTCACTCGTGTGTGTGTGAGTGCGCGTGTTTCCCCCCATCGTCACCAATCTCCATCTTTCCCCTCTCTCCCTCGAGAGAGGAGAGGAGAGGGGGGCCAGCAAGCAAGAGCAGTGTTCCTCCACGACCAGCATCGCCAGCGCACCACGAGCTAGCTAGGACCGACGACAAGCGCCGCA",
    ]
    labels = [1, 1]
    other_info = [
        {"gene": "TraesCS7A03G0567100.1"},
        {"gene": "TraesCS7B03G0393600.1"},
    ]
    
    export_attention_to_file(
        model_path=model_path,
        sequences=sequences,
        labels=labels,
        other_info=other_info,
        output_file="attention_from_sequences.atten",
        layer_indices=-1,
        head_aggregation="mean",
        position_aggregation="row_mean",
        batch_size=2,
        max_length=512,
    )
    
    print("\n✅ 所有示例完成！")

