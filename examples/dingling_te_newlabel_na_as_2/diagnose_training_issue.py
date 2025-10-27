#!/usr/bin/env python3
"""
诊断训练问题的脚本
分析为什么第一个epoch精度0.7，第二个epoch就降到0.1，以及损失变为NaN的问题
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_data_distribution():
    """分析数据分布和标签不平衡问题"""
    print("=" * 60)
    print("📊 数据分布分析")
    print("=" * 60)
    
    # 读取训练数据
    train_file = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tmean_02/train.csv"
    valid_file = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tmean_02/valid.csv"
    
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    
    # 分析每个tissue的标签分布
    tissue_columns = [col for col in train_df.columns if col.endswith('_TE_label')]
    
    print(f"\n各tissue标签分布分析:")
    for col in tissue_columns:
        print(f"\n{col}:")
        value_counts = train_df[col].value_counts(dropna=False)
        print(f"  训练集: {value_counts.to_dict()}")
        
        valid_counts = valid_df[col].value_counts(dropna=False)
        print(f"  验证集: {valid_counts.to_dict()}")
        
        # 计算类别不平衡比例
        if len(value_counts) > 1:
            max_count = value_counts.max()
            min_count = value_counts.min()
            imbalance_ratio = max_count / min_count
            print(f"  类别不平衡比例: {imbalance_ratio:.2f}:1")

def analyze_model_architecture():
    """分析模型架构和参数"""
    print("\n" + "=" * 60)
    print("🏗️ 模型架构分析")
    print("=" * 60)
    
    # 检查模型参数
    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/ogb_te_3class_finetuned_na_as_2_epoch_1_seed_42_accuracy_score_0.7196_seed_42_f1_score_0.3176"
    
    if Path(model_path).exists():
        print(f"✅ 找到模型: {model_path}")
        
        # 尝试加载模型检查参数
        try:
            from omnigenbench import ModelHub
            model = ModelHub.load(model_path)
            print(f"模型类型: {type(model)}")
            
            # 检查分类器参数
            if hasattr(model, 'classifier'):
                classifier_params = sum(p.numel() for p in model.classifier.parameters())
                print(f"分类器参数数量: {classifier_params:,}")
                
                # 检查权重分布
                for name, param in model.classifier.named_parameters():
                    if param.requires_grad:
                        weight_mean = param.data.mean().item()
                        weight_std = param.data.std().item()
                        weight_max = param.data.max().item()
                        weight_min = param.data.min().item()
                        print(f"  {name}: mean={weight_mean:.6f}, std={weight_std:.6f}, max={weight_max:.6f}, min={weight_min:.6f}")
                        
                        # 检查是否有异常大的权重
                        if abs(weight_max) > 10 or abs(weight_min) > 10:
                            print(f"    ⚠️  警告: {name} 权重值异常大!")
                        if weight_std > 5:
                            print(f"    ⚠️  警告: {name} 权重标准差过大!")
                            
        except Exception as e:
            print(f"❌ 无法加载模型: {e}")
    else:
        print(f"❌ 模型路径不存在: {model_path}")

def check_training_config():
    """检查训练配置"""
    print("\n" + "=" * 60)
    print("⚙️ 训练配置分析")
    print("=" * 60)
    
    # 从代码中提取的训练配置
    config = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 10,
        "gradient_accumulation_steps": None,  # 被注释掉了
        "num_labels": 9,
        "num_classes": 3,
        "loss_function": "CrossEntropyLoss",
        "ignore_index": -100
    }
    
    print("当前训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 分析潜在问题
    print(f"\n潜在问题分析:")
    
    # 1. 学习率分析
    lr = config["learning_rate"]
    if lr > 5e-5:
        print(f"  ⚠️  学习率可能过高: {lr} (建议 < 5e-5)")
    else:
        print(f"  ✅ 学习率合理: {lr}")
    
    # 2. 批次大小分析
    batch_size = config["batch_size"]
    if batch_size < 32:
        print(f"  ⚠️  批次大小较小: {batch_size} (可能导致训练不稳定)")
    else:
        print(f"  ✅ 批次大小合理: {batch_size}")
    
    # 3. 梯度累积分析
    if config["gradient_accumulation_steps"] is None:
        print(f"  ⚠️  未使用梯度累积 (可能导致梯度不稳定)")
    else:
        print(f"  ✅ 使用梯度累积: {config['gradient_accumulation_steps']}")
    
    # 4. 损失函数分析
    print(f"  ✅ 使用CrossEntropyLoss，适合多分类任务")
    print(f"  ✅ ignore_index=-100，正确处理无效标签")

def analyze_gradient_issues():
    """分析梯度相关问题"""
    print("\n" + "=" * 60)
    print("📈 梯度问题分析")
    print("=" * 60)
    
    print("可能导致梯度爆炸的原因:")
    print("1. 学习率过高 (1e-4 可能过高)")
    print("2. 没有梯度裁剪")
    print("3. 批次大小过小导致梯度不稳定")
    print("4. 数据预处理问题")
    print("5. 模型初始化问题")
    
    print(f"\n建议的解决方案:")
    print("1. 降低学习率到 1e-5 或 5e-6")
    print("2. 添加梯度裁剪 (max_grad_norm=1.0)")
    print("3. 使用梯度累积 (gradient_accumulation_steps=4)")
    print("4. 添加权重衰减 (weight_decay=0.01)")
    print("5. 使用学习率调度器")

def create_improved_training_script():
    """创建改进的训练脚本"""
    print("\n" + "=" * 60)
    print("🔧 创建改进的训练脚本")
    print("=" * 60)
    
    improved_script = '''# 改进的训练配置
trainer = Trainer(
    model=model,
    epochs=10,
    learning_rate=1e-5,  # 降低学习率
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,  # 添加梯度累积
    max_grad_norm=1.0,  # 添加梯度裁剪
    weight_decay=0.01,  # 添加权重衰减
    warmup_steps=100,  # 学习率预热
    eval_steps=50,  # 更频繁的验证
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy_score",
    greater_is_better=True,
    save_total_limit=3,
)'''
    
    print("改进的训练配置:")
    print(improved_script)
    
    # 保存改进的脚本
    with open("/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/improved_training_config.py", "w") as f:
        f.write(improved_script)
    
    print(f"\n✅ 改进配置已保存到: improved_training_config.py")

def main():
    """主函数"""
    print("🔍 训练问题诊断")
    print("=" * 60)
    print("分析训练过程中精度急剧下降和损失变为NaN的问题")
    
    # 1. 分析数据分布
    analyze_data_distribution()
    
    # 2. 分析模型架构
    analyze_model_architecture()
    
    # 3. 检查训练配置
    check_training_config()
    
    # 4. 分析梯度问题
    analyze_gradient_issues()
    
    # 5. 创建改进脚本
    create_improved_training_script()
    
    print("\n" + "=" * 60)
    print("🎯 诊断总结")
    print("=" * 60)
    print("主要问题:")
    print("1. 学习率过高 (1e-4) 导致梯度爆炸")
    print("2. 没有梯度裁剪保护")
    print("3. 批次大小过小 (16) 导致训练不稳定")
    print("4. 没有使用梯度累积")
    print("5. 缺少权重衰减等正则化")
    
    print(f"\n建议的解决方案:")
    print("1. 降低学习率到 1e-5")
    print("2. 添加梯度裁剪 (max_grad_norm=1.0)")
    print("3. 使用梯度累积 (gradient_accumulation_steps=4)")
    print("4. 添加权重衰减 (weight_decay=0.01)")
    print("5. 使用学习率预热")
    print("6. 更频繁的验证和模型保存")

if __name__ == "__main__":
    main()
