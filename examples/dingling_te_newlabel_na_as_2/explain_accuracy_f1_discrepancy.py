#!/usr/bin/env python3
"""
解释为什么Accuracy高但F1分数低的问题
分析NA类标签占比70-80%对指标的影响
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

def explain_metrics():
    """解释Accuracy和F1分数的含义和差异"""
    print("=" * 80)
    print("📊 Accuracy vs F1 Score 详细解释")
    print("=" * 80)
    
    print("\n🎯 指标定义:")
    print("1. Accuracy (准确率) = 正确预测数 / 总预测数")
    print("2. F1 Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print("   - Precision (精确率) = TP / (TP + FP)")
    print("   - Recall (召回率) = TP / (TP + FN)")
    
    print("\n📈 为什么会出现Accuracy高但F1低的情况？")
    print("=" * 60)
    
    # 模拟数据不平衡的情况
    print("\n🔍 模拟场景分析:")
    print("假设有1000个样本，类别分布如下:")
    print("- 类别0 (标签0): 100个样本 (10%)")
    print("- 类别1 (标签1): 100个样本 (10%)") 
    print("- 类别2 (NA): 800个样本 (80%)")
    
    # 模拟模型预测结果
    print("\n📊 模型预测结果模拟:")
    print("情况1: 模型完全偏向NA类（最坏情况）")
    
    # 真实标签
    y_true = [0]*100 + [1]*100 + [2]*800
    # 模型预测（全部预测为NA类）
    y_pred_na_bias = [2]*1000
    
    acc_na_bias = accuracy_score(y_true, y_pred_na_bias)
    f1_na_bias = f1_score(y_true, y_pred_na_bias, average='macro')
    
    print(f"  - Accuracy: {acc_na_bias:.3f} (80%正确，因为80%的样本确实是NA)")
    print(f"  - F1 Score: {f1_na_bias:.3f} (很低，因为类别0和1的F1都是0)")
    
    print("\n情况2: 模型稍微好一些")
    # 模型预测（对少数类有一些预测能力）
    y_pred_better = [0]*20 + [1]*20 + [2]*960  # 20%的少数类被正确预测
    
    acc_better = accuracy_score(y_true, y_pred_better)
    f1_better = f1_score(y_true, y_pred_better, average='macro')
    
    print(f"  - Accuracy: {acc_better:.3f} (82%正确)")
    print(f"  - F1 Score: {f1_better:.3f} (仍然较低)")
    
    print("\n情况3: 模型表现良好")
    # 模型预测（对少数类有较好的预测能力）
    y_pred_good = [0]*80 + [1]*80 + [2]*840  # 80%的少数类被正确预测
    
    acc_good = accuracy_score(y_true, y_pred_good)
    f1_good = f1_score(y_true, y_pred_good, average='macro')
    
    print(f"  - Accuracy: {acc_good:.3f} (84%正确)")
    print(f"  - F1 Score: {f1_good:.3f} (显著提升)")
    
    return y_true, y_pred_na_bias, y_pred_better, y_pred_good

def analyze_f1_calculation():
    """详细分析F1分数的计算过程"""
    print("\n" + "=" * 80)
    print("🧮 F1 Score 详细计算过程")
    print("=" * 80)
    
    # 使用情况2的数据进行详细分析
    y_true = [0]*100 + [1]*100 + [2]*800
    y_pred = [0]*20 + [1]*20 + [2]*960
    
    print(f"\n📊 混淆矩阵分析:")
    print("真实\\预测    0    1    2")
    print("0          20    0    80")  # 类别0: 20个正确，80个被误分为类别2
    print("1           0   20    80")  # 类别1: 20个正确，80个被误分为类别2  
    print("2           0    0   800")  # 类别2: 800个全部正确
    
    print(f"\n🎯 各类别F1计算:")
    
    # 类别0
    tp_0 = 20  # 正确预测为类别0的数量
    fp_0 = 0   # 错误预测为类别0的数量
    fn_0 = 80  # 实际是类别0但被预测为其他类别的数量
    
    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    print(f"类别0:")
    print(f"  Precision = {tp_0}/({tp_0}+{fp_0}) = {precision_0:.3f}")
    print(f"  Recall = {tp_0}/({tp_0}+{fn_0}) = {recall_0:.3f}")
    print(f"  F1 = 2×({precision_0:.3f}×{recall_0:.3f})/({precision_0:.3f}+{recall_0:.3f}) = {f1_0:.3f}")
    
    # 类别1
    tp_1 = 20
    fp_1 = 0
    fn_1 = 80
    
    precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
    recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    print(f"\n类别1:")
    print(f"  Precision = {tp_1}/({tp_1}+{fp_1}) = {precision_1:.3f}")
    print(f"  Recall = {tp_1}/({tp_1}+{fn_1}) = {recall_1:.3f}")
    print(f"  F1 = 2×({precision_1:.3f}×{recall_1:.3f})/({precision_1:.3f}+{recall_1:.3f}) = {f1_1:.3f}")
    
    # 类别2
    tp_2 = 800
    fp_2 = 160  # 80+80个被错误预测为类别2
    fn_2 = 0
    
    precision_2 = tp_2 / (tp_2 + fp_2) if (tp_2 + fp_2) > 0 else 0
    recall_2 = tp_2 / (tp_2 + fn_2) if (tp_2 + fn_2) > 0 else 0
    f1_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0
    
    print(f"\n类别2 (NA):")
    print(f"  Precision = {tp_2}/({tp_2}+{fp_2}) = {precision_2:.3f}")
    print(f"  Recall = {tp_2}/({tp_2}+{fn_2}) = {recall_2:.3f}")
    print(f"  F1 = 2×({precision_2:.3f}×{recall_2:.3f})/({precision_2:.3f}+{recall_2:.3f}) = {f1_2:.3f}")
    
    # Macro F1
    macro_f1 = (f1_0 + f1_1 + f1_2) / 3
    print(f"\n🎯 Macro F1 = ({f1_0:.3f} + {f1_1:.3f} + {f1_2:.3f}) / 3 = {macro_f1:.3f}")
    
    return macro_f1

def explain_your_situation():
    """解释你的具体情况"""
    print("\n" + "=" * 80)
    print("🎯 你的具体情况分析")
    print("=" * 80)
    
    print("\n📊 你的训练结果:")
    print("- Epoch 1: Accuracy = 0.7196, F1 = 0.3176")
    print("- Epoch 2+: Accuracy = 0.1174, F1 = 0.0700")
    
    print("\n🔍 问题分析:")
    print("1. 第一个epoch表现相对较好:")
    print("   - Accuracy 0.72: 说明模型对多数类(NA)有较好的预测能力")
    print("   - F1 0.32: 说明模型对少数类(0,1)也有一定的预测能力")
    
    print("\n2. 第二个epoch开始崩溃:")
    print("   - Accuracy 0.12: 模型完全失去预测能力")
    print("   - F1 0.07: 所有类别的预测都失效")
    print("   - 这通常表明梯度爆炸或学习率过高")
    
    print("\n🎯 NA类占比70-80%的影响:")
    print("1. 高Accuracy但低F1的原因:")
    print("   - 模型容易学会预测多数类(NA)")
    print("   - 但难以学会预测少数类(0,1)")
    print("   - Accuracy主要反映多数类的预测能力")
    print("   - F1反映所有类别的平衡预测能力")
    
    print("\n2. 为什么F1更重要:")
    print("   - F1考虑了类别不平衡")
    print("   - 更公平地评估模型性能")
    print("   - 在医学/生物学任务中，少数类往往更重要")
    
    print("\n💡 解决方案:")
    print("1. 类别权重 (已添加): 降低NA类权重到0.1")
    print("2. 数据平衡: 使用重采样技术")
    print("3. 损失函数: 使用Focal Loss等")
    print("4. 评估指标: 关注F1而不是Accuracy")

def main():
    """主函数"""
    print("🔍 Accuracy vs F1 Score 详细分析")
    print("解释为什么会出现高Accuracy但低F1的情况")
    
    # 1. 解释指标含义
    explain_metrics()
    
    # 2. 分析F1计算过程
    analyze_f1_calculation()
    
    # 3. 解释具体情况
    explain_your_situation()
    
    print("\n" + "=" * 80)
    print("🎯 总结")
    print("=" * 80)
    print("1. Accuracy高但F1低 = 模型偏向多数类，忽略少数类")
    print("2. NA类占比70-80% = 严重的数据不平衡问题")
    print("3. 类别权重[1.0, 1.0, 0.1] = 降低NA类影响，提升少数类性能")
    print("4. F1比Accuracy更适合评估不平衡数据集的性能")

if __name__ == "__main__":
    main()


