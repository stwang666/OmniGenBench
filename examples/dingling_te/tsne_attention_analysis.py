# -*- coding: utf-8 -*-
# file: tsne_attention_analysis.py
# time: 09:35 29/10/2025
# author: t-SNE Attention Score Analysis for Classification Evaluation
# Copyright (C) 2019-2025. All Rights Reserved.

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import warnings
warnings.filterwarnings('ignore')

from omnigenbench import ModelHub
from explainable_triclass import (
    OmniModelForTriClassTESequenceClassification,
    TriClassTEDataset,
    label_names,
    tissue_names
)


def extract_last_hidden_states(model, sequences, batch_size=16):
    """
    提取模型的last hidden states用于t-SNE降维
    Extract last hidden states from the model for t-SNE visualization
    
    返回:
        hidden_states: [num_samples, hidden_dim] - 用于t-SNE降维
        predictions: [num_samples, num_tissues] - 预测类别
    
    注意：batch_size默认为16以节省GPU内存，如果仍然遇到OOM，可以进一步减小
    """
    print("\n🔍 正在提取last hidden states...")
    print(f"   批次大小: {batch_size}")
    
    model.eval()
    all_hidden_states = []
    all_predictions = []
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    print(f"   模型设备: {device}")
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            # Tokenize
            inputs = model.tokenizer(
                batch_seqs,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # 将输入移动到模型所在的设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True
            )
            
            # 获取最后一层的hidden state并进行pooling
            last_hidden = outputs.last_hidden_state
            pooled = model.pooler(inputs['input_ids'], last_hidden)
            
            # 获取预测
            logits = model.classifier(pooled)
            batch_size_actual = logits.shape[0]
            logits = logits.view(batch_size_actual, model.num_labels, model.num_classes)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_hidden_states.append(pooled.cpu())
            all_predictions.append(preds.cpu())
            
            if (i + batch_size) % 100 == 0:
                print(f"  处理进度: {min(i + batch_size, len(sequences))}/{len(sequences)}")
    
    hidden_states = torch.cat(all_hidden_states, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()
    
    print(f"✅ 提取完成！")
    print(f"   Hidden states shape: {hidden_states.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    
    return hidden_states, predictions


def perform_tsne_analysis(hidden_states, labels, tissue_idx, perplexity=30, max_iter=1000):
    """
    对hidden states进行t-SNE降维
    Perform t-SNE dimensionality reduction on hidden states
    
    参数:
        hidden_states: [num_samples, hidden_dim]
        labels: [num_samples] - 真实标签
        tissue_idx: 要分析的组织索引
        perplexity: t-SNE参数，建议值在5-50之间
        max_iter: 最大迭代次数
    
    返回:
        tsne_results: [num_samples, 2] - 降维后的2D坐标
    """
    print(f"\n🎯 对组织 '{tissue_names[tissue_idx]}' 进行t-SNE降维...")
    print(f"   参数: perplexity={perplexity}, max_iter={max_iter}")
    
    # 过滤掉无效标签（-100表示该组织无标注）
    valid_mask = labels[:, tissue_idx] != -100
    valid_hidden = hidden_states[valid_mask]
    valid_labels = labels[valid_mask, tissue_idx]
    
    print(f"   有效样本数: {valid_hidden.shape[0]}")
    print(f"   类别分布: Low={np.sum(valid_labels==0)}, "
          f"Medium={np.sum(valid_labels==1)}, High={np.sum(valid_labels==2)}")
    
    # t-SNE降维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
        verbose=1
    )
    
    tsne_results = tsne.fit_transform(valid_hidden)
    
    print(f"✅ t-SNE降维完成！")
    
    return tsne_results, valid_labels, valid_mask


def visualize_tsne_by_tissue(tsne_results, labels, tissue_name, save_path=None):
    """
    可视化单个组织的t-SNE结果，按三个类别着色
    Visualize t-SNE results for a single tissue with three classes colored differently
    
    参数:
        tsne_results: [num_samples, 2] - t-SNE降维后的2D坐标
        labels: [num_samples] - 类别标签 (0=Low, 1=Medium, 2=High)
        tissue_name: 组织名称
        save_path: 保存路径
    """
    print(f"\n🎨 生成组织 '{tissue_name}' 的t-SNE可视化图...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 颜色映射：Low=绿色, Medium=橙色, High=红色
    colors = {0: '#2ECC71', 1: '#F39C12', 2: '#E74C3C'}
    color_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    # 绘制每个类别的散点
    for class_idx in range(3):
        mask = labels == class_idx
        num_samples = np.sum(mask)
        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=colors[class_idx],
            label=f'{color_labels[class_idx]} (n={num_samples})',
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.8
        )
    
    ax.set_title(f't-SNE Visualization: {tissue_name}\nTE Expression Classification', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.legend(title='TE Expression Level', fontsize=11, title_fontsize=12, 
              loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加样本总数信息
    total_samples = len(labels)
    ax.text(0.02, 0.98, f'Total Samples: {total_samples}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图片已保存到: {save_path}")
    
    plt.show()
    print(f"✅ 可视化完成！")


def evaluate_clustering_quality(tsne_results, labels):
    """
    评估t-SNE聚类质量
    Evaluate the quality of t-SNE clustering using silhouette score
    """
    print(f"\n📊 评估聚类质量...")
    
    # 计算轮廓系数 (Silhouette Score)
    # 范围: [-1, 1]，越接近1表示聚类越好
    silhouette_avg = silhouette_score(tsne_results, labels)
    
    print(f"   轮廓系数 (Silhouette Score): {silhouette_avg:.4f}")
    print(f"   解释: 范围[-1, 1]，越接近1表示类别分离越好")
    
    if silhouette_avg > 0.5:
        print(f"   ✅ 聚类质量: 优秀 - 类别在t-SNE空间中分离良好")
    elif silhouette_avg > 0.3:
        print(f"   ⚠️  聚类质量: 中等 - 类别有一定分离但存在重叠")
    else:
        print(f"   ❌ 聚类质量: 较差 - 类别在t-SNE空间中混合严重")
    
    return silhouette_avg


def comprehensive_tsne_analysis(model, test_data, tissue_indices=[0, 1, 2], 
                                perplexity=30, save_dir="/home/sw1136/OmniGenBench/examples/dingling_te/"):
    """
    对多个组织进行完整的t-SNE分析
    Perform comprehensive t-SNE analysis for multiple tissues
    """
    print("="*80)
    print("🚀 开始基于Last Hidden States的t-SNE分析")
    print("="*80)
    
    sequences = test_data['seq'].tolist()
    
    # 提取last hidden states
    hidden_states, predictions = extract_last_hidden_states(model, sequences)
    
    # 准备真实标签
    label_cols = [f'{tissue_names[i]}_TE_label' for i in range(len(tissue_names))]
    label_map = {'Low': 0, 'Medium': 1, 'High': 2, 'nan': -100}
    
    true_labels = []
    for _, row in test_data.iterrows():
        labels = [label_map.get(str(row.get(col, 'nan')), -100) for col in label_cols]
        true_labels.append(labels)
    
    true_labels = np.array(true_labels)
    
    # 对每个组织进行t-SNE分析
    results_summary = []
    
    for tissue_idx in tissue_indices:
        tissue_name = tissue_names[tissue_idx]
        print(f"\n{'='*80}")
        print(f"📍 分析组织: {tissue_name} (索引 {tissue_idx})")
        print(f"{'='*80}")
        
        # t-SNE降维
        tsne_results, valid_labels, valid_mask = perform_tsne_analysis(
            hidden_states, true_labels, tissue_idx, perplexity=perplexity
        )
        
        # 评估聚类质量
        silhouette = evaluate_clustering_quality(tsne_results, valid_labels)
        
        # 可视化：按真实标签着色
        save_path = f"{save_dir}tsne_{tissue_name}_classification.png"
        visualize_tsne_by_tissue(tsne_results, valid_labels, tissue_name, save_path)
        
        # 获取对应的预测标签
        valid_predictions = predictions[valid_mask, tissue_idx]
        
        # 生成分类报告
        print(f"\n📊 分类性能报告:")
        print(classification_report(
            valid_labels, valid_predictions,
            target_names=label_names,
            digits=4
        ))
        
        # 保存结果摘要
        accuracy = np.mean(valid_labels == valid_predictions)
        results_summary.append({
            'Tissue': tissue_name,
            'Accuracy': accuracy,
            'Silhouette Score': silhouette,
            'Num Samples': len(valid_labels),
            'Low': np.sum(valid_labels == 0),
            'Medium': np.sum(valid_labels == 1),
            'High': np.sum(valid_labels == 2)
        })
    
    # 生成总结表格
    print(f"\n{'='*80}")
    print("📋 所有组织的分析总结")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # 保存总结
    summary_path = f"{save_dir}tsne_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 分析总结已保存到: {summary_path}")
    
    print(f"\n{'='*80}")
    print("🎉 t-SNE分析全部完成！")
    print(f"{'='*80}")
    
    return summary_df


# ==================== 主程序 ====================

if __name__ == "__main__":
    
    print("\n📚 什么是t-SNE降维？")
    print("="*80)
    print("t-SNE (t-distributed Stochastic Neighbor Embedding) 是一种非线性降维技术：")
    print("  • 将高维数据（如512维hidden states）降到2D用于可视化")
    print("  • 保持数据点之间的相似性关系：相似样本在低维空间中靠近")
    print("  • 适合发现数据中的聚类结构")
    print("")
    print("📊 在本分析中的应用：")
    print("  1️⃣  基于Last Hidden States进行降维")
    print("  2️⃣  针对每个组织（tissue）绘制独立的二维图")
    print("  3️⃣  每个类别（Low/Medium/High）用不同颜色表示")
    print("  4️⃣  评估类别的可分性和聚类质量")
    print("="*80)
    
    # 加载模型
    print("\n🔄 加载训练好的tri-class TE模型...")
    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
    model = ModelHub.load(model_path)
    
    device = next(model.parameters()).device
    print(f"✅ 模型已加载到: {device}")
    
    # 加载测试数据
    print("\n📂 加载测试数据...")
    test_data = pd.read_csv("/home/sw1136/OmniGenBench/examples/dingling_te/train.csv")
    
    # 使用整个测试集，不进行抽样
    print(f"✅ 测试数据加载完成，共 {len(test_data)} 个样本")
    print(f"   使用完整测试集进行分析")
    
    # 选择要分析的组织
    # 0=root, 1=seedling, 2=leaf
    tissues_to_analyze = [0, 1, 2, 3, 4, 5, 6, 7, 8]  
    
    # 执行综合分析
    summary_df = comprehensive_tsne_analysis(
        model=model,
        test_data=test_data,
        tissue_indices=tissues_to_analyze,
        perplexity=30,  # t-SNE参数，建议在5-50之间
        save_dir="/home/sw1136/OmniGenBench/examples/dingling_te/"
    )
    
    print("\n" + "="*80)
    print("💡 使用建议:")
    print("="*80)
    print("1. 如果t-SNE图中不同颜色的点聚成不同的簇 → 说明模型学到了有效特征")
    print("2. 如果轮廓系数(Silhouette Score) > 0.5 → 说明分类边界清晰")
    print("3. 如果三个类别在空间中分离良好 → 说明该组织的分类任务相对容易")
    print("4. 如果类别之间有重叠 → 说明存在难以区分的边界样本")
    print("="*80)
