# -*- coding: utf-8 -*-
# file: sequence_perturbation_analysis.py
# time: 10:00 19/10/2025
# author: Sequence Perturbation Analysis for TE Classification
# Description: Analyze sequence importance by random perturbation and visualize with heatmap

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

from omnigenbench import (
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)

# 导入自定义的 Dataset 类
import sys
sys.path.append(os.path.dirname(__file__))


class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """Dataset for 3-class (Low/Medium/High) multi-label TE classification
    
    继承说明：
    - 继承自 OmniDatasetForMultiLabelClassification，获得多标签分类数据集的基础功能
    - 重写 prepare_input() 方法以适配 TE 3分类任务的特殊需求
    """

    def __init__(self, **kwargs):
        # 调用父类的初始化方法，继承父类的属性和行为
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        # Map labels to indices: Low=0, Medium=1, High=2
        label2idx = {'Low': 0, 'Medium': 1, 'High': 2, 'nan': -100}

        # Extract labels for all 9 tissues
        root_TE_label = label2idx[str(instance["root_TE_label"])]
        seedling_TE_label = label2idx[str(instance["seedling_TE_label"])]
        leaf_TE_label = label2idx[str(instance["leaf_TE_label"])]
        FMI_TE_label = label2idx[str(instance["FMI_TE_label"])]
        FOD_TE_label = label2idx[str(instance["FOD_TE_label"])]
        Prophase_I_pollen_TE_label = label2idx[str(instance["Prophase-I-pollen_TE_label"])]
        Tricellular_pollen_TE_label = label2idx[str(instance["Tricellular-pollen_TE_label"])]
        flag_TE_label = label2idx[str(instance["flag_TE_label"])]
        grain_TE_label = label2idx[str(instance["grain_TE_label"])]
        sequence = instance["sequence"]

        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Stack all labels
        labels = torch.tensor([
            root_TE_label,
            seedling_TE_label,
            leaf_TE_label,
            FMI_TE_label,
            FOD_TE_label,
            Prophase_I_pollen_TE_label,
            Tricellular_pollen_TE_label,
            flag_TE_label,
            grain_TE_label,
        ], dtype=torch.long)  # Use long for CrossEntropyLoss

        tokenized_inputs["labels"] = labels

        return tokenized_inputs

class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """Model for 3-class multi-label TE classification"""

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, *args, **kwargs):
        # For multi-label with 3 classes each, output should be [batch, num_labels, num_classes]
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels  # 9 tissues
        self.num_classes = num_classes  # 3 classes (Low/Medium/High)
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        # Use CrossEntropyLoss for multi-class classification
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # 🔑 NEW: Store dataset class reference for saving
        self.dataset_class = kwargs.pop('dataset_class', TriClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with proper reshaping for multi-label multi-class"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Get the logits from classifier head
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        logits = self.classifier(self.pooler(input_ids, logits))
        # Reshape logits from [batch, num_labels * num_classes] to [batch, num_labels, num_classes]
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        loss = None
        if labels is not None:
            # labels shape: [batch, num_labels]
            # Flatten for CrossEntropyLoss
            logits_flat = logits.view(-1, self.num_classes)  # [batch * num_labels, num_classes]
            labels_flat = labels.view(-1)  # [batch * num_labels]

            loss = self.loss_fn(logits_flat, labels_flat)

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        }

    def predict(self, sequence_or_inputs, **kwargs):
        """Prediction with softmax for multi-class"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Apply softmax to get probabilities for each label
        probabilities = torch.softmax(logits, dim=-1)  # [batch, num_labels, num_classes]

        # Get predicted class for each label
        predictions = torch.argmax(probabilities, dim=-1)  # [batch, num_labels]

        outputs = {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        """Inference wrapper"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Apply softmax
        probabilities = torch.softmax(logits, dim=-1)

        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)

        # Get confidence (max probability for each label)
        confidence, _ = torch.max(probabilities, dim=-1)

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "probabilities": probabilities[0],
                "confidence": confidence[0],
                "last_hidden_state": last_hidden_state[0] if last_hidden_state is not None else None,
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "probabilities": probabilities,
                "confidence": confidence,
                "last_hidden_state": last_hidden_state,
            }

        return outputs

def random_nucleotide_substitution(sequence, position):
    """
    在指定位置进行随机核苷酸替换
    
    Args:
        sequence: DNA序列字符串
        position: 要替换的位置索引
    
    Returns:
        扰动后的序列
    """
    nucleotides = ['A', 'T', 'C', 'G']
    seq_list = list(sequence)
    original_nuc = seq_list[position].upper()
    
    # 选择一个不同的核苷酸
    alternative_nucs = [n for n in nucleotides if n != original_nuc]
    seq_list[position] = random.choice(alternative_nucs)
    
    return ''.join(seq_list)


def calculate_prediction_change(original_pred, perturbed_pred):
    """
    计算预测变化的幅度
    
    Args:
        original_pred: 原始预测 [num_labels]
        perturbed_pred: 扰动后预测 [num_labels]
    
    Returns:
        变化分数（0-1之间，越大表示变化越大）
    """
    # 计算预测类别是否改变的数量
    changes = (original_pred != perturbed_pred).sum()
    # 归一化到 0-1
    return changes / len(original_pred)


def perturbation_importance_analysis(model, sequence, sample_data, 
                                     num_perturbations=5, 
                                     step_size=10,
                                     tissue_names=None):
    """
    通过随机扰动分析序列重要性
    
    Args:
        model: 训练好的模型
        sequence: 输入序列
        sample_data: 原始样本数据（字典）
        num_perturbations: 每个位置的扰动次数
        step_size: 采样步长（每隔多少个碱基进行一次扰动）
        tissue_names: 组织名称列表
    
    Returns:
        importance_scores: 每个位置的重要性分数 [seq_length]
        position_tissue_importance: 每个位置对每个组织的重要性 [seq_length, num_tissues]
    """
    if tissue_names is None:
        tissue_names = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
    
    seq_length = len(sequence)
    num_tissues = len(tissue_names)
    
    # 获取原始预测
    with torch.no_grad():
        original_outputs = model.inference(sample_data)
        original_pred = original_outputs['predictions'].cpu().numpy()
    
    print(f"🔍 Original predictions: {original_pred}")
    
    # 初始化重要性分数矩阵
    position_tissue_importance = np.zeros((seq_length, num_tissues))
    
    # 对序列中的每个位置进行扰动分析（使用步长减少计算量）
    positions_to_test = range(0, seq_length, step_size)
    
    print(f"\n🧪 Starting perturbation analysis...")
    print(f"   Sequence length: {seq_length}")
    print(f"   Testing {len(list(positions_to_test))} positions (step={step_size})")
    print(f"   Perturbations per position: {num_perturbations}")
    
    for pos in tqdm(positions_to_test, desc="Analyzing positions"):
        position_changes = []
        tissue_changes = np.zeros(num_tissues)
        
        # 对每个位置进行多次随机扰动
        for _ in range(num_perturbations):
            # 生成扰动序列
            perturbed_seq = random_nucleotide_substitution(sequence, pos)
            
            # 创建扰动样本数据
            perturbed_data = sample_data.copy()
            perturbed_data['sequence'] = perturbed_seq
            
            # 获取扰动后的预测
            with torch.no_grad():
                perturbed_outputs = model.inference(perturbed_data)
                perturbed_pred = perturbed_outputs['predictions'].cpu().numpy()
            
            # 计算整体变化
            overall_change = calculate_prediction_change(original_pred, perturbed_pred)
            position_changes.append(overall_change)
            
            # 计算每个组织的变化
            for tissue_idx in range(num_tissues):
                if original_pred[tissue_idx] != perturbed_pred[tissue_idx]:
                    tissue_changes[tissue_idx] += 1
        
        # 平均多次扰动的结果
        avg_importance = np.mean(position_changes)
        
        # 填充到所有位置（包括步长之间的位置）
        start_pos = pos
        end_pos = min(pos + step_size, seq_length)
        for p in range(start_pos, end_pos):
            position_tissue_importance[p] = tissue_changes / num_perturbations
    
    # 计算总体重要性分数（所有组织的平均）
    importance_scores = position_tissue_importance.mean(axis=1)
    
    return importance_scores, position_tissue_importance


def plot_importance_heatmap(importance_scores, sequence, 
                           position_tissue_importance=None,
                           tissue_names=None,
                           sample_id="sample",
                           save_path=None):
    """
    绘制序列重要性热力图
    
    Args:
        importance_scores: 重要性分数 [seq_length]
        sequence: 原始序列
        position_tissue_importance: 每个位置对每个组织的重要性 [seq_length, num_tissues]
        tissue_names: 组织名称列表
        sample_id: 样本ID
        save_path: 保存路径
    """
    if tissue_names is None:
        tissue_names = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
    
    seq_length = len(sequence)
    
    # 创建图形
    if position_tissue_importance is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), 
                                       gridspec_kw={'height_ratios': [1, 3]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 4))
    
    # 绘制总体重要性得分
    positions = np.arange(seq_length)
    colors = []
    for nuc in sequence:
        if nuc.upper() == 'A':
            colors.append('#FF6B6B')
        elif nuc.upper() == 'T':
            colors.append('#4ECDC4')
        elif nuc.upper() == 'C':
            colors.append('#45B7D1')
        elif nuc.upper() == 'G':
            colors.append('#FFA07A')
        else:
            colors.append('#CCCCCC')
    
    ax1.bar(positions, importance_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sequence Importance Analysis - {sample_id}\n'
                  f'Overall Importance (Change in Predictions)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='A'),
        Patch(facecolor='#4ECDC4', label='T'),
        Patch(facecolor='#45B7D1', label='C'),
        Patch(facecolor='#FFA07A', label='G')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=4)
    
    # 绘制每个组织的重要性热力图
    if position_tissue_importance is not None:
        sns.heatmap(position_tissue_importance.T, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Prediction Change Frequency'},
                   xticklabels=False,
                   yticklabels=tissue_names,
                   ax=ax2,
                   vmin=0,
                   vmax=1)
        ax2.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Tissue Type', fontsize=12, fontweight='bold')
        ax2.set_title('Tissue-Specific Importance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Heatmap saved to: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """主函数"""
    print("="*80)
    print("🧬 Sequence Perturbation Importance Analysis")
    print("="*80)
    
    # 加载模型
    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"
    print(f"\n📦 Loading model from: {model_path}")
    model = ModelHub.load(model_path)
    model.eval()
    
    # 加载数据集
    print("\n📊 Loading dataset...")
    model_name_or_path = "yangheng/OmniGenome-52M"
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    datasets = TriClassTEDataset.from_hub(
        dataset_name_or_path="examples/dingling_te/",  # 指定具体的数据目录
        tokenizer=tokenizer,
        max_length=512,
        force_padding=False
    )
    
    # 选择测试样本
    test_samples = datasets['test'].examples[:3]  # 分析前3个测试样本
    
    tissue_names = [
        'root', 'seedling', 'leaf', 'FMI', 'FOD',
        'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
    ]
    
    # 对每个样本进行分析
    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"🔬 Analyzing Sample {idx+1}/{len(test_samples)}")
        print(f"   Sample ID: {sample['ID']}")
        print(f"   Sequence Length: {len(sample['sequence'])} bp")
        print(f"{'='*80}")
        
        sequence = sample['sequence']
        
        # 进行扰动重要性分析
        importance_scores, position_tissue_importance = perturbation_importance_analysis(
            model=model,
            sequence=sequence,
            sample_data=sample,
            num_perturbations=5,  # 每个位置扰动5次
            step_size=10,  # 每隔10个碱基测试一次
            tissue_names=tissue_names
        )
        
        print(f"\n📊 Analysis Results:")
        print(f"   Mean importance score: {importance_scores.mean():.4f}")
        print(f"   Max importance score: {importance_scores.max():.4f}")
        print(f"   Top 10 important positions: {np.argsort(importance_scores)[-10:][::-1]}")
        
        # 绘制热力图
        save_path = f"perturbation_heatmap_sample_{sample['ID']}.png"
        plot_importance_heatmap(
            importance_scores=importance_scores,
            sequence=sequence,
            position_tissue_importance=position_tissue_importance,
            tissue_names=tissue_names,
            sample_id=sample['ID'],
            save_path=save_path
        )
        
        print(f"\n✅ Sample {idx+1} analysis completed!")
    
    print("\n" + "="*80)
    print("🎉 All perturbation analyses completed!")
    print("="*80)


if __name__ == "__main__":
    main()
