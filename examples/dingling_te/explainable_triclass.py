# -*- coding: utf-8 -*-
# file: explainable_triclass.py
# time: 09:35 19/10/2025
# author: Explainability Analysis for TriClass TE Model
# Copyright (C) 2019-2025. All Rights Reserved.

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from omnigenbench import (
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)

label_names = ['Low', 'Medium', 'High']
tissue_names = [
    'root', 'seedling', 'leaf', 'FMI', 'FOD',
    'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
]


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



# ==================== 辅助函数 ====================

def get_prediction_prob(model, sequence, tissue_idx=0, target_class=2):
    """获取模型对于给定序列在特定组织中属于 target_class 的概率"""
    with torch.no_grad():
        outputs = model.inference(sequence)
        probabilities = outputs['probabilities']
        return probabilities[tissue_idx, target_class].cpu().item()


def get_all_predictions(model, sequence):
    """获取序列在所有组织上的预测结果"""
    with torch.no_grad():
        outputs = model.inference(sequence)
        predictions = outputs['predictions'].cpu().numpy()
        probabilities = outputs['probabilities'].cpu().numpy()
        return predictions, probabilities


# ==================== 定点突变分析 ====================

def in_silico_mutagenesis(model, sequence, tissue_idx=0, target_class=2):
    """对序列进行定点突变，并计算每个突变对模型预测概率的影响"""
    print(f"\n🔬 正在对序列进行计算机内定点突变分析...")
    print(f"📊 分析组织: {tissue_names[tissue_idx]}")
    print(f"🎯 目标类别: {label_names[target_class]}")
    
    NUCLEOTIDES = ['A', 'T', 'C', 'G']
    seq_len = len(sequence)
    importance_scores = np.zeros((len(NUCLEOTIDES), seq_len))
    
    baseline_score = get_prediction_prob(model, sequence, tissue_idx, target_class)
    print(f"📈 基准序列预测为'{label_names[target_class]}'的概率: {baseline_score:.4f}")
    
    for i in range(seq_len):
        original_nucleotide = sequence[i].upper()
        
        for j, mutated_nucleotide in enumerate(NUCLEOTIDES):
            if original_nucleotide == mutated_nucleotide:
                importance_scores[j, i] = 0
                continue
            
            mutated_sequence = list(sequence)
            mutated_sequence[i] = mutated_nucleotide
            mutated_sequence = "".join(mutated_sequence)
            
            mutated_score = get_prediction_prob(model, mutated_sequence, tissue_idx, target_class)
            score_change = baseline_score - mutated_score
            importance_scores[j, i] = score_change
        
        if (i + 1) % 50 == 0 or i == seq_len - 1:
            print(f"⏳ 进度: {i + 1}/{seq_len} 个位置 ({(i+1)/seq_len*100:.1f}%)")
    
    print("✅ 定点突变分析完成！")
    return importance_scores, baseline_score


# ==================== 可视化 ====================

def visualize_importance(importance_scores, baseline_score, sequence, 
                        tissue_name, target_class_name, save_path=None):
    """可视化重要性得分热力图"""
    print("\n🎨 正在生成重要性热力图...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    
    # 热力图
    sns.heatmap(
        importance_scores,
        cmap="RdBu_r",
        center=0,
        yticklabels=['A', 'T', 'C', 'G'],
        cbar_kws={'label': 'Score Change (Importance)'},
        ax=ax1
    )
    ax1.set_title(f"In-Silico Mutagenesis Saliency Map\n组织: {tissue_name} | 目标类别: {target_class_name} | 基准概率: {baseline_score:.4f}")
    ax1.set_xlabel("序列位置")
    ax1.set_ylabel("突变为")
    
    # 柱状图
    max_importance = np.max(importance_scores, axis=0)
    positions = np.arange(len(sequence))
    
    ax2.bar(positions, max_importance, color='steelblue', alpha=0.7)
    ax2.set_xlabel("序列位置")
    ax2.set_ylabel("最大重要性得分")
    ax2.set_title("每个位置的最大重要性得分")
    ax2.grid(True, alpha=0.3)
    
    # 标注最重要的位置
    top_positions = np.argsort(max_importance)[-min(10, len(sequence)):][::-1]
    for pos in top_positions:
        ax2.text(pos, max_importance[pos], f'{sequence[pos]}', 
                ha='center', va='bottom', fontsize=8, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图像已保存到: {save_path}")
    
    plt.show()


# ==================== 综合分析 ====================

def analyze_sequence_comprehensive(model, sequence, tissue_idx=0):
    """对一个序列在特定组织上进行全面分析"""
    tissue_name = tissue_names[tissue_idx]
    print(f"\n{'='*70}")
    print(f"🔍 开始综合分析")
    print(f"🧬 序列长度: {len(sequence)} bp")
    print(f"🏥 分析组织: {tissue_name}")
    print(f"{'='*70}")
    
    # 1. 获取预测结果
    predictions, probabilities = get_all_predictions(model, sequence)
    
    print(f"\n📊 所有组织的预测结果:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        print(f"  {tissue_names[i]:25s}: {label_names[pred]:6s} "
              f"(Low: {probs[0]:.3f}, Medium: {probs[1]:.3f}, High: {probs[2]:.3f})")
    
    # 2. 突变分析
    target_class = predictions[tissue_idx]
    print(f"\n🎯 将对 {tissue_name} 组织的预测类别 '{label_names[target_class]}' 进行突变分析...")
    
    importance_scores, baseline_score = in_silico_mutagenesis(
        model, sequence, tissue_idx, target_class
    )
    
    # 3. 可视化
    save_path = f"/home/sw1136/OmniGenBench/examples/dingling_te/mutagenesis_{tissue_name}_{label_names[target_class]}.png"
    visualize_importance(
        importance_scores, baseline_score, sequence,
        tissue_name, label_names[target_class], save_path
    )
    
    # 4. 输出关键位置
    max_importance = np.max(importance_scores, axis=0)
    top_positions = np.argsort(max_importance)[-min(10, len(sequence)):][::-1]
    
    print(f"\n🔝 最重要的{len(top_positions)}个位置:")
    for rank, pos in enumerate(top_positions, 1):
        nt = sequence[pos]
        score = max_importance[pos]
        best_mut_idx = np.argmax(importance_scores[:, pos])
        best_mut = ['A', 'T', 'C', 'G'][best_mut_idx]
        print(f"  {rank:2d}. 位置 {pos:4d}: {nt} → {best_mut} (重要性: {score:.4f})")
    
    return importance_scores, baseline_score


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 示例序列（使用较短序列进行演示）
    # 完整分析500bp序列大约需要10-15分钟


    # ==================== 加载模型 ====================

    print("🔄 正在加载训练好的三分类 TE 模型...")

    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"

    inference_model = ModelHub.load(model_path)


    example_sequence = "GAGGGAGGGAAACGGGGGAGGGGAATGGGATGCTCCATTAGCTAAGCTCTGGTCTGATTACACGCCATTTCAGGAGCCATCGGTGGATCCGCCTCCCCCTCGCCCCTCGCCTACACCCCC"
    
    # 选择要分析的组织
    tissues_to_analyze = [0]  
    
    for tissue_idx in tissues_to_analyze:
        analyze_sequence_comprehensive(inference_model, example_sequence, tissue_idx)
        print("\n" + "="*70 + "\n")
    
    print("🎉 所有可解释性分析完成！")
