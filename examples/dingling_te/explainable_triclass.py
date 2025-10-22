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

def in_silico_mutagenesis(model, sequence, tissue_idx=0, target_class=2, normalize='none'):
    """
    对序列进行定点突变，并计算每个突变对模型预测概率的影响
    Perform in-silico mutagenesis on the sequence and calculate mutation impact
    
    参数 / Parameters:
        normalize: 归一化方法 / Normalization method
            - 'none': 不归一化，绝对概率变化 / No normalization (absolute probability change)
            - 'relative': 相对归一化，相对于基准概率的百分比 / Relative to baseline (%)
            - 'minmax': Min-Max归一化到[0, 1] / Min-Max normalization to [0, 1]
            - 'zscore': Z-score标准化 / Z-score standardization
    
    返回:
        importance_scores: 重要性得分矩阵 [4, seq_len]
        baseline_score: 基准概率
        predicted_classes: 每个位置突变后的预测类别 [4, seq_len]
    """
    print(f"\n🔬 Performing in-silico mutagenesis analysis...")
    print(f"📊 Tissue: {tissue_names[tissue_idx]}")
    print(f"🎯 Target class: {label_names[target_class]}")
    print(f"📏 Normalization: {normalize}")
    
    NUCLEOTIDES = ['A', 'T', 'C', 'G']
    seq_len = len(sequence)
    importance_scores = np.zeros((len(NUCLEOTIDES), seq_len))
    predicted_classes = np.zeros((len(NUCLEOTIDES), seq_len), dtype=int)
    
    baseline_score = get_prediction_prob(model, sequence, tissue_idx, target_class)
    print(f"📈 Baseline probability for '{label_names[target_class]}': {baseline_score:.4f}")
    
    # 计算所有位置的重要性得分和预测类别
    for i in range(seq_len):
        original_nucleotide = sequence[i].upper()
        
        for j, mutated_nucleotide in enumerate(NUCLEOTIDES):
            if original_nucleotide == mutated_nucleotide:
                importance_scores[j, i] = 0
                # 对于原始碱基，使用基准预测类别
                with torch.no_grad():
                    outputs = model.inference(sequence)
                    predicted_classes[j, i] = outputs['predictions'][tissue_idx].cpu().item()
                continue
            
            mutated_sequence = list(sequence)
            mutated_sequence[i] = mutated_nucleotide
            mutated_sequence = "".join(mutated_sequence)
            
            with torch.no_grad():
                outputs = model.inference(mutated_sequence)
                mutated_score = outputs['probabilities'][tissue_idx, target_class].cpu().item() # 取得是target_class的概率，target_class是2，所以是High的概率；但low 和medium的概率变化这里没有计算
                predicted_class = outputs['predictions'][tissue_idx].cpu().item()
            
            score_change = baseline_score - mutated_score
            importance_scores[j, i] = score_change
            predicted_classes[j, i] = predicted_class
        
        if (i + 1) % 50 == 0 or i == seq_len - 1:
            print(f"⏳ Progress: {i + 1}/{seq_len} positions ({(i+1)/seq_len*100:.1f}%)")
    
    # 应用归一化
    original_scores = importance_scores.copy()  # 保存原始分数用于报告
    
    if normalize == 'relative':
        # 相对于基准概率的变化百分比
        if baseline_score > 1e-6:
            importance_scores = importance_scores / baseline_score
            print(f"✅ Applied relative normalization (divided by baseline {baseline_score:.4f})")
            print(f"   相对重要性：现在的值表示相对于基准概率的变化比例")
        else:
            print(f"⚠️  Baseline probability too low, skipping relative normalization")
    
    elif normalize == 'minmax':
        # Min-Max归一化到[0, 1]
        min_val = importance_scores.min()
        max_val = importance_scores.max()
        if max_val > min_val + 1e-6:
            importance_scores = (importance_scores - min_val) / (max_val - min_val)
            print(f"✅ Applied Min-Max normalization")
            print(f"   原始范围: [{min_val:.4f}, {max_val:.4f}] → [0, 1]")
        else:
            print(f"⚠️  All values similar, skipping Min-Max normalization")
    
    elif normalize == 'zscore':
        # Z-score标准化
        mean_val = importance_scores.mean()
        std_val = importance_scores.std()
        if std_val > 1e-6:
            importance_scores = (importance_scores - mean_val) / std_val
            print(f"✅ Applied Z-score standardization")
            print(f"   均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
        else:
            print(f"⚠️  Standard deviation too low, skipping Z-score normalization")
    
    elif normalize == 'none':
        print(f"✅ Using raw importance scores (absolute probability change)")
        print(f"   原始分数范围: [{importance_scores.min():.4f}, {importance_scores.max():.4f}]")
    
    else:
        print(f"⚠️  Unknown normalization method: {normalize}, using raw scores")
    
    print("✅ In-silico mutagenesis completed!")
    return importance_scores, baseline_score, predicted_classes


# ==================== 可视化 ====================

def visualize_importance(importance_scores, baseline_score, sequence, 
                        tissue_name, target_class_name, predicted_classes=None, save_path=None):
    """Visualize importance score heatmap with predicted class changes"""
    print("\n🎨 Generating importance heatmap...")
    
    # 创建3个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))
    
    # 子图1: 重要性热力图
    sns.heatmap(
        importance_scores,
        cmap="RdBu_r",
        center=0,
        yticklabels=['A', 'T', 'C', 'G'],
        cbar_kws={'label': 'Score Change (Importance)'},
        ax=ax1,
        xticklabels=False
    )
    ax1.set_title(f"In-Silico Mutagenesis Saliency Map\nTissue: {tissue_name} | Target Class: {target_class_name} | Baseline Probability: {baseline_score:.4f}")
    ax1.set_ylabel("Mutated To")
    
    # 在热力图下方添加序列
    seq_len = len(sequence)
    
    # 添加位置编号（每隔5个位置显示）
    for i in range(seq_len):
        if i % 5 == 0:
            ax1.text(i + 0.5, len(['A', 'T', 'C', 'G']) + 0.3, 
                    str(i), 
                    ha='center', va='top', fontsize=6, 
                    color='black', alpha=0.7)
    
    # 添加序列（在位置编号下方）
    for i, nucleotide in enumerate(sequence):
        # Color code nucleotides
        if nucleotide.upper() == 'A':
            color = '#FF6B6B'
        elif nucleotide.upper() == 'T':
            color = '#4ECDC4'
        elif nucleotide.upper() == 'C':
            color = '#45B7D1'
        elif nucleotide.upper() == 'G':
            color = '#95E1D3'
        else:
            color = 'gray'
        
        ax1.text(i + 0.5, len(['A', 'T', 'C', 'G']) + 0.6, 
                nucleotide.upper(), 
                ha='center', va='top', fontsize=8, 
                color=color, weight='bold')
    
    # 在序列下方添加xlabel
    ax1.text(seq_len / 2, len(['A', 'T', 'C', 'G']) + 1.0, 
            "Sequence Position", 
            ha='center', va='top', fontsize=10, 
            color='black', weight='bold')
    
    # 子图2: 最大重要性得分柱状图
    max_importance = np.max(importance_scores, axis=0)
    positions = np.arange(len(sequence))
    
    ax2.bar(positions, max_importance, color='steelblue', alpha=0.7)
    ax2.set_xlabel("")
    ax2.set_ylabel("Maximum Importance Score")
    ax2.set_title("Maximum Importance Score at Each Position")
    ax2.grid(True, alpha=0.3)
    
    # Annotate the most important positions
    top_positions = np.argsort(max_importance)[-min(10, len(sequence)):][::-1]
    for pos in top_positions:
        ax2.text(pos, max_importance[pos], f'{sequence[pos]}', 
                ha='center', va='bottom', fontsize=8, color='red')
    
    # 在柱状图下方添加序列和位置编号
    for i in range(seq_len):
        if i % 5 == 0:
            ax2.text(i, ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.05, 
                    str(i), 
                    ha='center', va='top', fontsize=6, 
                    color='black', alpha=0.7)
    
    for i, nucleotide in enumerate(sequence):
        # Color code nucleotides
        if nucleotide.upper() == 'A':
            color = '#FF6B6B'
        elif nucleotide.upper() == 'T':
            color = '#4ECDC4'
        elif nucleotide.upper() == 'C':
            color = '#45B7D1'
        elif nucleotide.upper() == 'G':
            color = '#95E1D3'
        else:
            color = 'gray'
        
        ax2.text(i, ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.10, 
                nucleotide.upper(), 
                ha='center', va='top', fontsize=8, 
                color=color, weight='bold')
    
    # 调整y轴范围以容纳序列和位置编号
    y_min, y_max = ax2.get_ylim()
    ax2.set_ylim(y_min - (y_max - y_min) * 0.15, y_max)
    
    # 子图3: 突变后预测类别热力图
    if predicted_classes is not None:
        # 定义类别颜色映射: Low=0(绿色), Medium=1(黄色), High=2(红色)
        class_colors = {0: '#2ECC71', 1: '#F39C12', 2: '#E74C3C'}  # Low, Medium, High
        
        # 创建自定义colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([class_colors[0], class_colors[1], class_colors[2]])
        
        sns.heatmap(
            predicted_classes,
            cmap=cmap,
            yticklabels=['A', 'T', 'C', 'G'],
            cbar_kws={'label': 'Predicted Class', 'ticks': [0, 1, 2]},
            ax=ax3,
            xticklabels=False,
            vmin=0,
            vmax=2
        )
        
        # 设置colorbar标签
        cbar = ax3.collections[0].colorbar
        cbar.set_ticklabels(['Low', 'Medium', 'High'])
        
        ax3.set_title(f"Predicted Class After Mutation\n(Green=Low, Yellow=Medium, Red=High)")
        ax3.set_ylabel("Mutated To")
        
        # 在热力图下方添加序列
        for i in range(seq_len):
            if i % 5 == 0:
                ax3.text(i + 0.5, len(['A', 'T', 'C', 'G']) + 0.3, 
                        str(i), 
                        ha='center', va='top', fontsize=6, 
                        color='black', alpha=0.7)
        
        for i, nucleotide in enumerate(sequence):
            if nucleotide.upper() == 'A':
                color = '#FF6B6B'
            elif nucleotide.upper() == 'T':
                color = '#4ECDC4'
            elif nucleotide.upper() == 'C':
                color = '#45B7D1'
            elif nucleotide.upper() == 'G':
                color = '#95E1D3'
            else:
                color = 'gray'
            
            ax3.text(i + 0.5, len(['A', 'T', 'C', 'G']) + 0.6, 
                    nucleotide.upper(), 
                    ha='center', va='top', fontsize=8, 
                    color=color, weight='bold')
        
        ax3.text(seq_len / 2, len(['A', 'T', 'C', 'G']) + 1.0, 
                "Sequence Position", 
                ha='center', va='top', fontsize=10, 
                color='black', weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Image saved to: {save_path}")
    
    plt.show()


# ==================== 综合分析 ====================

def analyze_sequence_comprehensive(model, sequence, tissue_idx=0, normalize='none'):
    """
    对一个序列在特定组织上进行全面分析
    Perform comprehensive analysis on a sequence for a specific tissue
    
    参数:
        normalize: 归一化方法 ('none', 'relative', 'minmax', 'zscore')
    """
    tissue_name = tissue_names[tissue_idx]
    print(f"\n{'='*70}")
    print(f"🔍 Starting comprehensive analysis")
    print(f"🧬 Sequence length: {len(sequence)} bp")
    print(f"🏥 Analyzing tissue: {tissue_name}")
    print(f"{'='*70}")
    
    # 1. Get prediction results
    predictions, probabilities = get_all_predictions(model, sequence)
    
    print(f"\n📊 Prediction results for all tissues:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        print(f"  {tissue_names[i]:25s}: {label_names[pred]:6s} "
              f"(Low: {probs[0]:.3f}, Medium: {probs[1]:.3f}, High: {probs[2]:.3f})")
    
    # 2. Mutagenesis analysis
    target_class = predictions[tissue_idx]
    print(f"\n🎯 Performing mutagenesis analysis for {tissue_name} tissue's predicted class '{label_names[target_class]}'...")
    
    importance_scores, baseline_score, predicted_classes = in_silico_mutagenesis(
        model, sequence, tissue_idx, target_class, normalize=normalize
    )
    
    # 3. Visualization
    norm_suffix = f"_{normalize}" if normalize != 'none' else ""
    save_path = f"/home/sw1136/OmniGenBench/examples/dingling_te/mutagenesis_{tissue_name}_{label_names[target_class]}{norm_suffix}.png"
    visualize_importance(
        importance_scores, baseline_score, sequence,
        tissue_name, label_names[target_class], predicted_classes, save_path
    )
    
    # 4. Output key positions
    max_importance = np.max(importance_scores, axis=0)
    top_positions = np.argsort(max_importance)[-min(10, len(sequence)):][::-1]
    
    print(f"\n🔝 Top {len(top_positions)} most important positions:")
    for rank, pos in enumerate(top_positions, 1):
        nt = sequence[pos]
        score = max_importance[pos]
        best_mut_idx = np.argmax(importance_scores[:, pos])
        best_mut = ['A', 'T', 'C', 'G'][best_mut_idx]
        print(f"  {rank:2d}. Position {pos:4d}: {nt} → {best_mut} (Importance: {score:.4f})")
    
    return importance_scores, baseline_score, predicted_classes




if __name__ == "__main__":
    
    print("🔄 Loading trained tri-class TE model...")

    model_path = "/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900"

    inference_model = ModelHub.load(model_path)

    example_sequence = "CTTGCACTCTCCACGCACGTCAGTCCAGCCTCGTCTCGTTTCGTGTCGTCTGCTCGGGACCAGGATAG"
    
    # Select tissues to analyze
    tissues_to_analyze = [2]  # 0 = root
    
    # ==================== 归一化方法选择 ====================
    # 可选的归一化方法：
    # - 'none': 不归一化（默认）- 适合单序列分析，关注绝对影响
    # - 'relative': 相对归一化 - 适合跨序列比较，关注相对影响
    # - 'minmax': Min-Max归一化 - 适合可视化对比，统一范围到[0,1]
    # - 'zscore': Z-score标准化 - 适合统计分析，识别显著异常值
    
    normalize_method = 'none'  # 👈 修改这里来选择不同的归一化方法
    for tissue_idx in tissues_to_analyze:
        analyze_sequence_comprehensive(
            inference_model, 
            example_sequence, 
            tissue_idx,
            normalize=normalize_method
        )
        print("\n" + "="*70 + "\n")
    
    print("🎉 All explainability analyses completed!")
    
    # # ==================== 使用建议 ====================
    # print("\n" + "="*70)
    # print("📚 归一化方法使用建议:")
    # print("="*70)
    # print("\n1️⃣  不归一化 (normalize='none')")
    # print("   ✅ 适用场景：单序列深度分析")
    # print("   ✅ 优点：保留绝对概率变化信息")
    # print("   ✅ 解释：重要性得分 = 基准概率 - 突变后概率")
    # print("   📊 例如：0.5 表示突变导致概率下降了50个百分点\n")
    
    # print("2️⃣  相对归一化 (normalize='relative')")
    # print("   ✅ 适用场景：跨序列比较，不同基因间对比")
    # print("   ✅ 优点：消除基准概率差异的影响")
    # print("   ✅ 解释：重要性得分 = (基准概率 - 突变后概率) / 基准概率")
    # print("   📊 例如：0.5 表示突变导致概率下降了50%\n")
    
    # print("3️⃣  Min-Max归一化 (normalize='minmax')")
    # print("   ✅ 适用场景：热力图可视化对比，跨实验比较")
    # print("   ✅ 优点：统一尺度，便于设定阈值")
    # print("   ✅ 解释：将所有得分线性缩放到[0, 1]")
    # print("   📊 1.0表示该位置最重要，0.0表示最不重要\n")
    
    # print("4️⃣  Z-score标准化 (normalize='zscore')")
    # print("   ✅ 适用场景：统计显著性分析，识别异常位点")
    # print("   ✅ 优点：考虑分布特征，突出显著变化")
    # print("   ✅ 解释：(得分 - 均值) / 标准差")
    # print("   📊 |Z| > 2 表示该位置显著重要（超过95%分位）")
    # print("\n" + "="*70)
