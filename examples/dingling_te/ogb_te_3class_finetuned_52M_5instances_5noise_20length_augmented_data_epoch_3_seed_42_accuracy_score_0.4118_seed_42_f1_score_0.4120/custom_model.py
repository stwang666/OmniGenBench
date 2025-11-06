# -*- coding: utf-8 -*-
# file: triclass_te.py
# time: 09:35 07/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import torch
import math

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)

model_name_or_path = "yangheng/OmniGenome-52M"
# model_name_or_path = "yangheng/OmniGenome-186M"
# model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "SpliceBERT-510nt"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


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
        self.dropout = torch.nn.Dropout(0.3)  # 可以增大到0.4-0.5训练损失下降但验证精度低，可能是过拟合
        # 正则化能减少模型对训练集的“记忆”，提升泛化能力

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
        logits = self.pooler(input_ids, logits)
        logits = self.dropout(logits)  # ← 在这里应用 dropout
        logits = self.classifier(logits)
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


# Load datasets
print("📊 Loading datasets...")
datasets = TriClassTEDataset.from_hub(
    "/home/sw1136/OmniGenBench/examples/dingling_te/te_augmentation/remove_duplicated_data_from_train",  # 指定具体的数据目录
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 Data loading completed!")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Initialize model
print("\n🚀 Initializing model...")
model = OmniModelForTriClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,  # 9 tissues
    num_classes=3,  # 3 classes: Low, Medium, High
    trust_remote_code=True
)

# Define metrics: accuracy and F1 score
# - accuracy_score: 计算整体分类准确率，忽略标签为-100的样本（通常用于padding或无效标签）
# - f1_score: 计算F1分数（精确率和召回率的调和平均），使用macro平均（对每个类别计算F1后取平均，适合类别不平衡的情况）
# 
# 计算时机：这些metrics会在训练过程中应用于eval_dataset（验证集）和test_dataset（测试集）
# 计算方式：
#   1. 模型对验证集/测试集进行前向传播，得到预测结果（logits）
#   2. 将logits转换为预测类别（argmax）
#   3. 将预测类别与真实标签进行比较，忽略标签为-100的位置
#   4. accuracy_score: 正确预测数 / 有效样本总数
#   5. f1_score (macro): 对每个类别分别计算F1值，然后取平均值
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,  # 宏平均F1：(TP) / (TP + 0.5*(FP+FN))，对所有类别取平均
    ClassificationMetric(ignore_y=-100).classification_report,
]

# Initialize trainer
 # batch_size: 每次从数据集中加载并处理的样本数量
    # - 这里设置为16，表示每个训练步骤会处理16个序列样本
    # - 较小的batch_size可以减少GPU内存占用，但训练可能不够稳定
    # - 较大的batch_size可以提高训练稳定性和速度，但需要更多GPU内存

# gradient_accumulation_steps: 梯度累积步数
    # - 这里设置为4，表示每4个batch才进行一次参数更新
    # - 实际有效batch_size = batch_size × gradient_accumulation_steps = 16 × 4 = 64
    # - 作用：在GPU内存有限的情况下，通过累积多个小batch的梯度来模拟大batch训练
    # - 工作原理：
    #   1. 前向传播和反向传播计算梯度（但不更新参数）
    #   2. 将梯度累加到之前的梯度上
    #   3. 重复步骤1-2共4次
    #   4. 第4次后，使用累积的梯度更新模型参数，然后清零梯度
    # - 优点：可以用较小的GPU内存训练出与大batch相当的效果
    
# 训练时不会用到test_dataset，它仅在训练完成后用于最终评估
# - train_dataset: 用于模型训练，更新模型参数
# - eval_dataset: 用于训练过程中的验证，监控过拟合，选择最佳模型
# - test_dataset: 仅在训练完成后用于最终性能评估，不参与训练过程

trainer = AccelerateTrainer(
    model=model,
    epochs=20,
    learning_rate=2e-5,
    batch_size=16,  # 每次训练的样本数量
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],  # 仅用于训练后的最终测试，不影响训练过程
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    early_stopping=True,
    patience=3,  # 验证集指标3个epoch不提升就停止
    monitor='valid_f1_score',  # 监控验证集F1分数
)
# trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_3class_finetuned_52M_5instances_5noise_20length_augmented_data", dataset_class=TriClassTEDataset)
print('📊 Final Metrics:', metrics)

# === Model Inference ===
print("\n🔮 Starting inference on test samples...")

inference_model = ModelHub.load("/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900")

# Get some test samples
# sample_sequences = datasets['test'].sample(1000).examples
#sample_sequences = datasets['valid'].sample(1000).examples
sample_sequences = datasets['train'].examples[:1]

label_names = ['Low', 'Medium', 'High']
tissue_names = [
    'root', 'seedling', 'leaf', 'FMI', 'FOD',
    'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
]

with torch.no_grad():
    for row in sample_sequences:
        sequence = row["sequence"]
        print(f"\n{'='*60}")
        print(f"🧬 Sample ID: {row['ID']}")
        print(f"📏 Sequence length: {len(sequence)} bp")

        outputs = inference_model.inference(sequence, **row)
        predictions = outputs['predictions'].cpu().numpy() # tensor([0, 2, 1, 2, 0, 2, 2, 1, 2], device='cuda:0') 9个tissue的预测类别
        probabilities = outputs['probabilities'].cpu().numpy() # 9*3的tensor，每个tissue的3个类别的概率 （logits --> softmax）
        confidence = outputs['confidence'].cpu().numpy() # 9个tissue的预测置信度 tensor([0.9990, 1.0000, 0.5112, 0.9834, 1.0000, 0.9985, 0.9990, 0.9995, 1.0000], probabilities中的最大值
        # last_hidden_state = outputs['last_hidden_state'].cpu().numpy() # 9*512的tensor，每个tissue的512个token的隐藏状态


        print(f"\n📊 Predictions for 9 tissues:")
        for i, tissue in enumerate(tissue_names):
            pred_class = predictions[i]
            pred_label = label_names[pred_class]
            conf = confidence[i]
            probs = probabilities[i]

            # Get ground truth if available
            gt_col = f"{tissue}_TE_label"
            if gt_col in row:
                gt_label = row[gt_col]
                if isinstance(gt_label, float) and math.isnan(gt_label):
                    continue
                match_emoji = "✅" if pred_label == gt_label else "❌"
                print(f"  {match_emoji} {tissue:25s}: {pred_label:6s} (conf: {conf:.3f}) [GT: {gt_label}]")
            else:
                print(f"  🔹 {tissue:25s}: {pred_label:6s} (conf: {conf:.3f})")

            # Show probability distribution
            print(f"      Probs - Low: {probs[0]:.3f}, Medium: {probs[1]:.3f}, High: {probs[2]:.3f}")

print("\n🎉 All tasks completed!")