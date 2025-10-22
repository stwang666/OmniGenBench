# -*- coding: utf-8 -*-
# file: triclass_te_improved.py
# 改进版训练脚本：防止过拟合
# time: 20/10/2025

import torch
import torch.nn as nn
import math
import numpy as np

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

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """Dataset for 3-class (Low/Medium/High) multi-label TE classification"""

    def __init__(self, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.augment = augment

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

        # 🆕 数据增强：随机反向互补
        if self.augment and torch.rand(1).item() > 0.5:
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            sequence = ''.join([complement.get(base, base) for base in sequence[::-1]])

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
        ], dtype=torch.long)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs


class ImprovedOmniModelForTriClassTE(OmniModelForMultiLabelSequenceClassification):
    """改进的模型：添加正则化防止过拟合"""

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, 
                 dropout_rate=0.3, freeze_layers=0, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels  # 9 tissues
        self.num_classes = num_classes  # 3 classes (Low/Medium/High)
        
        # 🆕 冻结预训练模型的底层（可选）
        if freeze_layers > 0:
            print(f"🔒 冻结预训练模型的前 {freeze_layers} 层")
            for i, layer in enumerate(self.model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.pooler = OmniPooling(self.config)
        
        # 🆕 添加Dropout层防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
        # 🆕 使用多层分类器而非单层
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size // 2, self.num_classes * self.num_labels)
        )
        
        # 🆕 使用Label Smoothing减少过拟合
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean", label_smoothing=0.1)

        self.dataset_class = kwargs.pop('dataset_class', TriClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with dropout"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Get the logits from classifier head
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 🆕 应用dropout
        pooled = self.pooler(input_ids, logits)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        # Reshape logits
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        loss = None
        if labels is not None:
            logits_flat = logits.view(-1, self.num_classes)
            labels_flat = labels.view(-1)
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

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

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

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
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


if __name__ == "__main__":
    # Load datasets
    print("📊 Loading datasets...")
    datasets = TriClassTEDataset.from_hub(
        dataset_name_or_path="examples/dingling_te/",
        tokenizer=tokenizer,
        max_length=512,
        force_padding=False,
        augment=True  # 🆕 启用数据增强
    )

    print("📝 Data loading completed!")
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")

    # 🆕 数据量检查
    train_size = len(datasets['train'])
    if train_size < 500:
        print("⚠️  警告: 训练集样本较少，建议使用更强的正则化或数据增强")

    # Initialize improved model
    print("\n🚀 Initializing improved model...")
    model = ImprovedOmniModelForTriClassTE(
        model_name_or_path,
        tokenizer,
        num_labels=9,
        num_classes=3,
        dropout_rate=0.4,  # 🆕 较大的dropout防止过拟合
        freeze_layers=6,   # 🆕 冻结底层6层，只微调顶层
        trust_remote_code=True,
        dataset_class=TriClassTEDataset
    )

    # Define metrics
    metric_functions = [
        ClassificationMetric(ignore_y=-100).accuracy_score,
        ClassificationMetric(ignore_y=-100, average='macro').f1_score,
    ]

    # 🆕 改进的训练配置
    trainer = AccelerateTrainer(
        model=model,
        epochs=30,  # 🆕 减少epoch数量（从50降到30）
        learning_rate=1e-5,  # 🆕 降低学习率（从2e-5降到1e-5）
        batch_size=8,  # 🆕 减小batch size增加正则化效果
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
        gradient_accumulation_steps=8,  # 🆕 增加梯度累积（有效batch=64）
        weight_decay=0.01,  # 🆕 添加权重衰减
        warmup_steps=100,  # 🆕 学习率预热
        eval_steps=50,  # 🆕 更频繁的验证
        save_strategy="steps",  # 🆕 按步数保存
        save_steps=50,
        load_best_model_at_end=True,  # 🆕 训练结束加载最佳模型
        metric_for_best_model="accuracy_score",  # 🆕 基于验证集accuracy选择最佳模型
        greater_is_better=True,
        save_total_limit=3,  # 🆕 只保留最好的3个checkpoint
    )

    print("\n🎯 开始训练（防过拟合配置）")
    print("="*70)
    print("防过拟合策略:")
    print("  ✅ 1. Dropout rate = 0.4")
    print("  ✅ 2. 冻结底层6层")
    print("  ✅ 3. Label smoothing = 0.1")
    print("  ✅ 4. Weight decay = 0.01")
    print("  ✅ 5. 降低学习率到1e-5")
    print("  ✅ 6. 减少epoch到30")
    print("  ✅ 7. 数据增强（反向互补）")
    print("  ✅ 8. Early stopping（自动加载最佳模型）")
    print("  ✅ 9. 多层分类器")
    print("="*70)

    metrics = trainer.train(
        path_to_save="ogb_te_3class_improved",
        dataset_class=TriClassTEDataset
    )
    
    print('\n📊 Final Metrics:', metrics)
    print("\n✅ 训练完成！检查验证集和测试集的性能差异。")
    print("如果测试集性能仍然很低，运行 diagnose_overfitting.py 进行诊断。")



