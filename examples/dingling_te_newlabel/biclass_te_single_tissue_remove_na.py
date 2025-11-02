# -*- coding: utf-8 -*-
# file: biclass_te_single_tissue.py
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
    Trainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    OmniPooling,
)

#model_name_or_path = "yangheng/OmniGenome-52M"
model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "SpliceBERT-510nt"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


class BiClassTEDataset(OmniDatasetForSequenceClassification):
    """Dataset for 2-class single-label TE classification
    
    继承说明：
    - 继承自 OmniDatasetForSequenceClassification，获得单标签分类数据集的基础功能
    - 重写 prepare_input() 方法以适配 TE 2分类任务的特殊需求
    """

    def __init__(self, **kwargs):
        # 调用父类的初始化方法，继承父类的属性和行为
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        label2idx = {'0.0': 0, '1.0': 1, 'nan': -100}

        # Extract label for single tissue
        FOD_TE_label = label2idx[str(instance["label"])]  
        sequence = instance["Seq"]  # 使用Seq字段
        
        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Single label for binary classification
        tokenized_inputs["labels"] = torch.tensor(FOD_TE_label, dtype=torch.long)

        return tokenized_inputs


class OmniModelForBiClassTESequenceClassification(OmniModelForSequenceClassification):
    """Model for 2-class single-label TE classification"""

    def __init__(self, config_or_model, tokenizer, num_labels=2, *args, **kwargs):
        # 单标签分类：num_labels=2 (0, 1)
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels  # 2 classes (0, 1)
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_labels)
        # 使用CrossEntropyLoss进行单标签分类
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # 🔑 NEW: Store dataset class reference for saving
        self.dataset_class = kwargs.pop('dataset_class', BiClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass for single-label classification"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Get the logits from classifier head
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        logits = self.classifier(self.pooler(input_ids, logits))
        # logits shape: [batch, num_labels] where num_labels=2

        loss = None
        if labels is not None:
            # labels shape: [batch]
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        }

    def predict(self, sequence_or_inputs, **kwargs):
        """Prediction with softmax for binary classification"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)  # [batch, 2]

        # Get predicted class
        predictions = torch.argmax(probabilities, dim=-1)  # [batch]

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

        # Get confidence (max probability)
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
datasets = BiClassTEDataset.from_hub(
    # "examples/dingling_te_newlabel",  # 指定具体的数据目录
    #"examples/dingling_te_newlabel/preprocess_data_Tmean_04/split_ab_train_d",  # 使用A+B训练，D划分的数据
   # "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_04/split_a", #仅使用A划分的数据
    "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE_remove_na", #仅使用ALL_FOD_TE数据
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
model = OmniModelForBiClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=2,  # 2 classes: 0, 1
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

trainer = Trainer(
    model=model,
    epochs=15,  # 增加epochs，因为学习率降低需要更多时间
    learning_rate=2e-5,  # 🔑🔑 大幅降低学习率！从1e-4→5e-6 (降低20倍)
    batch_size=8,  # 每次训练的样本数量
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],  # 仅用于训练后的最终测试，不影响训练过程
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    device="cuda:0",
)
# trainer.save_model(path_to_save="ogb_te_2class_single_tissue_finetuned", dataset_class=BiClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_2class_single_tissue_finetuned_all_fod_te_remove_na_v1.5", dataset_class=BiClassTEDataset)
print('📊 Final Metrics:', metrics)

# === Model Inference ===
print("\n🔮 Starting inference on test samples...")

inference_model = ModelHub.load("ogb_te_2class_single_tissue_finetuned_a")

# Get some test samples
# sample_sequences = datasets['test'].sample(1000).examples
#sample_sequences = datasets['valid'].sample(1000).examples
sample_sequences = datasets['train'].examples[:1]

label_names = ['0', '1']  # 二分类标签

with torch.no_grad():
    for row in sample_sequences:
        sequence = row["Seq"]  # 使用Seq字段，与数据加载时保持一致
        print(f"\n{'='*60}")
        print(f"🧬 Sample ID: {row.get('ID', 'Unknown')}")
        print(f"📏 Sequence length: {len(sequence)} bp")

        outputs = inference_model.inference(sequence, **row)
        prediction = outputs['predictions'].cpu().numpy()  # 预测类别
        probabilities = outputs['probabilities'].cpu().numpy()  # 类别概率
        confidence = outputs['confidence'].cpu().numpy()  # 预测置信度

        print(f"\n📊 Prediction:")
        pred_label = label_names[prediction]
        print(f"  🔹 Predicted class: {pred_label} (confidence: {confidence:.3f})")
        
        # 显示概率分布
        print(f"      Probabilities - 0: {probabilities[0]:.3f}, 1: {probabilities[1]:.3f}")

        # 获取真实标签进行比较
        gt_label = row.get("FOD_TE_label")
        if gt_label is not None and not (isinstance(gt_label, float) and math.isnan(gt_label)):
            gt_label_str = str(gt_label)
            match_emoji = "✅" if pred_label == gt_label_str else "❌"
            print(f"  {match_emoji} Ground truth: {gt_label_str}")

print("\n🎉 All tasks completed!")