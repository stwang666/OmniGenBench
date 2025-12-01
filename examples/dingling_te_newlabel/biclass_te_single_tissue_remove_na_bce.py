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

model_name_or_path = "yangheng/OmniGenome-52M"
#model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "SpliceBERT-510nt"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# 1. Initialize tokenizer
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
    """Model for 2-class single-label TE classification using BCEWithLogitsLoss"""
    
    def __init__(self, config_or_model, tokenizer, num_labels=2, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        # 二分类：输出1个值（而不是2个）
        self.classifier = torch.nn.Linear(self.config.hidden_size, 1)
        # 使用BCEWithLogitsLoss
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.sigmoid = torch.nn.Sigmoid()
    
    def loss_function(self, logits, labels):
        """Custom loss function for BCEWithLogitsLoss"""
        # logits shape: [batch, 1]
        # labels shape: [batch]
        logits = logits.view(-1)  # [batch, 1] -> [batch]
        labels = labels.view(-1).float()  # [batch] -> [batch], 转为float
        
        # 过滤掉 ignore_index=-100 的样本
        valid_mask = labels != -100
        if valid_mask.any():
            valid_logits = logits[valid_mask]
            valid_labels = labels[valid_mask]
            loss = self.loss_fn(valid_logits, valid_labels)
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss
    
    def forward(self, **inputs):
        """Override forward to use 1 output"""
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)  # [batch, 1]
        
        # 计算损失（如果需要）
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels)
        
        outputs = {
            "loss": loss,
            "logits": logits,  # [batch, 1]
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs
    
    def predict(self, sequence_or_inputs, **kwargs):
        """Override predict for BCEWithLogitsLoss: use sigmoid > 0.5 instead of argmax"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        
        logits = raw_outputs["logits"]  # [batch, 1]
        last_hidden_state = raw_outputs["last_hidden_state"]
        
        # 使用 sigmoid 得到类别1的概率，然后根据阈值0.5判断类别
        probabilities = self.sigmoid(logits)  # [batch, 1]
        predictions = (probabilities > 0.5).long().view(-1)  # [batch]
        
        outputs = {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "last_hidden_state": last_hidden_state,
        }
        
        return outputs

# 2. Initialize model with PlantRNA-FM for plant-specific features
model = OmniModelForBiClassTESequenceClassification(
    model_name_or_path,
    tokenizer=tokenizer,
    num_labels=2,  # Binary: High TE vs Low TE
    trust_remote_code=True,
)

# 3. Prepare your dataset
# Format: [{"sequence": "AUGC...", "label": 1}, ...]
datasets = BiClassTEDataset.from_hub(
    "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/new_data",
    tokenizer=tokenizer,
    max_length=512
    #balanced_sampling=True  # Oversample minority class
)

# Option 1: Weighted loss with BCEWithLogitsLoss
# Estimate class weights from your dataset distribution
#pos_weight = torch.tensor([2.0])  # >1.0 increases weight on positive class
loss_fn = torch.nn.BCEWithLogitsLoss()#

metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,  # 宏平均F1：(TP) / (TP + 0.5*(FP+FN))，对所有类别取平均
    ClassificationMetric(ignore_y=-100).classification_report,
]
# 4. Train
trainer = Trainer(
  model=model,
#   train_dataset=datasets,
  epochs=15,
  batch_size=16,
  loss_fn=loss_fn,
  train_dataset=datasets["train"],
  eval_dataset=datasets["valid"],
  test_dataset=datasets["test"],  # 仅用于训练后的最终测试，不影响训练过程
  compute_metrics=metric_functions,
  gradient_accumulation_steps=4,
)
# metrics = trainer.train()
metrics = trainer.train(path_to_save="ogb_te_2class_single_tissue_finetuned_all_fod_te_remove_na_bcewithlogitsloss_52M_new_data", dataset_class=BiClassTEDataset)
print('📊 Final Metrics:', metrics)
