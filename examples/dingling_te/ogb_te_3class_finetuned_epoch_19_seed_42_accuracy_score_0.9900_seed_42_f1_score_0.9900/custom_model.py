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
# model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "SpliceBERT-510nt"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """Dataset for 3-class (Low/Medium/High) multi-label TE classification"""

    def __init__(self, **kwargs):
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
            padding="max_length" if self.force_padding else False,
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
        """Inference wrapper - custom implementation to avoid dataset issues"""
        # Step 1: Preprocess inputs directly using tokenizer
        if isinstance(sequence_or_inputs, str):
            # Single sequence
            inputs = self.tokenizer(
                sequence_or_inputs,
                max_length=512,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
        elif isinstance(sequence_or_inputs, list):
            # Multiple sequences
            inputs = self.tokenizer(
                sequence_or_inputs,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        elif isinstance(sequence_or_inputs, dict):
            # Already processed dict
            inputs = sequence_or_inputs
        else:
            # BatchEncoding or other - convert to dict
            inputs = dict(sequence_or_inputs)
        
        # Convert BatchEncoding to dict if needed
        if not isinstance(inputs, dict):
            inputs = dict(inputs)
        
        # Step 2: Move to model device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Step 3: Forward pass
        with torch.no_grad():
            raw_outputs = self.forward(**inputs, **kwargs)
        
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # Step 4: Apply softmax
        probabilities = torch.softmax(logits, dim=-1)

        # Step 5: Get predictions
        predictions = torch.argmax(probabilities, dim=-1)

        # Step 6: Get confidence (max probability for each label)
        confidence, _ = torch.max(probabilities, dim=-1)

        # Step 7: Format outputs
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
    dataset_name_or_path="./",
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
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,
]

# Initialize trainer
trainer = AccelerateTrainer(
    model=model,
    epochs=50,
    learning_rate=2e-5,
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
)
# trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
print('📊 Final Metrics:', metrics)

# === Model Inference ===
print("\n🔮 Starting inference on test samples...")

inference_model = ModelHub.load("ogb_te_3class_finetuned_epoch_50_seed_42_accuracy_score_0.9922_seed_42_f1_score_0.9922")

# Get some test samples
# sample_sequences = datasets['test'].sample(1000).examples
sample_sequences = datasets['valid'].sample(1000).examples

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

        outputs = inference_model.inference(sequence)
        predictions = outputs['predictions'].cpu().numpy()
        probabilities = outputs['probabilities'].cpu().numpy()
        confidence = outputs['confidence'].cpu().numpy()

        print(f"\n📊 Predictions for 9 tissues:")
        for i, tissue in enumerate(tissue_names):
            pred_class = predictions[i]
            pred_label = label_names[pred_class]
            conf = confidence[i]
            probs = probabilities[i]

            # Get ground truth if available
            gt_col = f"{tissue.replace('-', '_')}_TE_label"
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