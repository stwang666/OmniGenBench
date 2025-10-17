# -*- coding: utf-8 -*-
# file: triclass_te_improved.py
# Improved version with overfitting prevention techniques

import os
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from omnigenbench import (
    ModelHub,
    ClassificationMetric,
    AccelerateTrainer,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
    OmniModelForAugmentation
)
model_name_or_path = "yangheng/OmniGenome-52M"

# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

augmentation_model = OmniModelForAugmentation(
    model_name_or_path,
    noise_ratio=0.15,
    max_length=512,
    instance_num=1,
    batch_size=16,
)


class ImprovedTriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """Enhanced dataset with data augmentation capabilities"""

    def __init__(self, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.augment = augment
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def reverse_complement(self, seq):
        """Generate reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join([complement.get(base, 'N') for base in seq[::-1]])
    
    def random_mask(self, seq, mask_prob=0.1):
        """Randomly mask some bases in the sequence"""
        seq_list = list(seq)
        for i in range(len(seq_list)):
            if random.random() < mask_prob:
                seq_list[i] = 'N'
        return ''.join(seq_list)

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
        sequences = [instance["sequence"]]
        
        # # Apply data augmentation (only during training)
        # if hasattr(self, 'augment') and self.augment and random.random() < 0.5:
        #     # 50% chance to use reverse complement
        #     sequence = self.reverse_complement(sequence)
        
        # if hasattr(self, 'augment') and self.augment and random.random() < 0.3:
        #     # 30% chance to apply random masking
        #     sequence = self.random_mask(sequence, mask_prob=0.05)

        sequences.append(self.reverse_complement(sequences[0]))
        sequences.extend(augmentation_model.augment(sequences[0], k=6))
        sequences = [seq.replace(' ', '') for seq in sequences]
        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequences,
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
        ], dtype=torch.long)
        labels = labels.repeat(len(sequences), 1)

        # Properly create a list of dicts, where each dict represents one sequence and its label set
        tokenized_inputs["labels"] = labels
        # Each entry: {col: tensor[i]} so batching works correctly
        batch_size = len(sequences)
        split_tokenized_inputs = []
        for i in range(batch_size):
            item = {key: value[i] if value.shape[0] == batch_size else value for key, value in tokenized_inputs.items()}
            split_tokenized_inputs.append(item)
        return split_tokenized_inputs


class ImprovedOmniModelForTriClassTE(OmniModelForMultiLabelSequenceClassification):
    """Enhanced model with dropout and label smoothing"""

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, 
                 dropout_prob=0.2, label_smoothing=0.1, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.pooler = OmniPooling(self.config)
        
        # Add dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        
        # Use CrossEntropyLoss with label smoothing
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100, 
            reduction="mean",
            label_smoothing=label_smoothing
        )
        
        self.dataset_class = kwargs.pop('dataset_class', ImprovedTriClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with dropout"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        pooled = self.pooler(input_ids, logits)
        
        # Apply dropout
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
        """Prediction with softmax"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "last_hidden_state": last_hidden_state,
        }


# Load datasets with augmentation for training, with caching using dill
import os
import dill

print("📊 Loading datasets...")
cache_path = os.path.join(os.path.dirname(__file__), "triclass_te_dataset.cache.dill")

if os.path.isfile(cache_path):
    print(f"🔄 Loading dataset from cache: {cache_path}")
    with open(cache_path, "rb") as f:
        datasets = dill.load(f)
else:
    datasets = ImprovedTriClassTEDataset.from_hub(
        dataset_name_or_path=os.path.dirname(__file__),
        tokenizer=tokenizer,
        max_length=512,
        force_padding=False,
        max_examples=1000
    )
    with open(cache_path, "wb") as f:
        dill.dump(datasets, f)
    print(f"💾 Dataset cached to: {cache_path}")

# Enable augmentation for training set
if 'train' in datasets:
    datasets['train'].augment = True
    print("✅ Data augmentation enabled for training set")

print("📝 Data loading completed!")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Initialize improved model
print("\n🚀 Initializing improved model with regularization...")
model = ImprovedOmniModelForTriClassTE(
    model_name_or_path,
    tokenizer,
    num_labels=9,
    num_classes=3,
    dropout_prob=0.2,  # Add dropout
    label_smoothing=0.1,  # Add label smoothing
    trust_remote_code=True
)

print(f"  ✅ Dropout: 0.2")
print(f"  ✅ Label smoothing: 0.1")

# Define metrics
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,
]
# Initialize trainer with improved settings
trainer = AccelerateTrainer(
    model=model,
    epochs=15,  # Reduced from 50
    learning_rate=2e-5,  # Reduced from 2e-5
    batch_size=64,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    patience=3,  # Stricter early stopping
    warmup_steps=100,
)

# # Train the model
# print("\n🚂 Starting training...")
# save_path = "ogb_te_3class_improved"

# metrics = trainer.train(
#     path_to_save=save_path,
#     dataset_class=ImprovedTriClassTEDataset,
#     save_on_metric='f1_score',  # Save based on F1 score, not accuracy
#     monitor='max'  # Maximize F1 score
# )

# print('\n📊 Final Metrics:', metrics)
# print(f"\n✅ Model saved to: {save_path}")
# print(f"💡 Training complete! Check if train-val gap has reduced.")


# === Model Inference ===
print("\n🔮 Starting inference on test samples...")

inference_model = ModelHub.load("/home/sw1136/OmniGenBench/examples/dingling_te/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900")

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