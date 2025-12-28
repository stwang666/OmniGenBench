# -*- coding: utf-8 -*-
# file: te_bpp_model.py
# time: 12:00 01/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Translation Efficiency Prediction with BPP Features - OmniGenBench Implementation
==================================================================================

This module implements a translation efficiency (TE) prediction model that integrates
Base Pairing Probability (BPP) structural features with sequence embeddings from 
genomic foundation models.

Architecture Overview:
    1. Custom Dataset: Computes BPP matrices using ViennaRNA during data loading
    2. Custom Model: Fuses BPP features with transformer embeddings via CNN + MLP
    3. End-to-end Training: Jointly optimizes all components for TE prediction

Key Features:
    - Efficient BPP computation with caching
    - Flexible sequence length handling (padding/truncation to 512)
    - Multi-scale CNN for BPP matrix processing
    - Attention-based feature fusion
    - Compatible with all OmniGenBench foundation models
    - Multi-model testing support

Usage:
    # Train with default model
    python te_bpp_model.py
    
    # Test all available models
    python te_bpp_model.py --test-all-models
    
    # Train with specific model
    python te_bpp_model.py --model yangheng/OmniGenome-52M

Author: YANG, HENG (杨恒)
Contact: hy345@exeter.ac.uk
Homepage: https://yangheng95.github.io
"""

import os
import json
import argparse
import warnings
import traceback
from typing import Dict, List, Optional, Union, Tuple

# ============================================================================
# Available Models for Testing
# ============================================================================

AVAILABLE_GFMS = [
    'yangheng/OmniGenome-52M',
    # 'yangheng/OmniGenome-186M',
    # 'yangheng/OmniGenome-v1.5',
    # 'zhihan1996/DNABERT-2-117M',
    # 'LongSafari/hyenadna-large-1m-seqlen-hf',
    # 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    # 'kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16',
    # 'multimolecule/rnafm',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

try:
    import RNA  # ViennaRNA package
except ImportError:
    raise ImportError(
        "ViennaRNA is required for BPP computation. "
        "Install it with: conda install -c bioconda viennarna"
    )

from omnigenbench import (
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    AccelerateTrainer,
    ClassificationMetric,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Part 1: Custom Dataset with BPP Feature Engineering
# ============================================================================

class TEDatasetWithBPP(OmniDatasetForSequenceClassification):
    """
    Custom dataset for translation efficiency prediction with BPP features.
    
    This dataset extends OmniDatasetForSequenceClassification to compute
    Base Pairing Probability (BPP) matrices for RNA sequences using ViennaRNA.
    The BPP matrix captures secondary structure information that influences
    translation efficiency.
    
    Features:
        - Automatic sequence padding/truncation to fixed length
        - Efficient BPP computation with error handling
        - Compatible with standard OmniGenBench workflows
        - Supports both local files and HuggingFace datasets
    
    Args:
        dataset_name_or_path: Path to dataset file(s) or dataset name
        tokenizer: OmniTokenizer instance for sequence encoding
        max_length: Maximum sequence length (default: 512)
        **kwargs: Additional arguments (label2id, etc.)
    
    Example:
        >>> tokenizer = OmniTokenizer("yangheng/PlantRNA-FM")
        >>> dataset = TEDatasetWithBPP(
        ...     "train.json",
        ...     tokenizer=tokenizer,
        ...     max_length=512,
        ...     label2id={"0": 0, "1": 1}
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'bpp_matrix', 'labels'])
    """
    
    def __init__(
        self, 
        dataset_name_or_path: Union[str, List[str]], 
        tokenizer: OmniTokenizer, 
        max_length: int = 512, 
        **kwargs
    ):
        """Initialize dataset with BPP computation capability."""
        self.max_length = max_length
        self._bpp_cache = {}  # Cache for computed BPP matrices
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # Update metadata
        self.metadata.update({
            "features": "sequence + BPP matrix",
            "bpp_method": "ViennaRNA partition function",
            "max_length": max_length
        })
    
    def compute_bpp_matrix(self, sequence: str) -> np.ndarray:
        """
        Compute Base Pairing Probability matrix using ViennaRNA.
        
        This method uses the partition function approach to calculate the
        probability that each pair of nucleotides forms a base pair in the
        RNA secondary structure ensemble.
        
        Algorithm:
            1. Create ViennaRNA fold compound for the sequence
            2. Compute partition function (thermodynamic ensemble)
            3. Extract base pairing probabilities from the ensemble
            4. Build symmetric probability matrix
        
        Args:
            sequence: RNA sequence string (A, U, G, C, N)
            
        Returns:
            Symmetric BPP matrix of shape (len(sequence), len(sequence))
            Matrix[i,j] = probability that positions i and j are base-paired
            
        Note:
            - Returns zero matrix if computation fails (handles invalid sequences)
            - Uses float32 for memory efficiency
            - ViennaRNA uses 1-based indexing internally
        """
        # Check cache first
        if sequence in self._bpp_cache:
            return self._bpp_cache[sequence]
        
        try:
            # Create fold compound and compute partition function
            fc = RNA.fold_compound(sequence)
            fc.pf()  # Compute partition function
            
            # Get base pair probability matrix
            bpp = fc.bpp()
            
            # Convert to numpy array (ViennaRNA uses 1-based indexing)
            seq_len = len(sequence)
            bpp_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
            
            for i in range(1, seq_len + 1):
                for j in range(i + 1, seq_len + 1):
                    prob = bpp[i][j]
                    if prob > 0:
                        bpp_matrix[i-1, j-1] = prob
                        bpp_matrix[j-1, i-1] = prob  # Symmetric matrix
            
            # Cache the result
            if len(self._bpp_cache) < 10000:  # Limit cache size
                self._bpp_cache[sequence] = bpp_matrix
            
            return bpp_matrix
            
        except Exception as e:
            # Handle invalid sequences gracefully
            warnings.warn(f"BPP computation failed for sequence: {e}", RuntimeWarning)
            return np.zeros((len(sequence), len(sequence)), dtype=np.float32)
    
    def prepare_input(self, instance: Union[str, Dict], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare a single training instance with BPP features.
        
        This method overrides the parent class to add BPP matrix computation
        to the standard sequence tokenization pipeline.
        
        Processing Steps:
            1. Extract sequence and label from instance
            2. Convert T->U for RNA sequences
            3. Pad or truncate sequence to max_length
            4. Compute BPP matrix
            5. Tokenize sequence
            6. Return combined features
        
        Args:
            instance: Either a sequence string or dict with 'sequence' and 'label'
            **kwargs: Tokenization arguments (padding, truncation, etc.)
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs (tensor)
                - attention_mask: Attention mask (tensor)
                - bpp_matrix: BPP matrix (tensor, shape: [max_length, max_length])
                - labels: Classification label (tensor)
        
        Example:
            >>> dataset = TEDatasetWithBPP(...)
            >>> instance = {"sequence": "AUGCAUGC...", "label": 1}
            >>> features = dataset.prepare_input(instance)
            >>> print(features['bpp_matrix'].shape)  # torch.Size([512, 512])
        """
        # Extract sequence and label
        if isinstance(instance, str):
            sequence = instance
            labels = -100  # Ignore label for inference
        elif isinstance(instance, dict):
            sequence = instance.get("sequence", instance.get("seq", ""))
            label = instance.get("label", instance.get("labels"))
            labels = label if label is not None else -100
        else:
            raise ValueError(f"Unknown instance format: {type(instance)}")
        
        # RNA preprocessing: Convert T to U
        sequence = sequence.replace("T", "U").replace("t", "u").upper()
        
        # Pad or truncate sequence to max_length
        original_length = len(sequence)
        if original_length > self.max_length:
            sequence = sequence[:self.max_length]
        elif original_length < self.max_length:
            # Pad with 'N' (unknown nucleotide)
            sequence = sequence + "N" * (self.max_length - original_length)
        
        # Compute BPP matrix for the sequence
        bpp_matrix = self.compute_bpp_matrix(sequence)
        
        # Pad BPP matrix if needed (should not be necessary after sequence padding)
        if bpp_matrix.shape[0] < self.max_length:
            padded_bpp = np.zeros((self.max_length, self.max_length), dtype=np.float32)
            padded_bpp[:bpp_matrix.shape[0], :bpp_matrix.shape[1]] = bpp_matrix
            bpp_matrix = padded_bpp
        
        # Tokenize sequence using the foundation model's tokenizer
        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "max_length"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Remove batch dimension (will be added by DataLoader)
        for key in tokenized_inputs:
            tokenized_inputs[key] = tokenized_inputs[key].squeeze(0)
        
        # Add BPP matrix to the input features
        tokenized_inputs["bpp_matrix"] = torch.tensor(bpp_matrix, dtype=torch.float32)
        
        # Process labels
        if labels != -100:
            if isinstance(labels, str):
                labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise ValueError("Label must be an integer for sequence classification")
        
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return tokenized_inputs


# ============================================================================
# Part 2: Custom Model with BPP Feature Fusion
# ============================================================================

class BPPProcessor(nn.Module):
    """
    CNN-based processor for BPP matrices.
    
    This module uses a multi-scale convolutional architecture to extract
    structural features from the BPP matrix. The architecture progressively
    reduces spatial dimensions while increasing feature depth.
    
    Architecture:
        Input: [batch, 1, 512, 512] (single-channel BPP matrix)
        Conv1: [batch, 32, 512, 512] -> [batch, 32, 256, 256] (pool)
        Conv2: [batch, 64, 256, 256] -> [batch, 64, 128, 128] (pool)
        Conv3: [batch, 128, 128, 128] -> [batch, 128, 1, 1] (global pool)
        Output: [batch, 128] (feature vector)
    
    Args:
        output_dim: Dimension of output feature vector (default: 128)
    """
    
    def __init__(self, output_dim: int = 128):
        """Initialize BPP processor."""
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Layer 1: Initial feature extraction
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 512 -> 256
            
            # Layer 2: Intermediate features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 -> 128
            
            # Layer 3: High-level features
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            
            # Global average pooling: reduce to feature vector
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    
    def forward(self, bpp_matrix: torch.Tensor) -> torch.Tensor:
        """
        Process BPP matrix to extract structural features.
        
        Args:
            bpp_matrix: BPP matrix [batch, seq_len, seq_len]
            
        Returns:
            Feature vector [batch, output_dim]
        """
        # Add channel dimension if needed
        if bpp_matrix.dim() == 3:
            bpp_matrix = bpp_matrix.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
        
        features = self.cnn(bpp_matrix)  # [batch, output_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch, output_dim]
        
        return features


class FeatureFusion(nn.Module):
    """
    Multi-layer perceptron for fusing sequence and BPP features.
    
    This module combines sequence embeddings from the foundation model with
    structural features from the BPP processor using a gated fusion mechanism.
    
    Args:
        seq_dim: Dimension of sequence features (e.g., 768 for BERT-base)
        bpp_dim: Dimension of BPP features (default: 128)
        hidden_dim: Hidden layer dimension (default: 512)
        output_dim: Output dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self, 
        seq_dim: int, 
        bpp_dim: int = 128, 
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """Initialize fusion module."""
        super().__init__()
        
        # Projection layers for each modality
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.bpp_proj = nn.Linear(bpp_dim, hidden_dim)
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(seq_dim + bpp_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    
    def forward(
        self, 
        seq_features: torch.Tensor, 
        bpp_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse sequence and BPP features.
        
        Args:
            seq_features: Sequence features [batch, seq_dim]
            bpp_features: BPP features [batch, bpp_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Project to common space
        seq_proj = self.seq_proj(seq_features)  # [batch, hidden_dim]
        bpp_proj = self.bpp_proj(bpp_features)  # [batch, hidden_dim]
        
        # Compute gating weights
        concat_features = torch.cat([seq_features, bpp_features], dim=1)
        gate_weights = self.gate(concat_features)  # [batch, hidden_dim]
        
        # Gated fusion
        fused = gate_weights * seq_proj + (1 - gate_weights) * bpp_proj
        
        # Final MLP
        output = self.fusion_mlp(fused)
        
        return output


class TEModelWithBPP(OmniModelForSequenceClassification):
    """
    Translation Efficiency prediction model with BPP feature integration.
    
    This model extends OmniModelForSequenceClassification to incorporate
    RNA secondary structure information via Base Pairing Probability matrices.
    It combines a pre-trained genomic foundation model with a custom BPP
    processor for improved translation efficiency prediction.
    
    Architecture:
        1. Transformer Encoder: Process sequence to get embeddings
        2. BPP Processor: Extract structural features from BPP matrix
        3. Feature Fusion: Combine sequence and structure features
        4. Classification Head: Predict translation efficiency
    
    Args:
        config_or_model: Path to pre-trained model or model name
        tokenizer: OmniTokenizer instance
        num_labels: Number of classification labels (default: 2 for binary)
        bpp_dim: BPP feature dimension (default: 128)
        fusion_dim: Fusion layer output dimension (default: 256)
        **kwargs: Additional arguments for parent class
    
    Example:
        >>> tokenizer = OmniTokenizer("yangheng/PlantRNA-FM")
        >>> model = TEModelWithBPP(
        ...     "yangheng/PlantRNA-FM",
        ...     tokenizer=tokenizer,
        ...     num_labels=2,
        ...     label2id={"0": 0, "1": 1}
        ... )
        >>> # Use with AccelerateTrainer
        >>> trainer = AccelerateTrainer(model=model, ...)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config_or_model: str,
        tokenizer: OmniTokenizer,
        num_labels: int = 2,
        bpp_dim: int = 128,
        fusion_dim: int = 256,
        **kwargs
    ):
        """Initialize model with BPP fusion."""
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, **kwargs)
        
        # Get hidden size from the foundation model
        hidden_size = self.model.config.hidden_size
        
        # BPP matrix processor
        self.bpp_processor = BPPProcessor(output_dim=bpp_dim)
        
        # Feature fusion module
        self.fusion = FeatureFusion(
            seq_dim=hidden_size,
            bpp_dim=bpp_dim,
            hidden_dim=512,
            output_dim=fusion_dim,
            dropout=0.1
        )
        
        # Replace the original classifier
        self.classifier = nn.Linear(fusion_dim, num_labels)
        
        # Initialize new components
        self._init_custom_layers()
    
    def _init_custom_layers(self):
        """Initialize weights for custom layers using Xavier initialization."""
        for module in [self.bpp_processor, self.fusion, self.classifier]:
            for layer in module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bpp_matrix: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with BPP feature fusion.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            bpp_matrix: BPP matrix [batch, seq_len, seq_len]
            labels: Ground truth labels [batch]
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - loss: Cross-entropy loss (if labels provided)
                - logits: Classification logits [batch, num_labels]
                - sequence_features: Sequence embeddings [batch, hidden_size]
                - bpp_features: BPP features [batch, bpp_dim]
        """
        # Process sequence with foundation model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract sequence representation (use [CLS] token)
        sequence_features = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Process BPP matrix
        if bpp_matrix is not None:
            bpp_features = self.bpp_processor(bpp_matrix)  # [batch, bpp_dim]
        else:
            # If no BPP matrix provided, use zero features
            batch_size = sequence_features.size(0)
            bpp_features = torch.zeros(
                batch_size, 128, 
                device=sequence_features.device,
                dtype=sequence_features.dtype
            )
        
        # Fuse features
        fused_features = self.fusion(sequence_features, bpp_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "sequence_features": sequence_features,
            "bpp_features": bpp_features,
            "last_hidden_state": outputs.last_hidden_state,
        }


# ============================================================================
# Part 3: Training and Evaluation Functions
# ============================================================================

def create_demo_dataset(output_dir: str = "translation_efficiency_prediction"):
    """
    Create a demo dataset for testing.
    
    This function generates synthetic RNA sequences with random labels for
    demonstration purposes. In practice, you should replace this with real data.
    
    Args:
        output_dir: Directory to save dataset files
        
    Returns:
        Path to the output directory
    """
    print(f"[INFO] Creating demo dataset in '{output_dir}/'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic sequences
    nucleotides = ['A', 'U', 'G', 'C']
    np.random.seed(42)
    
    def generate_sequence(length: int = 480) -> str:
        """Generate a random RNA sequence."""
        return ''.join(np.random.choice(nucleotides, length))
    
    # Create train/valid/test splits
    datasets = {
        "train": 100,
        "valid": 20,
        "test": 20
    }
    
    for split, num_samples in datasets.items():
        data = []
        for i in range(num_samples):
            seq = generate_sequence()
            label = i % 2  # Alternate labels for balance
            data.append({"sequence": seq, "label": label})
        
        # Save to JSON Lines format
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"  [SUCCESS] Created {split}.json with {num_samples} samples")
    
    return output_dir


def train_te_model(
    model_name: str = "yangheng/PlantRNA-FM",
    train_file: Optional[str] = None,
    valid_file: Optional[str] = None,
    test_file: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 8,
    epochs: int = 20,
    learning_rate: float = 2e-5,
    output_dir: str = "te_bpp_finetuned",
    use_demo: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Train a translation efficiency prediction model with BPP features.
    
    This is a high-level function that handles the entire training pipeline:
        1. Load or create dataset
        2. Initialize tokenizer and model
        3. Setup trainer
        4. Train and evaluate
        5. Save model
    
    Args:
        model_name: Name or path of the foundation model
        train_file: Path to training data file
        valid_file: Path to validation data file
        test_file: Path to test data file
        max_length: Maximum sequence length
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        output_dir: Directory to save the trained model
        use_demo: Whether to create and use demo data
        **kwargs: Additional arguments for trainer
        
    Returns:
        Dictionary of evaluation metrics
        
    Example:
        >>> results = train_te_model(
        ...     model_name="yangheng/PlantRNA-FM",
        ...     train_file="data/train.json",
        ...     valid_file="data/valid.json",
        ...     test_file="data/test.json",
        ...     batch_size=16,
        ...     epochs=30
        ... )
        >>> print(f"Test F1 Score: {results['f1_score']:.4f}")
    """
    print("=" * 80)
    print("Translation Efficiency Prediction with BPP Features")
    print("=" * 80)
    
    # Configuration
    print(f"\n[CONFIG] Model: {model_name}")
    print(f"[CONFIG] Max Length: {max_length}")
    print(f"[CONFIG] Batch Size: {batch_size}")
    print(f"[CONFIG] Epochs: {epochs}")
    print(f"[CONFIG] Learning Rate: {learning_rate}")
    
    # Step 1: Initialize tokenizer
    print(f"\n[STEP 1] Loading tokenizer...")
    tokenizer = OmniTokenizer.from_pretrained(model_name)
    print(f"  [SUCCESS] Tokenizer loaded")
    
    # Step 2: Load or create datasets
    print(f"\n[STEP 2] Loading datasets...")
    
    label2id = {"0": 0, "1": 1}
    
    if use_demo or not all([train_file, valid_file, test_file]):
        print(f"  [INFO] Creating demo dataset...")
        dataset_dir = create_demo_dataset()
        train_file = f"{dataset_dir}/train.json"
        valid_file = f"{dataset_dir}/valid.json"
        test_file = f"{dataset_dir}/test.json"
    
    train_dataset = TEDatasetWithBPP(
        dataset_name_or_path=train_file,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    
    valid_dataset = TEDatasetWithBPP(
        dataset_name_or_path=valid_file,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    
    test_dataset = TEDatasetWithBPP(
        dataset_name_or_path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=label2id,
    )
    
    print(f"  [SUCCESS] Loaded datasets:")
    print(f"    - Train: {len(train_dataset)} samples")
    print(f"    - Valid: {len(valid_dataset)} samples")
    print(f"    - Test: {len(test_dataset)} samples")
    
    # Step 3: Initialize model
    print(f"\n[STEP 3] Initializing model with BPP fusion...")
    model = TEModelWithBPP(
        config_or_model=model_name,
        tokenizer=tokenizer,
        num_labels=2,
        label2id=label2id,
        dataset_class=TEDatasetWithBPP,  # For inference
    )
    print(f"  [SUCCESS] Model initialized")
    print(f"    - Architecture: Transformer + BPP CNN + Feature Fusion")
    
    # Step 4: Setup trainer
    print(f"\n[STEP 4] Setting up trainer...")
    
    metric_functions = [
        ClassificationMetric().f1_score,
        ClassificationMetric().accuracy,
        ClassificationMetric().precision,
        ClassificationMetric().recall,
    ]
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        test_dataset=test_dataset,
        compute_metrics=metric_functions,
        model_save_path=output_dir,
        batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        patience=5,
        **kwargs
    )
    
    print(f"  [SUCCESS] Trainer configured")
    
    # Step 5: Train model
    print(f"\n[STEP 5] Training model...")
    print(f"  [INFO] This may take a while depending on your hardware...")
    
    metrics = trainer.train()
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"  [INFO] Model saved to: {output_dir}/")
    
    # Step 6: Evaluate on test set
    print(f"\n[STEP 6] Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    print(f"\n[RESULTS] Test Set Performance:")
    for metric_name, value in test_results.items():
        print(f"  - {metric_name}: {value:.4f}")
    
    # Step 7: Demo inference
    print(f"\n[STEP 7] Demo inference...")
    model.eval()
    sample_seq = "AUGCAUGCAUGCAUGCAUGC" * 20
    
    with torch.no_grad():
        outputs = model.inference({"sequence": sample_seq})
        prediction = outputs.get("predictions", [0])[0]
        confidence = outputs.get("confidence", 0.5)
        
        print(f"  [DEMO] Sample sequence: {sample_seq[:50]}...")
        print(f"  [DEMO] Prediction: {'High TE' if prediction == 1 else 'Low TE'}")
        print(f"  [DEMO] Confidence: {confidence:.4f}")
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Training pipeline completed!")
    print(f"{'='*80}\n")
    
    return test_results


# ============================================================================
# Part 4: Multi-Model Testing Functions
# ============================================================================

def test_all_models(
    models: List[str] = None,
    use_demo: bool = True,
    epochs: int = 5,
    batch_size: int = 4
) -> Dict[str, Dict]:
    """
    Test all available models with BPP features sequentially.
    
    Args:
        models: List of model names to test. If None, uses AVAILABLE_GFMS
        use_demo: Whether to use demo data for testing
        epochs: Number of training epochs for each model
        batch_size: Batch size for training
        
    Returns:
        Dictionary mapping model names to their test results
    """
    if models is None:
        models = AVAILABLE_GFMS
    
    print("\n" + "=" * 80)
    print("🧪 MULTI-MODEL TESTING FOR TE PREDICTION WITH BPP FEATURES")
    print("=" * 80)
    print(f"\n📋 Models to test: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    all_results = {}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"📍 Testing Model {i}/{len(models)}: {model_name}")
        print(f"{'='*80}")
        
        model_short_name = model_name.split('/')[-1]
        output_dir = f"te_bpp_{model_short_name}"
        
        try:
            # Train model
            results = train_te_model(
                model_name=model_name,
                use_demo=use_demo,
                batch_size=batch_size,
                epochs=epochs,
                output_dir=output_dir
            )
            
            all_results[model_name] = {
                "status": "SUCCESS",
                "metrics": results,
                "output_dir": output_dir
            }
            print(f"\n✅ Model {model_name} completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Model {model_name} failed with error:")
            print(f"   {str(e)}")
            traceback.print_exc()
            all_results[model_name] = {
                "status": "FAILED",
                "error": str(e)
            }
    
    # Print summary
    print_test_summary(all_results)
    
    return all_results


def print_test_summary(results: Dict[str, Dict]):
    """Print a summary of all model test results."""
    print("\n" + "=" * 80)
    print("📊 MULTI-MODEL TEST SUMMARY (BPP Features)")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r.get("status") == "SUCCESS")
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    
    print(f"\n{'Model':<60} {'Status':<15}")
    print("-" * 75)
    
    for model_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_emoji} {model_name:<58} {status:<15}")
        
        # Print metrics for successful models
        if status == "SUCCESS" and "metrics" in result:
            metrics = result["metrics"]
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      - {metric_name}: {value:.4f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Part 5: Main Entry Point
# ============================================================================

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Translation Efficiency Prediction with BPP Features - Multi-Model Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default model (PlantRNA-FM) using demo data
    python te_bpp_model.py
    
    # Test all available models
    python te_bpp_model.py --test-all-models
    
    # Train with specific model
    python te_bpp_model.py --model yangheng/OmniGenome-52M
    
    # Test all models with custom epochs
    python te_bpp_model.py --test-all-models --epochs 10 --batch-size 8
    
    # Train with real data files
    python te_bpp_model.py --train-file train.json --valid-file valid.json --test-file test.json
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yangheng/PlantRNA-FM",
        help="Model name or path (default: yangheng/PlantRNA-FM)"
    )
    parser.add_argument(
        "--test-all-models", "-a",
        action="store_true",
        help="Test all models in AVAILABLE_GFMS list"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--use-demo",
        action="store_true",
        default=True,
        help="Use demo data for testing (default: True)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training data file"
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        default=None,
        help="Path to validation data file"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test data file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        print("\n📋 Available Models for Testing:")
        print("-" * 50)
        for i, model in enumerate(AVAILABLE_GFMS, 1):
            print(f"   {i}. {model}")
        print("\nTo enable more models, edit the AVAILABLE_GFMS list in te_bpp_model.py")
        return
    
    print("\n" + "=" * 80)
    print("🧬 Translation Efficiency Prediction with BPP Features")
    print("=" * 80)
    
    if args.test_all_models:
        # Test all available models
        results = test_all_models(
            models=AVAILABLE_GFMS,
            use_demo=args.use_demo,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        # Check if using real data or demo data
        use_demo = args.use_demo
        if args.train_file and args.valid_file and args.test_file:
            use_demo = False
        
        # Train single model
        model_short_name = args.model.split('/')[-1]
        output_dir = args.output_dir or f"te_bpp_{model_short_name}"
        
    results = train_te_model(
            model_name=args.model,
            train_file=args.train_file,
            valid_file=args.valid_file,
            test_file=args.test_file,
            use_demo=use_demo,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=output_dir
    )
    
        print("\n📊 Training Results:")
    for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"   - {metric}: {value:.4f}")
    
        # Test inference
        print("\n🔮 Testing inference with trained model...")
    from omnigenbench import ModelHub
    
        model = ModelHub.load(output_dir)
    test_sequences = [
        "AUGCAUGCAUGCAUGC" * 30,
        "AAAUUUGGGCCCAAAUUUGGGCCC" * 20,
    ]
    
    for i, seq in enumerate(test_sequences, 1):
        outputs = model.inference({"sequence": seq})
            pred = outputs.get("predictions", [0])[0]
            conf = outputs.get("confidence", 0.5)
            print(f"   Sequence {i}: {'High TE' if pred == 1 else 'Low TE'} (confidence: {conf:.4f})")
    
    print("\n🎉 All tasks completed!")


if __name__ == "__main__":
    main()
