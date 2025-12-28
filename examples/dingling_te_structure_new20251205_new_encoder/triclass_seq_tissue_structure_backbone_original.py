# -*- coding: utf-8 -*-
"""
RNA Translation Efficiency Classification with Direct Backbone Modification

This script implements structure-aware attention by directly patching the backbone
model's attention layers, adding structure bias to attention scores without adding
extra layers.

Key Advantages:
1. Minimal parameter overhead: ~800-1000 parameters per layer (vs 1.05M for extra attention layers)
2. True structure integration: Structure bias is added inside each attention layer
3. Efficient: No extra forward passes through additional layers
4. Backward compatible: Model works normally when structure info is not provided

Architecture Comparison:
    Previous (Interleaved):
        Backbone L1 → Extra Attn L1 → Backbone L2 → Extra Attn L2 → ...
        Parameters per layer: ~1.05M (Q/K/V/O projections)
    
    New (Direct Backbone Modification):
        [Backbone L1 + Structure Bias] → [Backbone L2 + Structure Bias] → ...
        Parameters per layer: ~800 (SPD) or ~1000 (Graphormer)

Usage:
    # SPD encoding
    python triclass_seq_tissue_structure_backbone.py --encoding_type spd
    
    # Graphormer encoding (SPD + Edge)
    python triclass_seq_tissue_structure_backbone.py --encoding_type graphormer
    
    # Pairing encoding (simplest)
    python triclass_seq_tissue_structure_backbone.py --encoding_type pairing

Reference:
    Ying et al. "Do Transformers Really Perform Badly for Graph Representation?" NeurIPS 2021
"""

import torch
import gc
import warnings
import os
import argparse
from collections import deque
import numpy as np
from tqdm import tqdm

torch.cuda.empty_cache()
gc.collect()

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)
import torch.nn as nn
from typing import Optional, List, Dict, Tuple

# Import the structure-aware backbone module
from structure_aware_backbone import (
    patch_backbone_with_structure,
    StructureAwareBackbone,
    SPDStructureBias,
    GraphormerStructureBias,
    PairingStructureBias,
    dot_bracket_to_edges,
    compute_shortest_path_distance,
    compute_spd_and_edge_paths,
    EDGE_TYPE_NONE,
    EDGE_TYPE_BACKBONE,
    EDGE_TYPE_PAIR,
)


# ============================================================================
# Part 1: Dataset with Structure Information
# ============================================================================

class OmniDatasetWithStructure(OmniDatasetForSequenceClassification):
    """
    Dataset that computes and caches structure information (SPD, edge paths, pairing).
    
    Supports multiple encoding types:
    - 'spd': Shortest Path Distance only
    - 'graphormer': SPD + Edge paths
    - 'pairing': Simple paired/unpaired matrix
    """
    
    # Tissue name to ID mapping
    tissue_mapping = {
        "anthers": 0, "flag_leaf": 1, "florets": 2, "grain_06DAA": 3,
        "grain_20DAA": 4, "lemma": 5, "roots_seedling": 6, 
        "seedling_shoot": 7, "whole_spikelet": 8
    }
    
    def __init__(
        self,
        encoding_type: str = 'spd',
        max_distance: int = 32,
        max_path_length: int = 8,
        cache_dir: str = None,
        **kwargs
    ):
        self.encoding_type = encoding_type
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.cache_dir = cache_dir
        
        # Cache for structure matrices
        self._structure_cache: Dict[str, Dict] = {}
        
        super().__init__(**kwargs)
    
    def _get_structure_info(self, structure: str) -> Dict[str, torch.Tensor]:
        """
        Compute or retrieve cached structure information.
        """
        # Check cache
        cache_key = f"{structure}_{self.encoding_type}"
        if cache_key in self._structure_cache:
            return self._structure_cache[cache_key]
        
        # Compute structure info
        seq_len = len(structure)
        pair_edges = dot_bracket_to_edges(structure)
        
        result = {}
        
        if self.encoding_type == 'spd':
            spd_matrix = compute_shortest_path_distance(
                seq_len, pair_edges, self.max_distance
            )
            result['spd_matrix'] = spd_matrix
            
        elif self.encoding_type == 'graphormer':
            spd_matrix, edge_path_matrix = compute_spd_and_edge_paths(
                seq_len, pair_edges, self.max_distance, self.max_path_length
            )
            result['spd_matrix'] = spd_matrix
            result['edge_path_matrix'] = edge_path_matrix
            
        elif self.encoding_type == 'pairing':
            # Create pairing matrix
            pairing_matrix = torch.zeros(seq_len, seq_len, dtype=torch.float32)
            for i, j in pair_edges:
                pairing_matrix[i, j] = 1.0
                pairing_matrix[j, i] = 1.0
            result['pairing_matrix'] = pairing_matrix
        
        # Cache the result
        self._structure_cache[cache_key] = result
        
        return result
    
    def _pad_structure_matrices(
        self,
        structure_info: Dict[str, torch.Tensor],
        seq_len: int,
        max_length: int
    ) -> Dict[str, torch.Tensor]:
        """Pad structure matrices to max_length."""
        result = {}
        
        for key, matrix in structure_info.items():
            if key == 'spd_matrix':
                # Pad with max_distance (indicating no connection)
                padded = torch.full((max_length, max_length), self.max_distance, dtype=matrix.dtype)
                padded[:seq_len, :seq_len] = matrix
                result[key] = padded
                
            elif key == 'edge_path_matrix':
                # Pad with EDGE_TYPE_NONE
                path_len = matrix.shape[-1]
                padded = torch.zeros((max_length, max_length, path_len), dtype=matrix.dtype)
                padded[:seq_len, :seq_len, :] = matrix
                result[key] = padded
                
            elif key == 'pairing_matrix':
                # Pad with 0 (unpaired)
                padded = torch.zeros((max_length, max_length), dtype=matrix.dtype)
                padded[:seq_len, :seq_len] = matrix
                result[key] = padded
        
        return result
    
    def prepare_input(self, instance: dict, **kwargs) -> dict:
        """Prepare a single instance with structure information."""
        # Get sequence and structure
        sequence = instance.get("sequence", instance.get("seq", ""))
        structure = instance.get("structure", instance.get("ss", ""))
        
        # Tokenize
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            tokenized = self.tokenizer(
                sequence,
                padding='max_length',
                truncation=True,
                max_length=getattr(self, 'max_length', 512),
                return_tensors='pt'
            )
            inputs = {k: v.squeeze(0) for k, v in tokenized.items()}
        else:
            inputs = {}
        
        # Get structure information
        if structure:
            seq_len = len(structure)
            max_length = getattr(self, 'max_length', 512)
            
            structure_info = self._get_structure_info(structure)
            padded_info = self._pad_structure_matrices(structure_info, seq_len, max_length)
            inputs.update(padded_info)
        
        # Get tissue ID
        tissue = instance.get("tissue", instance.get("tissue_type", None))
        if tissue and tissue in self.tissue_mapping:
            inputs["tissue_id"] = torch.tensor(self.tissue_mapping[tissue], dtype=torch.long)
        
        # Get label
        label = instance.get("label", instance.get("labels", None))
        if label is not None:
            if hasattr(self, 'label2id') and self.label2id:
                label = self.label2id.get(str(label), label)
            inputs["labels"] = torch.tensor(label, dtype=torch.long)
        
        return inputs
    
    def __getitem__(self, index):
        """Get a single item with structure information."""
        instance = self.data[index]
        return self.prepare_input(instance)


# ============================================================================
# Part 2: Model with Structure-Aware Backbone
# ============================================================================

class OmniModelWithStructureBackbone(OmniModelForSequenceClassification):
    """
    Sequence classification model with structure-aware backbone.
    
    This model patches the backbone's attention layers to add structure bias
    directly to the attention scores, without adding extra layers.
    
    Key Features:
    1. Minimal parameters: ~800-1000 per layer instead of ~1.05M
    2. Deep integration: Structure info is used in every layer
    3. Backward compatible: Works without structure info
    """
    
    def __init__(
        self,
        config_or_model,
        tokenizer,
        *args,
        encoding_type: str = 'spd',
        max_distance: int = 32,
        max_path_length: int = 8,
        share_bias_across_layers: bool = False,
        layers_to_patch: Optional[List[int]] = None,
        **kwargs
    ):
        # Pop our custom args before passing to parent
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithStructure)
        self._encoding_type = encoding_type
        self._max_distance = max_distance
        self._max_path_length = max_path_length
        self._share_bias = share_bias_across_layers
        self._layers_to_patch = layers_to_patch
        
        # Initialize parent
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        
        # Get backbone hidden size and num_heads
        hidden_size = self.config.hidden_size
        num_heads = getattr(self.config, 'num_attention_heads', 12)
        
        # Patch the backbone with structure-aware attention
        print(f"\n🔧 Patching backbone with structure-aware attention...")
        self.patched_backbone = patch_backbone_with_structure(
            backbone=self.model,
            structure_bias_type=encoding_type,
            num_heads=num_heads,
            max_distance=max_distance,
            max_path_length=max_path_length,
            share_across_layers=share_bias_across_layers,
            layers_to_patch=layers_to_patch
        )
        
        # Tissue embedding layer
        self.tissue_embed_dim = hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,  # 9 tissue types
            embedding_dim=self.tissue_embed_dim
        )
        
        # Classifier with tissue features
        self.classifier = nn.Linear(
            hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        # Print parameter summary
        self._print_parameter_summary()
    
    def _print_parameter_summary(self):
        """Print summary of model parameters."""
        backbone_params = sum(p.numel() for p in self.model.parameters())
        structure_params = sum(
            p.numel() for p in self.patched_backbone.structure_bias_modules.parameters()
        )
        tissue_params = sum(p.numel() for p in self.tissue_embedding.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        print(f"\n📊 Parameter Summary:")
        print(f"   - Backbone: {backbone_params:,}")
        print(f"   - Structure bias: {structure_params:,}")
        print(f"   - Tissue embedding: {tissue_params:,}")
        print(f"   - Classifier: {classifier_params:,}")
        print(f"   - Total trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, **inputs):
        """
        Forward pass with structure-aware attention.
        """
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        
        # Pop structure matrices
        spd_matrix = inputs.pop("spd_matrix", None)
        edge_path_matrix = inputs.pop("edge_path_matrix", None)
        pairing_matrix = inputs.pop("pairing_matrix", None)
        
        # Set structure info in patched backbone
        structure_kwargs = {}
        if spd_matrix is not None:
            structure_kwargs['spd_matrix'] = spd_matrix.to(self.device)
        if edge_path_matrix is not None:
            structure_kwargs['edge_path_matrix'] = edge_path_matrix.to(self.device)
        if pairing_matrix is not None:
            structure_kwargs['pairing_matrix'] = pairing_matrix.to(self.device)
        
        if structure_kwargs:
            self.patched_backbone.set_structure_info(**structure_kwargs)
        else:
            self.patched_backbone.clear_structure_info()
        
        # Get backbone output
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        
        # Clear structure info after forward
        self.patched_backbone.clear_structure_info()
        
        # Apply dropout and activation
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        
        # Pooling
        pooled_state = self.pooler(inputs, last_hidden_state)
        
        # Tissue embedding
        if tissue_id is not None:
            tissue_id = tissue_id.to(pooled_state.device)
            if tissue_id.ndim > 1:
                tissue_id = tissue_id.squeeze(-1)
            tissue_embed = self.tissue_embedding(tissue_id)
        else:
            batch_size = last_hidden_state.shape[0]
            tissue_embed = torch.zeros(
                batch_size, self.tissue_embed_dim,
                device=last_hidden_state.device
            )
        
        # Concatenate features
        combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
        
        # Classification
        logits = self.classifier(combined_features)
        logits = self.softmax(logits)
        
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        
        return outputs


# ============================================================================
# Part 3: Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RNA TE classifier with structure-aware backbone"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", type=str, default="yangheng/OmniGenome-52M",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--encoding_type", type=str, default="spd",
        choices=["spd", "graphormer", "pairing"],
        help="Type of structure encoding"
    )
    parser.add_argument(
        "--max_distance", type=int, default=32,
        help="Maximum distance for SPD encoding"
    )
    parser.add_argument(
        "--max_path_length", type=int, default=8,
        help="Maximum path length for edge encoding"
    )
    parser.add_argument(
        "--share_bias", action="store_true",
        help="Share structure bias across all layers"
    )
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    
    # Data configuration
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="ogb_te_3class_finetuned_backbone"
    )
    
    return parser.parse_args()


# ============================================================================
# Part 4: Main Training Script
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 60)
    print("Structure-Aware Backbone Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Encoding type: {args.encoding_type}")
    print(f"Max distance: {args.max_distance}")
    print(f"Share bias: {args.share_bias}")
    print("=" * 60)
    
    # Label mapping
    label2id = {"0": 0, "1": 1, "2": 2}
    
    # Initialize tokenizer
    tokenizer = OmniTokenizer.from_pretrained(args.model)
    
    # Load datasets
    print("\n📊 Loading datasets with structure computation...")
    datasets = OmniDatasetWithStructure.from_hub(
        args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        encoding_type=args.encoding_type,
        max_distance=args.max_distance,
        max_path_length=args.max_path_length,
        label2id=label2id,
    )
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")
    
    # Verify structure data
    sample = datasets["train"][0]
    print(f"\n📋 Sample data keys: {list(sample.keys())}")
    
    if args.encoding_type == 'spd' and 'spd_matrix' in sample:
        spd = sample["spd_matrix"]
        print(f"  - spd_matrix shape: {spd.shape}")
        print(f"  - spd_matrix dtype: {spd.dtype}")
        
    elif args.encoding_type == 'graphormer':
        if 'spd_matrix' in sample:
            print(f"  - spd_matrix shape: {sample['spd_matrix'].shape}")
        if 'edge_path_matrix' in sample:
            print(f"  - edge_path_matrix shape: {sample['edge_path_matrix'].shape}")
            
    elif args.encoding_type == 'pairing' and 'pairing_matrix' in sample:
        print(f"  - pairing_matrix shape: {sample['pairing_matrix'].shape}")
    
    # Initialize model
    print("\n🔧 Initializing model with structure-aware backbone...")
    model = OmniModelWithStructureBackbone(
        args.model,
        tokenizer,
        num_labels=len(label2id),
        encoding_type=args.encoding_type,
        max_distance=args.max_distance,
        max_path_length=args.max_path_length,
        share_bias_across_layers=args.share_bias,
        dataset_class=OmniDatasetWithStructure,
    )
    
    # Training configuration
    metric_functions = [
        ClassificationMetric().accuracy_score,
        ClassificationMetric(average='macro').f1_score
    ]
    
    trainer = AccelerateTrainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        monitor='valid_accuracy_score',
        load_best_model_at_end=True,
    )
    
    # Add encoding type to output directory name
    output_dir = f"{args.output_dir}_{args.encoding_type}"
    
    print(f"\n🎓 Starting training...")
    print(f"   Output directory: {output_dir}")
    
    metrics = trainer.train(
        path_to_save=output_dir,
        dataset_class=OmniDatasetWithStructure
    )
    
    # Print learned structure bias parameters
    print("\n📐 Learned structure bias parameters:")
    for i, bias_module in enumerate(model.patched_backbone.structure_bias_modules):
        if hasattr(bias_module, 'spatial_bias'):
            print(f"  Layer/Module {i}:")
            for d in range(min(6, args.max_distance + 1)):
                mean_bias = bias_module.spatial_bias[:, d].mean().item()
                print(f"    Distance {d}: {mean_bias:.4f}")
    
    print(f'\n✅ Final Metrics: {metrics}')
