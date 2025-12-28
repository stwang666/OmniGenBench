# -*- coding: utf-8 -*-
"""
RNA Translation Efficiency Classification with Interleaved Structure-Aware Attention

This script implements the Interleaved architecture where:
1. Structure attention layers are inserted BETWEEN backbone layers
2. Each structure attention layer inherits Q/K/V from corresponding backbone layer
3. Structure information (SPD or Graphormer) is injected at every layer
4. The output of each structure attention feeds into the next backbone layer

Architecture:
    Input → Embedding → 
    [Backbone Layer 1] → [Structure Attn 1 (SPD/Graphormer)] →
    [Backbone Layer 2] → [Structure Attn 2 (SPD/Graphormer)] →
    ...
    [Backbone Layer N] → [Structure Attn N (SPD/Graphormer)] →
    Pooling → Classifier

Key Features:
- Backbone is NOT frozen by default (full fine-tuning)
- Q/K/V weights are copied from backbone and then fine-tuned
- Structure bias (SPD or Graphormer) is added to attention scores
- Supports both 'spd' and 'graphormer' encoding modes

Cache File Naming Convention:
    The cache files are saved with the following naming pattern:
    {encoding_type}_cache_{split_name}_len{max_length}_dist{max_distance}.pt
    
    Examples:
    - spd_cache_train_len512_dist32.pt
    - graphormer_cache_valid_len512_dist32.pt
    - spd_cache_test_len1024_dist64.pt
    
    The cache files are stored in:
    - {dataset_dir}/.{encoding_type}_cache/ (when dataset is a directory)
    - {dataset_parent_dir}/.{encoding_type}_cache/ (when dataset is a file)

Usage:
    python triclass_seq_tissue_structure_interleaved.py --encoding spd
    python triclass_seq_tissue_structure_interleaved.py --encoding graphormer
"""

import torch
import gc
import warnings
import os
import argparse
import hashlib
from collections import deque
import numpy as np
from tqdm import tqdm

torch.cuda.empty_cache()
gc.collect()

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Set

# Import our modular interleaved attention
from interleaved_structure_attention import (
    SPDStructureBias,
    GraphormerStructureBias,
    InterleavedStructureAttentionLayer,
    InterleavedStructureWrapper,
    create_interleaved_structure_bias,
    count_trainable_parameters,
    EDGE_TYPE_BACKBONE,
    EDGE_TYPE_PAIR,
    EDGE_TYPE_NONE,
)


# ============================================================================
# Part 1: Graph Construction Utilities
# ============================================================================

def dot_bracket_to_edges(structure: str) -> List[Tuple[int, int]]:
    """将点括号表示法转换为配对边列表"""
    edges = []
    stack = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                edges.append((j, i))
    
    return edges


def build_graph_with_edge_types(
    seq_len: int, 
    pair_edges: List[Tuple[int, int]]
) -> Tuple[Dict[int, List[Tuple[int, int]]], Set[Tuple[int, int]]]:
    """构建带有边类型的图"""
    adj = {i: [] for i in range(seq_len)}
    pair_edge_set = set()
    
    for i in range(seq_len - 1):
        adj[i].append((i + 1, EDGE_TYPE_BACKBONE))
        adj[i + 1].append((i, EDGE_TYPE_BACKBONE))
    
    for i, j in pair_edges:
        adj[i].append((j, EDGE_TYPE_PAIR))
        adj[j].append((i, EDGE_TYPE_PAIR))
        pair_edge_set.add((min(i, j), max(i, j)))
    
    return adj, pair_edge_set


def compute_spd_and_edge_paths(
    seq_len: int, 
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32,
    max_path_length: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 SPD 矩阵和边路径矩阵"""
    adj, pair_edge_set = build_graph_with_edge_types(seq_len, pair_edges)
    
    spd_matrix = torch.full((seq_len, seq_len), max_distance, dtype=torch.uint8)
    edge_path_matrix = torch.full(
        (seq_len, seq_len, max_path_length), 
        EDGE_TYPE_NONE, 
        dtype=torch.uint8
    )
    
    for start in range(seq_len):
        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            current_dist = distances[node]
            
            if current_dist >= max_distance:
                continue
            
            for neighbor, edge_type in adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    parent[neighbor] = (node, edge_type)
                    queue.append(neighbor)
        
        for end in range(seq_len):
            if end in distances:
                dist = min(distances[end], max_distance)
                spd_matrix[start, end] = dist
                
                if dist > 0 and dist <= max_path_length:
                    path_edges = []
                    curr = end
                    while parent.get(curr) is not None:
                        p_node, e_type = parent[curr]
                        path_edges.append(e_type)
                        curr = p_node
                    
                    path_edges = path_edges[::-1]
                    
                    for k, e_type in enumerate(path_edges[:max_path_length]):
                        edge_path_matrix[start, end, k] = e_type
    
    return spd_matrix, edge_path_matrix


# ============================================================================
# Part 2: Dataset with Structure Information
# ============================================================================

class OmniDatasetWithStructure(OmniDatasetForSequenceClassification):
    """
    支持 SPD 和 Graphormer 编码的数据集类
    
    根据 encoding_type 返回不同的结构信息:
    - 'spd': 返回 spd_matrix
    - 'graphormer': 返回 spd_matrix + edge_path_matrix
    """
    
    _preloaded_cache = {}
    
    def __init__(
        self, 
        dataset_name_or_path, 
        tokenizer, 
        max_length=None,
        encoding_type='spd',  # 'spd' or 'graphormer'
        max_distance=32,
        max_path_length=8,
        use_cache=True,
        cache_dir=None,
        split_name=None,
        **kwargs
    ):
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        self.max_length = max_length
        self.encoding_type = encoding_type
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.use_cache = use_cache
        
        if cache_dir is None:
            if os.path.isdir(dataset_name_or_path):
                cache_dir = os.path.join(dataset_name_or_path, f".{encoding_type}_cache")
            else:
                cache_dir = os.path.join(os.path.dirname(dataset_name_or_path), f".{encoding_type}_cache")
        self.cache_dir = cache_dir
        
        self._dataset_path = dataset_name_or_path
        self._split_name = split_name
        
        self._cache = None
        if self.use_cache:
            self._cache = self._load_cache()
        
        self._cache_loaded = self._cache is not None
        self._sample_index = 0
        
        if self._cache_loaded:
            print(f"   🚀 Using cached {encoding_type} matrices")
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
        
        if self.use_cache and not self._cache_loaded:
            self._save_cache()
    
    def _get_cache_path(self, split_name=None):
        if split_name is None:
            if hasattr(self, '_split_name') and self._split_name is not None:
                split_name = self._split_name
            else:
                path = self._dataset_path
                path_lower = path.lower()
                if 'train' in path_lower:
                    split_name = 'train'
                elif 'valid' in path_lower:
                    split_name = 'valid'
                elif 'test' in path_lower:
                    split_name = 'test'
                else:
                    split_name = hashlib.md5(path.encode()).hexdigest()[:8]
        
        cache_filename = f"{self.encoding_type}_cache_{split_name}_len{self.max_length}_dist{self.max_distance}.pt"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_cache(self, split_name=None):
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(split_name)
        
        if os.path.exists(cache_path):
            print(f"📂 Loading cache from: {cache_path}")
            try:
                raw_cache = torch.load(cache_path, weights_only=True)
                # 兼容旧格式：旧缓存 value 是 tensor，新缓存 value 是 {'spd': tensor, 'edge': tensor}
                cache = {}
                for k, v in raw_cache.items():
                    if isinstance(v, torch.Tensor):
                        cache[k] = {'spd': v}
                    elif isinstance(v, dict):
                        # 确保 key 一致
                        if 'spd' in v:
                            cache[k] = v
                        elif 'spd_matrix' in v:
                            cache[k] = {'spd': v['spd_matrix'], **{kk: vv for kk, vv in v.items() if kk != 'spd_matrix'}}
                        else:
                            print(f"   ⚠️ Skip cache item {k}: missing 'spd'")
                    else:
                        print(f"   ⚠️ Skip cache item {k}: unsupported type {type(v)}")

                print(f"   ✅ Loaded {len(cache)} cached matrices")
                return cache
            except Exception as e:
                print(f"   ⚠️ Failed to load cache: {e}")
                return None
        return None
    
    def _save_cache(self, split_name=None):
        if not self.use_cache:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_path(split_name)
        
        cache = {}
        for i, item in enumerate(self.data):
            cache_item = {}
            if "spd_matrix" in item:
                cache_item['spd'] = item["spd_matrix"]
            if "edge_path_matrix" in item:
                cache_item['edge'] = item["edge_path_matrix"]
            if cache_item:
                cache[i] = cache_item
        
        if cache:
            print(f"💾 Saving cache to: {cache_path}")
            torch.save(cache, cache_path)
            print(f"   ✅ Saved {len(cache)} matrices")
    
    @classmethod
    def from_hub(cls, dataset_name_or_path, tokenizer, max_length=None,
                 encoding_type='spd', max_distance=32, max_path_length=8,
                 use_cache=True, cache_dir=None, **kwargs):
        if cache_dir is None and os.path.isdir(dataset_name_or_path):
            cache_dir = os.path.join(dataset_name_or_path, f".{encoding_type}_cache")
        
        datasets = {}
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_name_or_path, f"{split}.csv")
            if os.path.exists(split_path):
                print(f"\n📊 Loading {split} split...")
                datasets[split] = cls(
                    split_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    encoding_type=encoding_type,
                    max_distance=max_distance,
                    max_path_length=max_path_length,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                    split_name=split,
                    **kwargs
                )
        
        return datasets
    
    def prepare_input(self, instance, **kwargs):
        labels = -100
        tissue_id = None
        structure = None
        
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
            
            tissue_name = instance.get("tissue", None)
            if tissue_name is not None:
                tissue_id = self.tissue2id.get(tissue_name, 0)
            
            structure = instance.get("structure", None)
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise Exception(
                    "The label must be an integer for sequence classification."
                )
        tokenized_inputs["labels"] = torch.tensor(labels)
        
        if tissue_id is not None:
            tokenized_inputs["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)
        else:
            tokenized_inputs["tissue_id"] = torch.tensor([0], dtype=torch.long)
        
        current_index = self._sample_index
        self._sample_index += 1
        
        # Load from cache or compute
        cache_hit = False
        if self._cache_loaded and self._cache is not None and current_index in self._cache:
            cached = self._cache[current_index]
            if isinstance(cached, torch.Tensor):
                tokenized_inputs["spd_matrix"] = cached
            elif isinstance(cached, dict):
                if 'spd' in cached:
                    tokenized_inputs["spd_matrix"] = cached['spd']
                elif 'spd_matrix' in cached:
                    tokenized_inputs["spd_matrix"] = cached['spd_matrix']
                if self.encoding_type == 'graphormer' and 'edge' in cached:
                    tokenized_inputs["edge_path_matrix"] = cached['edge']
            if self.encoding_type == 'graphormer' and "edge_path_matrix" not in tokenized_inputs and structure is not None:
                # Compute edge paths if not cached
                pair_edges = dot_bracket_to_edges(structure)
                _, edge_paths = compute_spd_and_edge_paths(
                    len(structure), pair_edges, 
                    self.max_distance, self.max_path_length
                )
                # Padding
                seq_len = len(structure)
                if seq_len < self.max_length:
                    padded = torch.full(
                        (self.max_length, self.max_length, self.max_path_length),
                        EDGE_TYPE_NONE, dtype=torch.uint8
                    )
                    padded[:seq_len, :seq_len, :] = edge_paths
                    edge_paths = padded
                tokenized_inputs["edge_path_matrix"] = edge_paths
            cache_hit = "spd_matrix" in tokenized_inputs
        if (not cache_hit) and structure is not None:
            pair_edges = dot_bracket_to_edges(structure)
            seq_len = len(structure)
            
            if self.encoding_type == 'graphormer':
                spd, edge_paths = compute_spd_and_edge_paths(
                    seq_len, pair_edges, 
                    self.max_distance, self.max_path_length
                )
            else:
                # SPD only
                spd, edge_paths = compute_spd_and_edge_paths(
                    seq_len, pair_edges, 
                    self.max_distance, self.max_path_length
                )
            
            # Padding
            if seq_len < self.max_length:
                padded_spd = torch.full(
                    (self.max_length, self.max_length), 
                    self.max_distance, dtype=torch.uint8
                )
                padded_spd[:seq_len, :seq_len] = spd
                spd = padded_spd
                
                if self.encoding_type == 'graphormer':
                    padded_edge = torch.full(
                        (self.max_length, self.max_length, self.max_path_length),
                        EDGE_TYPE_NONE, dtype=torch.uint8
                    )
                    padded_edge[:seq_len, :seq_len, :] = edge_paths
                    edge_paths = padded_edge
            elif seq_len > self.max_length:
                spd = spd[:self.max_length, :self.max_length]
                if self.encoding_type == 'graphormer':
                    edge_paths = edge_paths[:self.max_length, :self.max_length, :]
            
            tokenized_inputs["spd_matrix"] = spd
            if self.encoding_type == 'graphormer':
                tokenized_inputs["edge_path_matrix"] = edge_paths
        elif not cache_hit:
            # Default: linear distance
            spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
            for i in range(self.max_length):
                for j in range(self.max_length):
                    spd[i, j] = min(abs(i - j), self.max_distance)
            tokenized_inputs["spd_matrix"] = spd
            
            if self.encoding_type == 'graphormer':
                tokenized_inputs["edge_path_matrix"] = torch.full(
                    (self.max_length, self.max_length, self.max_path_length),
                    EDGE_TYPE_NONE, dtype=torch.uint8
                )
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        tissue_ids = []
        spd_matrices = []
        edge_path_matrices = []
        
        for item in self.data:
            tissue_ids.append(item.pop("tissue_id", None))
            spd_matrices.append(item.pop("spd_matrix", None))
            edge_path_matrices.append(item.pop("edge_path_matrix", None))
        
        super()._pad_and_truncate(pad_value)
        
        for i, item in enumerate(self.data):
            if tissue_ids[i] is not None:
                tid = tissue_ids[i]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim == 1:
                    item["tissue_id"] = tid
                else:
                    item["tissue_id"] = tid.flatten()[:1]
            else:
                item["tissue_id"] = torch.tensor([0], dtype=torch.long)
            
            if spd_matrices[i] is not None:
                item["spd_matrix"] = spd_matrices[i]
            else:
                spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
                for ii in range(self.max_length):
                    for jj in range(self.max_length):
                        spd[ii, jj] = min(abs(ii - jj), self.max_distance)
                item["spd_matrix"] = spd
            
            if self.encoding_type == 'graphormer':
                if edge_path_matrices[i] is not None:
                    item["edge_path_matrix"] = edge_path_matrices[i]
                else:
                    item["edge_path_matrix"] = torch.full(
                        (self.max_length, self.max_length, self.max_path_length),
                        EDGE_TYPE_NONE, dtype=torch.uint8
                    )


# ============================================================================
# Part 3: Model with Interleaved Structure Attention
# ============================================================================

class OmniModelWithInterleavedStructure(OmniModelForSequenceClassification):
    """
    支持 Interleaved Structure-Aware Attention 的序列分类模型
    
    关键特点:
    1. 每个 Backbone 层后面插入一个 Structure Attention 层
    2. Structure Attention 继承 Backbone 的 Q/K/V 权重
    3. 结构信息通过 bias 注入到 attention 计算中
    4. Backbone 和 Structure Attention 都可训练
    
    Architecture:
        Backbone L1 → Struct Attn 1 → Backbone L2 → Struct Attn 2 → ... → Output
    """
    
    def __init__(
        self, 
        config_or_model, 
        tokenizer, 
        *args,
        encoding_type: str = 'spd',
        max_distance: int = 32,
        max_path_length: int = 8,
        share_structure_bias: bool = False,
        share_qkv: bool = False,
        copy_weights_from_backbone: bool = True,
        freeze_backbone: bool = False,
        **kwargs
    ):
        """
        Args:
            config_or_model: Model name or config
            tokenizer: Tokenizer
            encoding_type: 'spd' or 'graphormer'
            max_distance: Maximum SPD distance
            max_path_length: Maximum path length for edge encoding
            share_structure_bias: If True, all layers share one structure bias (792 params)
            share_qkv: If True, all layers share Q/K/V/O projections (reduces ~12M params)
            copy_weights_from_backbone: If True, copy Q/K/V from backbone (first layer only when share_qkv)
            freeze_backbone: If True, freeze backbone parameters
        """
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithStructure)
        self.encoding_type = encoding_type
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.share_structure_bias = share_structure_bias
        self.share_qkv = share_qkv
        self.copy_weights_from_backbone = copy_weights_from_backbone
        self.freeze_backbone = freeze_backbone
        
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        hidden_size = self.config.hidden_size
        num_attention_heads = getattr(self.config, 'num_attention_heads', 12)
        num_hidden_layers = getattr(self.config, 'num_hidden_layers', 12)
        
        print(f"\n🔧 Initializing Interleaved Structure Attention Model:")
        print(f"   - Encoding type: {encoding_type}")
        print(f"   - Hidden size: {hidden_size}")
        print(f"   - Num heads: {num_attention_heads}")
        print(f"   - Num layers: {num_hidden_layers}")
        print(f"   - Share structure bias: {share_structure_bias}")
        print(f"   - Share Q/K/V/O: {share_qkv}")
        print(f"   - Copy weights from backbone: {copy_weights_from_backbone}")
        print(f"   - Freeze backbone: {freeze_backbone}")
        
        # Create structure bias modules
        if encoding_type == 'spd':
            structure_bias_class = SPDStructureBias
            structure_bias_kwargs = {'max_distance': max_distance}
        elif encoding_type == 'graphormer':
            structure_bias_class = GraphormerStructureBias
            structure_bias_kwargs = {
                'max_distance': max_distance,
                'max_path_length': max_path_length,
                'use_spatial': True,
                'use_edge': True
            }
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # Get backbone encoder layers
        self.backbone_layers = self._get_encoder_layers()
        
        # Create structure bias module(s)
        # 当 share_structure_bias=True 时，所有层共享同一个 bias module（只有 792 参数）
        if share_structure_bias:
            # 注册为模型属性以确保参数被正确追踪
            self.shared_structure_bias = structure_bias_class(
                num_heads=num_attention_heads,
                **structure_bias_kwargs
            )
            structure_biases = [self.shared_structure_bias] * len(self.backbone_layers)
            print(f"   ✅ Shared SPD Bias: {sum(p.numel() for p in self.shared_structure_bias.parameters())} params")
        else:
            structure_biases = nn.ModuleList([
                structure_bias_class(
                    num_heads=num_attention_heads,
                    **structure_bias_kwargs
                )
                for _ in range(len(self.backbone_layers))
            ])
            # Register as attribute for proper parameter tracking
            self.structure_biases = structure_biases
        
        # Create interleaved structure attention layers
        self.structure_attention_layers = nn.ModuleList()
        
        # 🚀 优化：当 share_qkv=True 时，所有层共享同一组 Q/K/V/O 投影
        shared_projections = None
        if share_qkv:
            self.shared_q_proj = nn.Linear(hidden_size, hidden_size)
            self.shared_k_proj = nn.Linear(hidden_size, hidden_size)
            self.shared_v_proj = nn.Linear(hidden_size, hidden_size)
            self.shared_out_proj = nn.Linear(hidden_size, hidden_size)
            shared_projections = {
                'q_proj': self.shared_q_proj,
                'k_proj': self.shared_k_proj,
                'v_proj': self.shared_v_proj,
                'out_proj': self.shared_out_proj
            }
            # 如果需要从 backbone 复制权重，只从第一层复制
            if copy_weights_from_backbone and len(self.backbone_layers) > 0:
                self._init_shared_projections_from_backbone(self.backbone_layers[0])
            
            shared_qkv_params = sum(p.numel() for p in [
                self.shared_q_proj, self.shared_k_proj, 
                self.shared_v_proj, self.shared_out_proj
            ] for p in p.parameters())
            print(f"   ✅ Shared Q/K/V/O: {shared_qkv_params:,} params (vs {shared_qkv_params * num_hidden_layers:,} if not shared)")
        
        for i, backbone_layer in enumerate(self.backbone_layers):
            # 当共享 QKV 时，不需要从 backbone 初始化（已经在上面初始化过了）
            init_from = backbone_layer if (copy_weights_from_backbone and not share_qkv) else None
            
            bias_module = structure_biases[i]
            
            struct_attn = InterleavedStructureAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                structure_bias_module=bias_module,
                dropout=0.1,
                init_from_backbone=init_from,
                shared_projections=shared_projections
            )
            self.structure_attention_layers.append(struct_attn)
        
        # Get structure keys
        self.structure_keys = ['spd_matrix']
        if encoding_type == 'graphormer':
            self.structure_keys.append('edge_path_matrix')
        
        # Tissue embedding
        self.tissue_embed_dim = hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,
            embedding_dim=self.tissue_embed_dim
        )
        
        # Classifier
        self.classifier = nn.Linear(
            hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            print("   ❄️ Freezing backbone parameters")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Print parameter counts
        self._print_param_counts()
    
    def _get_encoder_layers(self) -> List[nn.Module]:
        """Find encoder layers in backbone"""
        layers = []
        
        for attr_path in [
            'encoder.layer',
            'layers',
            'transformer.layer',
            'transformer.layers',
        ]:
            obj = self.model
            found = True
            for attr in attr_path.split('.'):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    found = False
                    break
            
            if found and hasattr(obj, '__len__') and len(obj) > 0:
                layers = list(obj)
                print(f"   Found encoder layers at: model.{attr_path}")
                break
        
        if not layers:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    first = module[0]
                    if hasattr(first, 'attention') or hasattr(first, 'self_attn'):
                        layers = list(module)
                        print(f"   Found encoder layers at: {name}")
                        break
        
        if not layers:
            raise ValueError("Could not find encoder layers in backbone model.")
        
        return layers
    
    def _init_shared_projections_from_backbone(self, backbone_layer: nn.Module):
        """
        Initialize shared Q/K/V/O projections from a backbone attention layer.
        Similar to InterleavedStructureAttentionLayer._init_from_backbone
        """
        copied = False
        
        # Try BERT-style
        if hasattr(backbone_layer, 'attention'):
            attn = backbone_layer.attention
            if hasattr(attn, 'self'):
                self_attn = attn.self
                if hasattr(self_attn, 'query'):
                    self.shared_q_proj.load_state_dict(self_attn.query.state_dict())
                    self.shared_k_proj.load_state_dict(self_attn.key.state_dict())
                    self.shared_v_proj.load_state_dict(self_attn.value.state_dict())
                    if hasattr(attn, 'output') and hasattr(attn.output, 'dense'):
                        self.shared_out_proj.load_state_dict(attn.output.dense.state_dict())
                    copied = True
                    print("   ✅ Copied shared Q/K/V/O from backbone (BERT-style)")
        
        # Try ESM/RoBERTa style
        if not copied and hasattr(backbone_layer, 'self_attn'):
            self_attn = backbone_layer.self_attn
            if hasattr(self_attn, 'q_proj'):
                self.shared_q_proj.load_state_dict(self_attn.q_proj.state_dict())
                self.shared_k_proj.load_state_dict(self_attn.k_proj.state_dict())
                self.shared_v_proj.load_state_dict(self_attn.v_proj.state_dict())
                if hasattr(self_attn, 'out_proj'):
                    self.shared_out_proj.load_state_dict(self_attn.out_proj.state_dict())
                copied = True
                print("   ✅ Copied shared Q/K/V/O from backbone (ESM-style)")
        
        if not copied:
            print("   ⚠️ Could not copy shared Q/K/V/O from backbone, using fresh init")
    
    def _print_param_counts(self):
        """Print parameter statistics"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        struct_params = sum(p.numel() for p in self.structure_attention_layers.parameters())
        if hasattr(self, 'structure_biases'):
            struct_params += sum(p.numel() for p in self.structure_biases.parameters())
        
        backbone_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        print(f"\n📊 Parameter Statistics:")
        print(f"   - Total parameters: {total:,}")
        print(f"   - Trainable parameters: {trainable:,}")
        print(f"   - Structure attention params: {struct_params:,}")
        print(f"   - Backbone trainable: {backbone_trainable:,}")
    
    def _interleaved_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **structure_kwargs
    ) -> torch.Tensor:
        """
        Perform interleaved forward pass through backbone and structure attention layers.
        
        The key is that structure attention output becomes the INPUT to the next backbone layer.
        
        优化：当 share_structure_bias=True 时，预计算一次 structure bias 然后复用于所有层
        """
        # Prepare extended attention mask for backbone layers
        # Most backbone layers expect 4D attention mask
        if attention_mask.dim() == 2:
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
        else:
            extended_mask = attention_mask
        
        # 🚀 优化：当共享 bias 时，预计算一次 structure bias
        precomputed_bias = None
        if self.share_structure_bias and hasattr(self, 'shared_structure_bias'):
            struct_input = {k: v for k, v in structure_kwargs.items() 
                          if k in self.structure_keys}
            precomputed_bias = self.shared_structure_bias(**struct_input)
        
        for backbone_layer, struct_attn in zip(
            self.backbone_layers,
            self.structure_attention_layers
        ):
            # Run backbone layer
            try:
                layer_output = backbone_layer(hidden_states, extended_mask)
            except TypeError:
                # Some layers might have different signatures
                try:
                    layer_output = backbone_layer(hidden_states, attention_mask=extended_mask)
                except:
                    layer_output = backbone_layer(hidden_states)
            
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
            
            # Run structure attention layer
            if precomputed_bias is not None:
                # 使用预计算的 bias（共享模式）
                hidden_states = struct_attn(
                    hidden_states,
                    attention_mask,
                    precomputed_bias=precomputed_bias
                )
            else:
                # 每层单独计算 bias
                struct_input = {k: v for k, v in structure_kwargs.items() 
                              if k in self.structure_keys}
                hidden_states = struct_attn(
                    hidden_states,
                    attention_mask,
                    **struct_input
                )
        
        return hidden_states
    
    def forward(self, **inputs):
        """Forward pass with interleaved structure attention"""
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        
        # Extract structure information
        structure_kwargs = {}
        for key in self.structure_keys:
            if key in inputs:
                structure_kwargs[key] = inputs.pop(key)
        
        attention_mask = inputs.get("attention_mask", None)
        
        # Get embeddings from backbone
        # We need to manually call the embedding layer
        input_ids = inputs.get("input_ids")
        
        # Get embeddings
        if hasattr(self.model, 'embeddings'):
            hidden_states = self.model.embeddings(input_ids, attention_mask=attention_mask)
        elif hasattr(self.model, 'embed_tokens'):
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            # Fallback: use backbone's forward to get hidden states
            # and then use our interleaved layers
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[0]  # Use first layer output
        
        # Move structure tensors to correct device
        device = hidden_states.device
        for key in structure_kwargs:
            if structure_kwargs[key] is not None:
                structure_kwargs[key] = structure_kwargs[key].to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Run interleaved forward
        last_hidden_state = self._interleaved_forward(
            hidden_states,
            attention_mask,
            **structure_kwargs
        )
        
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        
        # Pooling
        pooled_state = self.pooler(inputs, last_hidden_state)
        
        # Tissue embedding
        if tissue_id is not None:
            if tissue_id.device != pooled_state.device:
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
        
        combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            logits_flat = logits.view(-1, self.config.num_labels)
            labels_flat = labels.view(-1)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = self.loss_fn(logits_flat, labels_flat)
        
        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }


# ============================================================================
# Part 4: Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Interleaved Structure-Aware Attention Training')
    parser.add_argument('--encoding', type=str, default='spd', choices=['spd', 'graphormer'],
                       help='Structure encoding type (default: spd)')
    parser.add_argument('--model', type=str, default='yangheng/OmniGenome-52M',
                       help='Backbone model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--max_distance', type=int, default=32,
                       help='Maximum SPD distance')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone parameters')
    parser.add_argument('--share_bias', action='store_true',
                       help='Share structure bias across layers (792 params total)')
    parser.add_argument('--share_qkv', action='store_true',
                       help='Share Q/K/V/O projections across layers (reduces ~12M params)')
    parser.add_argument('--no_copy_weights', action='store_true',
                       help='Do not copy Q/K/V weights from backbone')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Interleaved Structure-Aware Attention Training")
    print("=" * 70)
    print(f"Encoding: {args.encoding}")
    print(f"Model: {args.model}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Share bias: {args.share_bias} (792 params if True)")
    print(f"Share Q/K/V/O: {args.share_qkv} (saves ~12M params if True)")
    print(f"Copy weights: {not args.no_copy_weights}")
    print("=" * 70)
    
    # Label mapping
    label2id = {"0": 0, "1": 1, "2": 2}
    
    # Initialize tokenizer
    tokenizer = OmniTokenizer.from_pretrained(args.model)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    datasets = OmniDatasetWithStructure.from_hub(
        "/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10",
        tokenizer=tokenizer,
        max_length=args.max_length,
        encoding_type=args.encoding,
        max_distance=args.max_distance,
        label2id=label2id,
    )
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")
    
    # Verify data
    sample = datasets["train"][0]
    print(f"\n📋 Sample data keys: {list(sample.keys())}")
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = OmniModelWithInterleavedStructure(
        args.model,
        tokenizer,
        num_labels=len(label2id),
        encoding_type=args.encoding,
        max_distance=args.max_distance,
        share_structure_bias=args.share_bias,
        share_qkv=args.share_qkv,
        copy_weights_from_backbone=not args.no_copy_weights,
        freeze_backbone=args.freeze_backbone,
        dataset_class=OmniDatasetWithStructure,
    )
    
    # Training
    metric_functions = [
        ClassificationMetric().accuracy_score,
        ClassificationMetric(average='macro').f1_score
    ]
    
    trainer = AccelerateTrainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
        gradient_accumulation_steps=4,
        device=torch.device("cuda:0"),
        monitor='valid_accuracy_score',
        load_best_model_at_end=True,
    )
    
    save_name = f"ogb_te_3class_finetuned_52M_interleaved_{args.encoding}"
    if args.freeze_backbone:
        save_name += "_frozen"
    
    print(f"\n🎓 Starting training...")
    print(f"   Save path: {save_name}")
    
    metrics = trainer.train(
        path_to_save=save_name,
        dataset_class=OmniDatasetWithStructure
    )
    
    print('\n✅ Final Metrics:', metrics)


if __name__ == "__main__":
    main()
