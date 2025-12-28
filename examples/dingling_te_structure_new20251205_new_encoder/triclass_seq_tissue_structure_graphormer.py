# -*- coding: utf-8 -*-
"""
RNA Translation Efficiency Classification with Full Graphormer-style Encoding

This script implements both Spatial Encoding and Edge Encoding following
the Graphormer paper (NeurIPS 2021).

1. Spatial Encoding (空间编码):
   - 计算任意两点间的最短路径距离 (SPD)
   - 为每个距离值分配一个可学习的 scalar bias b_φ(vi, vj)

2. Edge Encoding (边编码):
   - 找到两点间的最短路径
   - 考虑路径上每条边的类型 (骨架边 vs 氢键边)
   - 学习边类型的 embedding
   - 计算路径上边 embedding 的平均值作为额外的 bias c_ij

公式:
    A_ij = (h_i W_Q)(h_j W_K)^T / √d + b_φ(vi, vj) + c_ij

其中:
    - b_φ(vi, vj) 是 Spatial Encoding bias (基于最短路径距离)
    - c_ij = Average(Emb_backbone, Emb_pair, ...) 是 Edge Encoding bias

Reference: https://github.com/microsoft/Graphormer
Ying et al. "Do Transformers Really Perform Badly for Graph Representation?" NeurIPS 2021
"""

# Step 1: Data Preparation
import torch
import gc
import warnings
import os
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
import re


# ============================================================================
# Part 1: Graph Construction and Path Finding
# ============================================================================

# 边类型定义
EDGE_TYPE_BACKBONE = 0  # 骨架边 (i -> i+1)
EDGE_TYPE_PAIR = 1      # 氢键配对边
EDGE_TYPE_NONE = 2      # 不连通 (用于 padding)


def dot_bracket_to_edges(structure: str) -> List[Tuple[int, int]]:
    """
    将点括号表示法转换为配对边列表
    """
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
    """
    构建带有边类型的图
    
    Args:
        seq_len: 序列长度
        pair_edges: 配对边列表 [(i, j), ...]
    
    Returns:
        adj: 邻接表，adj[node] = [(neighbor, edge_type), ...]
        pair_edge_set: 配对边集合 (用于快速查询)
    """
    adj = {i: [] for i in range(seq_len)}
    pair_edge_set = set()
    
    # 添加骨架边 (backbone edges): i <-> i+1
    for i in range(seq_len - 1):
        adj[i].append((i + 1, EDGE_TYPE_BACKBONE))
        adj[i + 1].append((i, EDGE_TYPE_BACKBONE))
    
    # 添加氢键配对边 (hydrogen bond edges)
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
    """
    计算 SPD 矩阵和最短路径上的边类型
    
    Args:
        seq_len: 序列长度
        pair_edges: 配对边列表
        max_distance: 最大距离 (SPD)
        max_path_length: 最大路径长度 (用于 Edge Encoding)
    
    Returns:
        spd_matrix: (seq_len, seq_len) 最短路径距离矩阵
        edge_path_matrix: (seq_len, seq_len, max_path_length) 
                         边类型矩阵，edge_path_matrix[i,j,k] 表示从 i 到 j 
                         的最短路径上第 k 条边的类型
                         值: 0=backbone, 1=pair, 2=none(padding)
    """
    adj, pair_edge_set = build_graph_with_edge_types(seq_len, pair_edges)
    
    # 初始化 (使用 uint8 节省内存)
    spd_matrix = torch.full((seq_len, seq_len), max_distance, dtype=torch.uint8)
    edge_path_matrix = torch.full(
        (seq_len, seq_len, max_path_length), 
        EDGE_TYPE_NONE, 
        dtype=torch.uint8
    )
    
    # BFS 从每个节点计算最短路径
    for start in range(seq_len):
        # distances[node] = 距离
        distances = {start: 0}
        # parent[node] = (parent_node, edge_type)
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
        
        # 填充 SPD 矩阵和边路径矩阵
        for end in range(seq_len):
            if end in distances:
                dist = min(distances[end], max_distance)
                spd_matrix[start, end] = dist
                
                # 回溯路径，记录边类型
                if dist > 0 and dist <= max_path_length:
                    path_edges = []
                    curr = end
                    while parent.get(curr) is not None:
                        p_node, e_type = parent[curr]
                        path_edges.append(e_type)
                        curr = p_node
                    
                    # 反转路径 (从 start 到 end)
                    path_edges = path_edges[::-1]
                    
                    # 填充边路径矩阵
                    for k, e_type in enumerate(path_edges[:max_path_length]):
                        edge_path_matrix[start, end, k] = e_type
    
    return spd_matrix, edge_path_matrix


def batch_compute_spd_and_edges(
    structures: List[str], 
    max_length: int,
    max_distance: int = 32,
    max_path_length: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算 SPD 和边路径矩阵
    """
    spd_matrices = []
    edge_path_matrices = []
    
    for struct in structures:
        seq_len = len(struct)
        pair_edges = dot_bracket_to_edges(struct)
        
        spd, edge_paths = compute_spd_and_edge_paths(
            seq_len, pair_edges, max_distance, max_path_length
        )
        
        # Padding
        if seq_len < max_length:
            padded_spd = torch.full(
                (max_length, max_length), 
                max_distance, 
                dtype=torch.uint8
            )
            padded_spd[:seq_len, :seq_len] = spd
            spd = padded_spd
            
            padded_edge = torch.full(
                (max_length, max_length, max_path_length), 
                EDGE_TYPE_NONE, 
                dtype=torch.uint8
            )
            padded_edge[:seq_len, :seq_len, :] = edge_paths
            edge_paths = padded_edge
        elif seq_len > max_length:
            spd = spd[:max_length, :max_length]
            edge_paths = edge_paths[:max_length, :max_length, :]
        
        spd_matrices.append(spd)
        edge_path_matrices.append(edge_paths)
    
    return torch.stack(spd_matrices), torch.stack(edge_path_matrices)


def compute_edge_paths_only(
    seq_len: int, 
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32,
    max_path_length: int = 8
) -> torch.Tensor:
    """
    只计算边路径矩阵（当 SPD 已经缓存时使用）
    
    Args:
        seq_len: 序列长度
        pair_edges: 配对边列表
        max_distance: 最大距离
        max_path_length: 最大路径长度
    
    Returns:
        edge_path_matrix: (seq_len, seq_len, max_path_length)
    """
    adj, pair_edge_set = build_graph_with_edge_types(seq_len, pair_edges)
    
    edge_path_matrix = torch.full(
        (seq_len, seq_len, max_path_length), 
        EDGE_TYPE_NONE, 
        dtype=torch.uint8
    )
    
    # BFS 从每个节点计算最短路径
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
        
        # 填充边路径矩阵
        for end in range(seq_len):
            if end in distances:
                dist = distances[end]
                
                # 回溯路径，记录边类型
                if dist > 0 and dist <= max_path_length:
                    path_edges = []
                    curr = end
                    while parent.get(curr) is not None:
                        p_node, e_type = parent[curr]
                        path_edges.append(e_type)
                        curr = p_node
                    
                    # 反转路径
                    path_edges = path_edges[::-1]
                    
                    # 填充
                    for k, e_type in enumerate(path_edges[:max_path_length]):
                        edge_path_matrix[start, end, k] = e_type
    
    return edge_path_matrix


# ============================================================================
# Part 2: Spatial Encoding Module
# ============================================================================

class SpatialEncodingBias(nn.Module):
    """
    Graphormer Spatial Encoding
    
    为每个最短路径距离值分配一个可学习的 scalar bias
    
    公式:
        spatial_bias[h, d] = head h 对距离 d 的 bias
    """
    
    def __init__(self, num_heads: int, max_distance: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # 可学习的距离 bias: (num_heads, max_distance + 1)
        self.spatial_bias = nn.Parameter(
            torch.zeros(num_heads, max_distance + 1)
        )
        
        self._init_bias()
    
    def _init_bias(self):
        """距离越近，初始 bias 越大"""
        with torch.no_grad():
            for d in range(self.max_distance + 1):
                if d == 0:
                    self.spatial_bias[:, d] = 0.5
                elif d == 1:
                    self.spatial_bias[:, d] = 0.3
                elif d == 2:
                    self.spatial_bias[:, d] = 0.2
                else:
                    self.spatial_bias[:, d] = 0.1 * np.exp(-0.1 * (d - 2))
    
    def forward(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spd_matrix: (batch_size, seq_len, seq_len)
        
        Returns:
            attention_bias: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = spd_matrix.shape
        device = spd_matrix.device
        
        spd_clamped = spd_matrix.clamp(0, self.max_distance)
        spatial_bias = self.spatial_bias.to(device)
        
        # 索引查表
        spd_flat = spd_clamped.view(-1)
        bias_table = spatial_bias.t()  # (D+1, H)
        bias_flat = bias_table[spd_flat]  # (B*L*L, H)
        
        bias = bias_flat.view(batch_size, seq_len, seq_len, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        
        return bias


# ============================================================================
# Part 3: Edge Encoding Module
# ============================================================================

class EdgeEncodingBias(nn.Module):
    """
    Graphormer Edge Encoding
    
    核心思想:
    - 对于每对节点 (i, j)，找到它们的最短路径 SP_ij = (e_1, e_2, ..., e_N)
    - 每条边有一个类型 (骨架边 / 氢键边)
    - 学习每种边类型的 embedding
    - 计算路径上边 embedding 的加权平均作为 attention bias
    
    公式:
        c_ij = (1/N) * Σ(w_k * Emb(edge_type_k))
    
    其中:
        - edge_type_k 是第 k 条边的类型
        - w_k 是可学习的位置权重 (不同位置的边重要性不同)
        - Emb 是边类型的 embedding
    """
    
    def __init__(
        self, 
        num_heads: int, 
        num_edge_types: int = 3,  # backbone, pair, none
        max_path_length: int = 8
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.max_path_length = max_path_length
        
        # 边类型 embedding: (num_edge_types, num_heads)
        # 每种边类型对每个 head 有一个可学习的 bias 值
        self.edge_type_embedding = nn.Parameter(
            torch.zeros(num_edge_types, num_heads)
        )
        
        # 位置权重: (max_path_length, num_heads)
        # 路径上不同位置的边可以有不同的权重
        self.position_weights = nn.Parameter(
            torch.ones(max_path_length, num_heads)
        )
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """
        初始化边类型 embedding
        
        策略:
        - 骨架边 (backbone): 基础连接，给小的正 bias
        - 氢键边 (pair): 关键结构，给较大的正 bias
        - None (padding): 零 bias
        """
        with torch.no_grad():
            # Backbone edge: 小正值
            self.edge_type_embedding[EDGE_TYPE_BACKBONE, :] = 0.1
            # Pair edge: 较大正值 (氢键配对更重要)
            self.edge_type_embedding[EDGE_TYPE_PAIR, :] = 0.3
            # None: 零
            self.edge_type_embedding[EDGE_TYPE_NONE, :] = 0.0
    
    def forward(self, edge_path_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_path_matrix: (batch_size, seq_len, seq_len, max_path_length)
                             每个元素是边类型 (0, 1, 2)
        
        Returns:
            edge_bias: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _, path_len = edge_path_matrix.shape
        device = edge_path_matrix.device
        
        # 将 embedding 和 weights 移到正确设备
        edge_emb = self.edge_type_embedding.to(device)  # (E, H)
        pos_weights = self.position_weights.to(device)  # (P, H)
        
        # 获取每条边的 embedding
        # edge_path_matrix: (B, L, L, P) -> flatten -> index -> reshape
        edge_flat = edge_path_matrix.view(-1)  # (B*L*L*P,)
        edge_emb_flat = edge_emb[edge_flat]    # (B*L*L*P, H)
        edge_embs = edge_emb_flat.view(batch_size, seq_len, seq_len, path_len, self.num_heads)
        # (B, L, L, P, H)
        
        # 应用位置权重
        pos_weights_expanded = pos_weights.view(1, 1, 1, path_len, self.num_heads)
        weighted_embs = edge_embs * pos_weights_expanded  # (B, L, L, P, H)
        
        # 计算有效边的 mask (非 NONE 的边)
        valid_mask = (edge_path_matrix != EDGE_TYPE_NONE).float()  # (B, L, L, P)
        valid_mask = valid_mask.unsqueeze(-1)  # (B, L, L, P, 1)
        
        # 加权平均 (只考虑有效边)
        masked_embs = weighted_embs * valid_mask  # (B, L, L, P, H)
        sum_embs = masked_embs.sum(dim=3)  # (B, L, L, H)
        
        # 计算有效边数量 (避免除零)
        num_valid = valid_mask.sum(dim=3).clamp(min=1.0)  # (B, L, L, 1)
        
        # 平均
        edge_bias = sum_embs / num_valid  # (B, L, L, H)
        
        # 转置为 (B, H, L, L)
        edge_bias = edge_bias.permute(0, 3, 1, 2)
        
        return edge_bias
    
    def extra_repr(self) -> str:
        return (f'num_heads={self.num_heads}, '
                f'num_edge_types={self.num_edge_types}, '
                f'max_path_length={self.max_path_length}')


# ============================================================================
# Part 4: Combined Graphormer Encoding Module
# ============================================================================

class GraphormerEncodingBias(nn.Module):
    """
    完整的 Graphormer 编码模块
    
    组合 Spatial Encoding 和 Edge Encoding:
    
    公式:
        A_ij = (Q_i K_j^T) / √d + b_φ(vi, vj) + c_ij
    
    其中:
        - b_φ(vi, vj): Spatial Encoding (基于 SPD)
        - c_ij: Edge Encoding (基于路径上的边类型)
    """
    
    def __init__(
        self, 
        num_heads: int, 
        max_distance: int = 32,
        max_path_length: int = 8,
        use_spatial: bool = True,
        use_edge: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_spatial = use_spatial
        self.use_edge = use_edge
        
        if use_spatial:
            self.spatial_encoding = SpatialEncodingBias(
                num_heads=num_heads,
                max_distance=max_distance
            )
        
        if use_edge:
            self.edge_encoding = EdgeEncodingBias(
                num_heads=num_heads,
                num_edge_types=3,
                max_path_length=max_path_length
            )
    
    def forward(
        self, 
        spd_matrix: torch.Tensor,
        edge_path_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            spd_matrix: (batch_size, seq_len, seq_len)
            edge_path_matrix: (batch_size, seq_len, seq_len, max_path_length)
        
        Returns:
            total_bias: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = spd_matrix.shape
        device = spd_matrix.device
        
        total_bias = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len,
            device=device
        )
        
        # 添加 Spatial Encoding
        if self.use_spatial:
            spatial_bias = self.spatial_encoding(spd_matrix)
            total_bias = total_bias + spatial_bias
        
        # 添加 Edge Encoding
        if self.use_edge and edge_path_matrix is not None:
            edge_bias = self.edge_encoding(edge_path_matrix)
            total_bias = total_bias + edge_bias
        
        return total_bias


# ============================================================================
# Part 5: Custom Dataset with SPD and Edge Path
# ============================================================================

class OmniDatasetWithGraphormerEncoding(OmniDatasetForSequenceClassification):
    """
    支持完整 Graphormer Encoding 的数据集类
    
    返回:
    - input_ids, attention_mask: tokenized 序列
    - tissue_id: 组织类型编号 (0-8)
    - spd_matrix: 最短路径距离矩阵 (seq_len, seq_len)
    - edge_path_matrix: 边路径矩阵 (seq_len, seq_len, max_path_length)
    - labels: 分类标签
    
    缓存功能:
    - use_cache: 是否使用缓存
    - cache_dir: 缓存目录 (默认在数据集目录下创建 .graphormer_cache)
    - 缓存包含 spd_matrix 和 edge_path_matrix
    """
    
    # 类变量：存储预加载的缓存
    _preloaded_cache = {}
    
    def __init__(
        self, 
        dataset_name_or_path, 
        tokenizer, 
        max_length=None, 
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
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.use_cache = use_cache
        
        # 设置缓存目录
        if cache_dir is None:
            if os.path.isdir(dataset_name_or_path):
                cache_dir = os.path.join(dataset_name_or_path, ".graphormer_cache")
            else:
                cache_dir = os.path.join(os.path.dirname(dataset_name_or_path), ".graphormer_cache")
        self.cache_dir = cache_dir
        
        # 存储当前数据集路径和 split 名称
        self._dataset_path = dataset_name_or_path
        self._split_name = split_name
        
        # 尝试加载缓存
        self._cache = None
        self._spd_only_cache = False  # 是否只有 SPD 缓存
        if self.use_cache:
            self._cache = self._load_cache()
        
        # 标记是否使用缓存
        self._cache_loaded = self._cache is not None
        self._sample_index = 0
        
        if self._cache_loaded:
            if self._spd_only_cache:
                print(f"   🚀 Using SPD cache (edge paths will be computed)")
            else:
                print(f"   🚀 Using full Graphormer cache (skipping all computation)")
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
        
        # 如果没有使用缓存，保存新计算的数据
        if self.use_cache and not self._cache_loaded:
            self._save_cache()
    
    def _get_cache_path(self, split_name=None):
        """
        生成缓存文件路径
        
        缓存文件命名: graphormer_cache_{split}_len{max_length}_dist{max_distance}_path{max_path_length}.pt
        """
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
        
        cache_filename = f"graphormer_cache_{split_name}_len{self.max_length}_dist{self.max_distance}_path{self.max_path_length}.pt"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _get_spd_cache_path(self, split_name=None):
        """
        获取 SPD 版本的缓存路径（用于复用）
        """
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
        
        # SPD 缓存目录
        spd_cache_dir = self.cache_dir.replace('.graphormer_cache', '.spd_cache')
        cache_filename = f"spd_cache_{split_name}_len{self.max_length}_dist{self.max_distance}.pt"
        return os.path.join(spd_cache_dir, cache_filename)
    
    def _load_cache(self, split_name=None):
        """
        尝试加载缓存
        
        优先级:
        1. 加载完整的 Graphormer 缓存 (spd + edge)
        2. 加载 SPD 缓存 (只有 spd，需要额外计算 edge)
        
        Returns:
            dict or None: 缓存数据
        """
        if not self.use_cache:
            return None
        
        # 1. 尝试加载完整的 Graphormer 缓存
        cache_path = self._get_cache_path(split_name)
        
        if os.path.exists(cache_path):
            print(f"📂 Loading Graphormer cache from: {cache_path}")
            try:
                cache = torch.load(cache_path, weights_only=True)
                print(f"   ✅ Loaded {len(cache)} cached matrices (SPD + Edge)")
                self._spd_only_cache = False
                return cache
            except Exception as e:
                print(f"   ⚠️ Failed to load Graphormer cache: {e}")
        
        # 2. 尝试加载 SPD 缓存（可以复用）
        spd_cache_path = self._get_spd_cache_path(split_name)
        
        if os.path.exists(spd_cache_path):
            print(f"📂 Loading SPD cache (reusing from spd version): {spd_cache_path}")
            try:
                spd_cache = torch.load(spd_cache_path, weights_only=True)
                print(f"   ✅ Loaded {len(spd_cache)} SPD matrices")
                print(f"   ℹ️ Edge paths will be computed (not cached)")
                self._spd_only_cache = True
                # 转换格式: {index: spd_tensor} -> {index: {'spd': spd_tensor}}
                cache = {k: {'spd': v} for k, v in spd_cache.items()}
                return cache
            except Exception as e:
                print(f"   ⚠️ Failed to load SPD cache: {e}")
        
        return None
    
    def _save_cache(self, split_name=None):
        """
        保存缓存
        """
        if not self.use_cache:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._get_cache_path(split_name)
        
        # 收集所有矩阵
        cache = {}
        for i, item in enumerate(self.data):
            if "spd_matrix" in item and "edge_path_matrix" in item:
                cache[i] = {
                    'spd': item["spd_matrix"],
                    'edge': item["edge_path_matrix"]
                }
        
        if cache:
            print(f"💾 Saving Graphormer cache to: {cache_path}")
            torch.save(cache, cache_path)
            print(f"   ✅ Saved {len(cache)} matrices (SPD + Edge)")
    
    @classmethod
    def preload_cache(cls, cache_dir, max_length, max_distance, max_path_length, 
                      splits=['train', 'valid', 'test']):
        """
        预加载所有 split 的缓存
        """
        for split in splits:
            cache_filename = f"graphormer_cache_{split}_len{max_length}_dist{max_distance}_path{max_path_length}.pt"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            if os.path.exists(cache_path):
                print(f"📂 Preloading cache: {cache_path}")
                try:
                    cls._preloaded_cache[split] = torch.load(cache_path, weights_only=True)
                    print(f"   ✅ Preloaded {len(cls._preloaded_cache[split])} matrices for {split}")
                except Exception as e:
                    print(f"   ⚠️ Failed to preload {split}: {e}")
    
    @classmethod
    def from_hub(cls, dataset_name_or_path, tokenizer, max_length=None,
                 max_distance=32, max_path_length=8, use_cache=True, 
                 cache_dir=None, **kwargs):
        """
        从目录加载数据集（train, valid, test splits）
        """
        if cache_dir is None and os.path.isdir(dataset_name_or_path):
            cache_dir = os.path.join(dataset_name_or_path, ".graphormer_cache")
        
        datasets = {}
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(dataset_name_or_path, f"{split}.csv")
            if os.path.exists(split_path):
                print(f"\n📊 Loading {split} split...")
                datasets[split] = cls(
                    split_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    max_distance=max_distance,
                    max_path_length=max_path_length,
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                    split_name=split,
                    **kwargs
                )
        
        return datasets
    
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据
        """
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
        
        # 获取当前样本索引
        current_index = self._sample_index
        self._sample_index += 1
        
        # 🔑 计算或加载 SPD 矩阵和边路径矩阵
        # 优先使用缓存
        if self._cache_loaded and self._cache is not None and current_index in self._cache:
            cached = self._cache[current_index]
            tokenized_inputs["spd_matrix"] = cached['spd']
            
            # 检查是否有完整缓存（包含 edge）
            if 'edge' in cached:
                # 完整的 Graphormer 缓存
                tokenized_inputs["edge_path_matrix"] = cached['edge']
            elif structure is not None:
                # 只有 SPD 缓存，需要计算 edge_path
                pair_edges = dot_bracket_to_edges(structure)
                seq_len = len(structure)
                
                edge_paths = compute_edge_paths_only(
                    seq_len, 
                    pair_edges, 
                    self.max_distance,
                    self.max_path_length
                )
                
                # Padding
                if seq_len < self.max_length:
                    padded_edge = torch.full(
                        (self.max_length, self.max_length, self.max_path_length), 
                        EDGE_TYPE_NONE, 
                        dtype=torch.uint8
                    )
                    padded_edge[:seq_len, :seq_len, :] = edge_paths
                    edge_paths = padded_edge
                elif seq_len > self.max_length:
                    edge_paths = edge_paths[:self.max_length, :self.max_length, :]
                
                tokenized_inputs["edge_path_matrix"] = edge_paths
            else:
                # 没有结构信息，使用默认值
                tokenized_inputs["edge_path_matrix"] = torch.full(
                    (self.max_length, self.max_length, self.max_path_length),
                    EDGE_TYPE_NONE,
                    dtype=torch.uint8
                )
        elif structure is not None:
            # 没有任何缓存，完整计算
            pair_edges = dot_bracket_to_edges(structure)
            seq_len = len(structure)
            
            spd, edge_paths = compute_spd_and_edge_paths(
                seq_len, 
                pair_edges, 
                self.max_distance,
                self.max_path_length
            )
            
            # Padding
            if seq_len < self.max_length:
                padded_spd = torch.full(
                    (self.max_length, self.max_length), 
                    self.max_distance, 
                    dtype=torch.uint8
                )
                padded_spd[:seq_len, :seq_len] = spd
                spd = padded_spd
                
                padded_edge = torch.full(
                    (self.max_length, self.max_length, self.max_path_length), 
                    EDGE_TYPE_NONE, 
                    dtype=torch.uint8
                )
                padded_edge[:seq_len, :seq_len, :] = edge_paths
                edge_paths = padded_edge
            elif seq_len > self.max_length:
                spd = spd[:self.max_length, :self.max_length]
                edge_paths = edge_paths[:self.max_length, :self.max_length, :]
            
            tokenized_inputs["spd_matrix"] = spd
            tokenized_inputs["edge_path_matrix"] = edge_paths
        else:
            # 默认: 线性距离，全骨架边
            spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
            edge_paths = torch.full(
                (self.max_length, self.max_length, self.max_path_length),
                EDGE_TYPE_NONE,
                dtype=torch.uint8
            )
            for i in range(self.max_length):
                for j in range(self.max_length):
                    spd[i, j] = min(abs(i - j), self.max_distance)
            tokenized_inputs["spd_matrix"] = spd
            tokenized_inputs["edge_path_matrix"] = edge_paths
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写 padding 方法
        """
        tissue_ids = []
        spd_matrices = []
        edge_path_matrices = []
        
        for item in self.data:
            tissue_ids.append(item.pop("tissue_id", None))
            spd_matrices.append(item.pop("spd_matrix", None))
            edge_path_matrices.append(item.pop("edge_path_matrix", None))
        
        super()._pad_and_truncate(pad_value)
        
        for i, item in enumerate(self.data):
            # 恢复 tissue_id
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
            
            # 恢复 spd_matrix
            if spd_matrices[i] is not None:
                item["spd_matrix"] = spd_matrices[i]
            else:
                spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
                for ii in range(self.max_length):
                    for jj in range(self.max_length):
                        spd[ii, jj] = min(abs(ii - jj), self.max_distance)
                item["spd_matrix"] = spd
            
            # 恢复 edge_path_matrix
            if edge_path_matrices[i] is not None:
                item["edge_path_matrix"] = edge_path_matrices[i]
            else:
                item["edge_path_matrix"] = torch.full(
                    (self.max_length, self.max_length, self.max_path_length),
                    EDGE_TYPE_NONE,
                    dtype=torch.uint8
                )


# ============================================================================
# Part 6: Graphormer-style Attention Layer
# ============================================================================

class GraphormerAttentionLayer(nn.Module):
    """
    完整的 Graphormer 风格 Attention 层
    
    包含:
    - 标准 Multi-Head Attention
    - Spatial Encoding (SPD-based bias)
    - Edge Encoding (path-based bias)
    
    公式:
        A_ij = (Q_i K_j^T) / √d + b_φ(vi, vj) + c_ij
        Attention = Softmax(A) V
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        max_distance: int = 32,
        max_path_length: int = 8,
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_edge: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 🔑 Graphormer Encoding (Spatial + Edge)
        self.graphormer_encoding = GraphormerEncodingBias(
            num_heads=num_heads,
            max_distance=max_distance,
            max_path_length=max_path_length,
            use_spatial=use_spatial,
            use_edge=use_edge
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale = self.head_dim ** -0.5
        
        self.use_spatial = use_spatial
        self.use_edge = use_edge
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        spd_matrix: torch.Tensor,
        edge_path_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            spd_matrix: (batch, seq_len, seq_len)
            edge_path_matrix: (batch, seq_len, seq_len, max_path_length)
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 🔑 添加 Graphormer bias (Spatial + Edge Encoding)
        graphormer_bias = self.graphormer_encoding(
            spd_matrix, 
            edge_path_matrix
        )
        attn_scores = attn_scores + graphormer_bias
        
        # Padding mask
        padding_mask = attention_mask[:, None, None, :].float()
        padding_mask = (1.0 - padding_mask) * -10000.0
        attn_scores = attn_scores + padding_mask
        
        # Softmax + Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Output
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual + LayerNorm
        output = self.layer_norm(hidden_states + attn_output)
        
        return output


# ============================================================================
# Part 7: Model with Full Graphormer Encoding
# ============================================================================

class OmniModelWithGraphormerEncoding(OmniModelForSequenceClassification):
    """
    支持完整 Graphormer Encoding 的序列分类模型
    
    包含:
    1. Spatial Encoding: 基于最短路径距离 (SPD)
    2. Edge Encoding: 基于路径上的边类型 (骨架边 vs 氢键边)
    
    公式:
        A_ij = (h_i W_Q)(h_j W_K)^T / √d + b_φ(vi, vj) + c_ij
    """
    
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithGraphormerEncoding)
        self.max_distance = kwargs.pop('max_distance', 32)
        self.max_path_length = kwargs.pop('max_path_length', 8)
        self.use_spatial = kwargs.pop('use_spatial', True)
        self.use_edge = kwargs.pop('use_edge', True)
        
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        hidden_size = self.config.hidden_size
        num_attention_heads = getattr(self.config, 'num_attention_heads', 12)
        
        # 🔑 Graphormer Attention 层
        self.structure_attention = GraphormerAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            max_distance=self.max_distance,
            max_path_length=self.max_path_length,
            dropout=0.1,
            use_spatial=self.use_spatial,
            use_edge=self.use_edge
        )
        
        # Tissue 嵌入
        self.tissue_embed_dim = hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,
            embedding_dim=self.tissue_embed_dim
        )
        
        self.classifier = nn.Linear(
            hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        print(f"📐 Initialized GraphormerAttentionLayer:")
        print(f"   - num_heads: {num_attention_heads}")
        print(f"   - max_distance: {self.max_distance}")
        print(f"   - max_path_length: {self.max_path_length}")
        print(f"   - use_spatial: {self.use_spatial}")
        print(f"   - use_edge: {self.use_edge}")
    
    def forward(self, **inputs):
        """
        Forward pass with Graphormer encoding
        """
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        spd_matrix = inputs.pop("spd_matrix", None)
        edge_path_matrix = inputs.pop("edge_path_matrix", None)
        
        attention_mask = inputs.get("attention_mask", None)
        
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        
        # 🔑 应用 Graphormer Attention
        if spd_matrix is not None and attention_mask is not None:
            device = last_hidden_state.device
            spd_matrix = spd_matrix.to(device)
            attention_mask = attention_mask.to(device)
            
            if edge_path_matrix is not None:
                edge_path_matrix = edge_path_matrix.to(device)
            
            last_hidden_state = self.structure_attention(
                last_hidden_state, 
                attention_mask, 
                spd_matrix,
                edge_path_matrix
            )
        
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        
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
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = self.loss_fn(logits_flat, labels_flat)
        
        outputs = {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
        return outputs


# ============================================================================
# Part 8: Visualization and Analysis
# ============================================================================

def print_graphormer_example():
    """
    打印 Graphormer Encoding 示例
    
    使用截图中的例子: ..((...))"
    """
    structure = "..((...))"
    seq_len = len(structure)
    
    print(f"\n{'='*70}")
    print("Graphormer Encoding 示例 (Spatial + Edge Encoding)")
    print(f"{'='*70}")
    print(f"结构: {structure}")
    print(f"长度: {seq_len}")
    
    # 解析配对
    pair_edges = dot_bracket_to_edges(structure)
    print(f"\n配对边 (Hydrogen Bond Edges):")
    for i, j in pair_edges:
        print(f"  - 位置 {i} <-> {j}")
    
    print(f"\n骨架边 (Backbone Edges):")
    for i in range(seq_len - 1):
        print(f"  - 位置 {i} <-> {i+1}")
    
    # 计算 SPD 和 Edge Paths
    spd, edge_paths = compute_spd_and_edge_paths(seq_len, pair_edges, max_path_length=8)
    
    print(f"\n最短路径距离 (SPD) 矩阵:")
    print("    ", end="")
    for j in range(seq_len):
        print(f"{j:3}", end="")
    print()
    for i in range(seq_len):
        print(f"{i:3} ", end="")
        for j in range(seq_len):
            print(f"{spd[i,j].item():3}", end="")
        print()
    
    # 边类型示例
    print(f"\n边类型 (0=Backbone, 1=Pair, 2=None):")
    edge_type_names = {0: "Backbone", 1: "Pair", 2: "None"}
    
    # 示例: 几个典型的路径
    examples = [(1, 4), (2, 7), (0, 8)]
    for i, j in examples:
        if i < seq_len and j < seq_len:
            path = edge_paths[i, j]
            path_edges = [edge_type_names[e.item()] for e in path if e.item() != EDGE_TYPE_NONE]
            print(f"  路径 {i} -> {j}:")
            print(f"    距离: {spd[i,j].item()}")
            print(f"    边类型序列: {' -> '.join(path_edges)}")


def visualize_graphormer_encoding(structure: str, save_path: str = None):
    """
    可视化 Graphormer Encoding
    """
    import matplotlib.pyplot as plt
    
    seq_len = len(structure)
    pair_edges = dot_bracket_to_edges(structure)
    spd, edge_paths = compute_spd_and_edge_paths(seq_len, pair_edges)
    
    # 计算边编码的统计信息
    pair_count = (edge_paths == EDGE_TYPE_PAIR).sum(dim=-1).float()
    backbone_count = (edge_paths == EDGE_TYPE_BACKBONE).sum(dim=-1).float()
    pair_ratio = pair_count / (pair_count + backbone_count + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # SPD 矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(spd.numpy(), cmap='viridis', aspect='auto')
    ax1.set_title(f'Spatial Encoding: SPD Matrix\n{structure}')
    ax1.set_xlabel('Position j')
    ax1.set_ylabel('Position i')
    plt.colorbar(im1, ax=ax1, label='SPD')
    
    # 氢键边比例
    ax2 = axes[1]
    im2 = ax2.imshow(pair_ratio.numpy(), cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Edge Encoding: Pair Edge Ratio')
    ax2.set_xlabel('Position j')
    ax2.set_ylabel('Position i')
    plt.colorbar(im2, ax=ax2, label='Pair Ratio')
    
    # 线性距离 vs SPD 对比
    ax3 = axes[2]
    linear_dist = torch.abs(
        torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    ).float()
    diff = linear_dist - spd.float()
    im3 = ax3.imshow(diff.numpy(), cmap='RdBu', aspect='auto', vmin=-5, vmax=5)
    ax3.set_title('Linear Distance - SPD\n(Positive = Structure shortcut)')
    ax3.set_xlabel('Position j')
    ax3.set_ylabel('Position i')
    plt.colorbar(im3, ax=ax3, label='Difference')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Part 9: Training Script
# ============================================================================

if __name__ == "__main__":
    # 打印示例
    print_graphormer_example()
    
    # Model configuration
    model_name_or_path = "yangheng/OmniGenome-52M"
    
    label2id = {"0": 0, "1": 1, "2": 2}
    
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
    
    # Graphormer 参数
    max_distance = 32
    max_path_length = 8
    
    # Load datasets
    print("\n📊 Loading datasets with Graphormer encoding...")
    datasets = OmniDatasetWithGraphormerEncoding.from_hub(
        "/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10",
        tokenizer=tokenizer,
        max_length=512,
        max_distance=max_distance,
        max_path_length=max_path_length,
        label2id=label2id,
    )
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")
    
    # Verify data
    sample = datasets["train"][0]
    print(f"\n📋 Sample data keys: {list(sample.keys())}")
    
    if "spd_matrix" in sample:
        spd = sample["spd_matrix"]
        print(f"  - spd_matrix shape: {spd.shape}")
        print(f"  - spd_matrix range: [{spd.min().item()}, {spd.max().item()}]")
    
    if "edge_path_matrix" in sample:
        edge = sample["edge_path_matrix"]
        print(f"  - edge_path_matrix shape: {edge.shape}")
        backbone_count = (edge == EDGE_TYPE_BACKBONE).sum().item()
        pair_count = (edge == EDGE_TYPE_PAIR).sum().item()
        print(f"  - Backbone edges: {backbone_count}")
        print(f"  - Pair edges: {pair_count}")
    
    # Initialize model
    print("\n🔧 Initializing model with Graphormer Encoding...")
    model = OmniModelWithGraphormerEncoding(
        model_name_or_path,
        tokenizer,
        num_labels=len(label2id),
        dataset_class=OmniDatasetWithGraphormerEncoding,
        max_distance=max_distance,
        max_path_length=max_path_length,
        use_spatial=True,
        use_edge=True,
    )
    
    # Print encoding parameters
    print(f"\n📐 Graphormer Encoding parameters:")
    
    ge = model.structure_attention.graphormer_encoding
    
    if ge.use_spatial:
        se = ge.spatial_encoding
        print(f"\n  Spatial Encoding (distance -> bias):")
        for d in range(min(6, max_distance + 1)):
            mean_bias = se.spatial_bias[:, d].mean().item()
            print(f"      Distance {d}: {mean_bias:.4f}")
    
    if ge.use_edge:
        ee = ge.edge_encoding
        print(f"\n  Edge Encoding (edge_type -> bias):")
        print(f"      Backbone: {ee.edge_type_embedding[EDGE_TYPE_BACKBONE].mean().item():.4f}")
        print(f"      Pair:     {ee.edge_type_embedding[EDGE_TYPE_PAIR].mean().item():.4f}")
        print(f"      None:     {ee.edge_type_embedding[EDGE_TYPE_NONE].mean().item():.4f}")
    
    # Training
    metric_functions = [
        ClassificationMetric().accuracy_score,
        ClassificationMetric(average='macro').f1_score
    ]
    
    trainer = AccelerateTrainer(
        model=model,
        epochs=20,
        learning_rate=2e-5,
        batch_size=16,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        test_dataset=datasets["test"],
        compute_metrics=metric_functions,
        gradient_accumulation_steps=4,
        device=torch.device("cuda:0"),
        monitor='valid_accuracy_score',
        load_best_model_at_end=True,
    )
    
    print("\n🎓 Starting training with Graphormer Encoding (Spatial + Edge)...")
    metrics = trainer.train(
        path_to_save="ogb_te_3class_finetuned_52M_seq_tissue_graphormer",
        dataset_class=OmniDatasetWithGraphormerEncoding
    )
    
    # Print final parameters
    print(f"\n📐 Graphormer Encoding parameters (after training):")
    
    if ge.use_spatial:
        se = ge.spatial_encoding
        print(f"\n  Spatial Encoding (distance -> bias):")
        for d in range(min(6, max_distance + 1)):
            mean_bias = se.spatial_bias[:, d].mean().item()
            print(f"      Distance {d}: {mean_bias:.4f}")
    
    if ge.use_edge:
        ee = ge.edge_encoding
        print(f"\n  Edge Encoding (edge_type -> bias):")
        print(f"      Backbone: {ee.edge_type_embedding[EDGE_TYPE_BACKBONE].mean().item():.4f}")
        print(f"      Pair:     {ee.edge_type_embedding[EDGE_TYPE_PAIR].mean().item():.4f}")
    
    print('\n✅ Final Metrics:', metrics)
