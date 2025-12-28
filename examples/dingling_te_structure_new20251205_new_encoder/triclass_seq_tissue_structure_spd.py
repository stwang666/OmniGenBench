# -*- coding: utf-8 -*-
"""
RNA Translation Efficiency Classification with Graphormer-style Spatial Encoding

This script implements Spatial Encoding based on Shortest Path Distance (SPD)
following the Graphormer paper (NeurIPS 2021).

Spatial Encoding (空间编码):
- 将RNA二级结构转换为图: 骨架边 (i, i+1) + 氢键配对边
- 计算任意两点间的最短路径距离 (SPD)
- 为每个距离值分配一个可学习的 scalar bias b_φ(vi, vj)
- 公式: A_ij = (h_i W_Q)(h_j W_K)^T / √d + b_φ(vi, vj)

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
from typing import Optional, List, Dict, Tuple
import re


# ============================================================================
# Part 1: Graph Construction from RNA Secondary Structure
# ============================================================================

def dot_bracket_to_edges(structure: str) -> List[Tuple[int, int]]:
    """
    将点括号表示法转换为边列表
    
    Args:
        structure: 点括号表示法，如 "..((...))"
    
    Returns:
        edges: 配对边列表 [(i, j), ...]
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


def compute_shortest_path_distance(
    seq_len: int, 
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32
) -> torch.Tensor:
    """
    计算 RNA 结构图中任意两点间的最短路径距离 (SPD)
    
    图的构建:
    - 骨架边 (Backbone): i -> i+1，所有相邻核苷酸
    - 氢键边 (Hydrogen Bond): 碱基配对边
    
    使用 BFS 计算每个节点到其他所有节点的最短路径
    
    Args:
        seq_len: 序列长度
        pair_edges: 配对边列表 [(i, j), ...]
        max_distance: 最大距离，超过此值的距离会被截断
    
    Returns:
        spd_matrix: (seq_len, seq_len) 最短路径距离矩阵
                   spd_matrix[i,j] = 从节点 i 到节点 j 的最短路径距离
                   不连通的节点对距离为 max_distance
    """
    # 构建邻接表
    adj = [[] for _ in range(seq_len)]
    
    # 添加骨架边 (backbone edges): i <-> i+1
    for i in range(seq_len - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    
    # 添加氢键配对边 (hydrogen bond edges)
    for i, j in pair_edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # BFS 计算每个节点的最短路径距离
    spd_matrix = torch.full((seq_len, seq_len), max_distance, dtype=torch.uint8)
    
    for start in range(seq_len):
        # BFS from start node
        distances = [-1] * seq_len
        distances[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            current_dist = distances[node]
            
            if current_dist >= max_distance:
                continue
            
            for neighbor in adj[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # 填充 SPD 矩阵
        for end, dist in enumerate(distances):
            if dist >= 0:
                spd_matrix[start, end] = min(dist, max_distance)
    
    return spd_matrix


def batch_compute_spd(
    structures: List[str], 
    max_length: int,
    max_distance: int = 32
) -> torch.Tensor:
    """
    批量计算 SPD 矩阵
    
    Args:
        structures: 结构列表
        max_length: 统一的最大长度 (用于 padding)
        max_distance: 最大距离值
    
    Returns:
        spd_matrices: (batch_size, max_length, max_length)
    """
    matrices = []
    for struct in structures:
        seq_len = len(struct)
        pair_edges = dot_bracket_to_edges(struct)
        
        # 计算 SPD 矩阵
        spd = compute_shortest_path_distance(seq_len, pair_edges, max_distance)
        
        # Padding 到 max_length
        if seq_len < max_length:
            padded = torch.full((max_length, max_length), max_distance, dtype=torch.uint8)
            padded[:seq_len, :seq_len] = spd
            spd = padded
        elif seq_len > max_length:
            spd = spd[:max_length, :max_length]
        
        matrices.append(spd)
    
    return torch.stack(matrices)


# ============================================================================
# Part 2: Spatial Encoding Module (Graphormer-style)
# ============================================================================

class SpatialEncodingBias(nn.Module):
    """
    Graphormer 风格的空间编码 (Spatial Encoding)
    
    核心思想: 为每个最短路径距离值分配一个可学习的 scalar bias
    
    公式:
        A_ij = (Q_i K_j^T) / √d + b_φ(vi, vj)
    
    其中:
        - φ(vi, vj) 是节点 i 和 j 之间的最短路径距离
        - b_φ 是距离 φ 对应的可学习 bias
        - 每个 attention head 有独立的 bias 参数
    
    Args:
        num_heads: attention head 数量
        max_distance: 最大距离值 (距离 >= max_distance 使用同一个 bias)
    
    参数:
        spatial_bias: (num_heads, max_distance + 1) 
                     spatial_bias[h, d] 表示 head h 对距离 d 的 bias
    """
    
    def __init__(self, num_heads: int, max_distance: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # 可学习的距离 bias
        # +1 是因为距离从 0 开始 (自身距离为 0)
        self.spatial_bias = nn.Parameter(
            torch.zeros(num_heads, max_distance + 1)
        )
        
        # 初始化: 距离越近，bias 越大 (鼓励关注近邻)
        self._init_bias()
    
    def _init_bias(self):
        """
        初始化 bias 参数
        
        策略: 距离越近，初始 bias 越大
        - 距离 0 (自身): 0.5
        - 距离 1 (直接相邻): 0.3
        - 距离 2: 0.2
        - 距离 3+: 逐渐衰减到 0
        """
        with torch.no_grad():
            for d in range(self.max_distance + 1):
                if d == 0:
                    self.spatial_bias[:, d] = 0.5
                elif d == 1:
                    self.spatial_bias[:, d] = 0.3
                elif d == 2:
                    self.spatial_bias[:, d] = 0.2
                else:
                    # 指数衰减
                    self.spatial_bias[:, d] = 0.1 * np.exp(-0.1 * (d - 2))
    
    def forward(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        将 SPD 矩阵转换为 attention bias
        
        Args:
            spd_matrix: (batch_size, seq_len, seq_len) 
                       最短路径距离矩阵，值域 [0, max_distance]
        
        Returns:
            attention_bias: (batch_size, num_heads, seq_len, seq_len)
                           可以直接加到 attention scores 上
        """
        batch_size, seq_len, _ = spd_matrix.shape
        device = spd_matrix.device
        
        # 确保距离值在有效范围内，并转换为整型索引用于查表
        spd_clamped = spd_matrix.to(torch.long).clamp(0, self.max_distance)  # (B, L, L)
        
        # 将 spatial_bias 移动到正确的设备
        spatial_bias = self.spatial_bias.to(device)  # (H, D+1)
        
        # 使用索引查表获取每个位置对的 bias
        # spd_clamped: (B, L, L) -> 用于索引 spatial_bias 的第二维
        # 结果: (B, H, L, L)
        
        # 方法: 展平后索引，再 reshape
        spd_flat = spd_clamped.view(-1)  # (B*L*L)
        
        # 为每个 head 生成 bias
        # spatial_bias: (H, D+1) -> 转置为 (D+1, H)
        bias_table = spatial_bias.t()  # (D+1, H)
        
        # 索引: (B*L*L,) -> (B*L*L, H)
        bias_flat = bias_table[spd_flat]  # (B*L*L, H)
        
        # Reshape: (B*L*L, H) -> (B, L, L, H) -> (B, H, L, L)
        bias = bias_flat.view(batch_size, seq_len, seq_len, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        
        return bias
    
    def extra_repr(self) -> str:
        return f'num_heads={self.num_heads}, max_distance={self.max_distance}'


# ============================================================================
# Part 3: Custom Dataset with SPD Information
# ============================================================================

class OmniDatasetWithSPD(OmniDatasetForSequenceClassification):
    """
    支持 tissue 和 SPD (Shortest Path Distance) 信息的数据集类
    
    从 CSV 中读取:
    - sequence: RNA 序列
    - structure: 点括号表示法的二级结构
    - tissue: 组织类型名称
    - label: 分类标签
    
    返回:
    - input_ids, attention_mask: tokenized 序列
    - tissue_id: 组织类型编号 (0-8)
    - spd_matrix: 最短路径距离矩阵 (seq_len, seq_len)
    - labels: 分类标签
    
    缓存功能:
    - use_cache: 是否使用缓存
    - cache_dir: 缓存目录 (默认在数据集目录下创建 .spd_cache)
    - 缓存文件名基于 split、max_length、max_distance 生成
    """
    
    # 类变量：存储预加载的缓存
    _preloaded_cache = {}
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, 
                 max_distance=32, use_cache=True, cache_dir=None, 
                 split_name=None, **kwargs):
        # 在调用 super().__init__() 之前初始化映射
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        self.max_length = max_length
        self.max_distance = max_distance
        self.use_cache = use_cache
        
        # 设置缓存目录
        if cache_dir is None:
            if os.path.isdir(dataset_name_or_path):
                cache_dir = os.path.join(dataset_name_or_path, ".spd_cache")
            else:
                cache_dir = os.path.join(os.path.dirname(dataset_name_or_path), ".spd_cache")
        self.cache_dir = cache_dir
        
        # 存储当前数据集路径和 split 名称
        self._dataset_path = dataset_name_or_path
        self._split_name = split_name  # 可以显式指定 split 名称
        
        # 尝试加载缓存
        self._spd_cache = None
        if self.use_cache:
            self._spd_cache = self._load_spd_cache()
        
        # 标记是否需要计算 SPD（如果有缓存就跳过计算）
        self._cache_loaded = self._spd_cache is not None
        self._sample_index = 0  # 用于跟踪当前处理的样本索引
        
        if self._cache_loaded:
            print(f"   🚀 Using cached SPD matrices (skipping computation)")
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 后处理：确保 tissue_id 的维度正确
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
        
        # 如果没有使用缓存，保存新计算的 SPD
        if self.use_cache and not self._cache_loaded:
            self._save_spd_cache()
    
    def _get_cache_path(self, split_name=None):
        """
        生成缓存文件路径
        
        缓存文件命名: spd_cache_{split}_{max_length}_{max_distance}.pt
        """
        if split_name is None:
            # 优先使用显式指定的 split 名称
            if hasattr(self, '_split_name') and self._split_name is not None:
                split_name = self._split_name
            else:
                # 尝试从路径中提取 split 名称
                path = self._dataset_path
                path_lower = path.lower()
                if 'train' in path_lower:
                    split_name = 'train'
                elif 'valid' in path_lower:
                    split_name = 'valid'
                elif 'test' in path_lower:
                    split_name = 'test'
                else:
                    # 使用路径的 hash
                    split_name = hashlib.md5(path.encode()).hexdigest()[:8]
        
        cache_filename = f"spd_cache_{split_name}_len{self.max_length}_dist{self.max_distance}.pt"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_spd_cache(self, split_name=None):
        """
        尝试加载 SPD 缓存
        
        Returns:
            dict or None: 缓存数据 {index: spd_matrix} 或 None
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(split_name)
        
        if os.path.exists(cache_path):
            print(f"📂 Loading SPD cache from: {cache_path}")
            try:
                cache = torch.load(cache_path, weights_only=True)
                print(f"   ✅ Loaded {len(cache)} cached SPD matrices")
                return cache
            except Exception as e:
                print(f"   ⚠️ Failed to load cache: {e}")
                return None
        return None
    
    def _save_spd_cache(self, split_name=None):
        """
        保存 SPD 缓存
        """
        if not self.use_cache:
            return
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        cache_path = self._get_cache_path(split_name)
        
        # 收集所有 SPD 矩阵
        cache = {}
        for i, item in enumerate(self.data):
            if "spd_matrix" in item:
                cache[i] = item["spd_matrix"]
        
        if cache:
            print(f"💾 Saving SPD cache to: {cache_path}")
            torch.save(cache, cache_path)
            print(f"   ✅ Saved {len(cache)} SPD matrices")
    
    @classmethod
    def preload_cache(cls, cache_dir, max_length, max_distance, splits=['train', 'valid', 'test']):
        """
        预加载所有 split 的缓存到类变量中
        
        使用方法:
            OmniDatasetWithSPD.preload_cache('/path/to/cache', 512, 32)
            datasets = OmniDatasetWithSPD.from_hub(...)  # 会自动使用预加载的缓存
        """
        for split in splits:
            cache_filename = f"spd_cache_{split}_len{max_length}_dist{max_distance}.pt"
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
                 max_distance=32, use_cache=True, cache_dir=None, **kwargs):
        """
        从目录加载数据集（train, valid, test splits）
        
        重写父类方法以支持 SPD 缓存
        """
        # 设置缓存目录
        if cache_dir is None and os.path.isdir(dataset_name_or_path):
            cache_dir = os.path.join(dataset_name_or_path, ".spd_cache")
        
        # 调用父类方法，传递额外参数
        # 父类会为每个 split 创建一个 Dataset 实例
        from omnigenbench import OmniDatasetForSequenceClassification
        
        # 获取所有 splits
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
                    use_cache=use_cache,
                    cache_dir=cache_dir,
                    split_name=split,
                    **kwargs
                )
        
        return datasets
    
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据，包括 tissue 和 SPD 信息
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
            
            # 获取 tissue 信息
            tissue_name = instance.get("tissue", None)
            if tissue_name is not None:
                tissue_id = self.tissue2id.get(tissue_name, 0)
            
            # 获取 structure 信息
            structure = instance.get("structure", None)
        else:
            raise Exception("Unknown instance format.")

        # Tokenize 序列
        tokenized_inputs = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        # 处理 labels
        if labels is not None:
            labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise Exception(
                    "The label must be an integer for sequence classification."
                )
        tokenized_inputs["labels"] = torch.tensor(labels)
        
        # 添加 tissue_id
        if tissue_id is not None:
            tokenized_inputs["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)
        else:
            tokenized_inputs["tissue_id"] = torch.tensor([0], dtype=torch.long)
        
        # 获取当前样本索引
        current_index = self._sample_index
        self._sample_index += 1
        
        # 🔑 计算或加载 SPD 矩阵 (Graphormer Spatial Encoding)
        # 优先使用缓存
        if self._cache_loaded and self._spd_cache is not None and current_index in self._spd_cache:
            tokenized_inputs["spd_matrix"] = self._spd_cache[current_index]
        elif structure is not None:
            pair_edges = dot_bracket_to_edges(structure)
            seq_len = len(structure)
            spd = compute_shortest_path_distance(seq_len, pair_edges, self.max_distance)
            
            # Padding 到 max_length
            if seq_len < self.max_length:
                padded = torch.full(
                    (self.max_length, self.max_length), 
                    self.max_distance, 
                    dtype=torch.uint8
                )
                padded[:seq_len, :seq_len] = spd
                spd = padded
            elif seq_len > self.max_length:
                spd = spd[:self.max_length, :self.max_length]
            
            tokenized_inputs["spd_matrix"] = spd
        else:
            # 如果没有结构信息，使用线性距离 (只有骨架边)
            spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
            for i in range(self.max_length):
                for j in range(self.max_length):
                    spd[i, j] = min(abs(i - j), self.max_distance)
            tokenized_inputs["spd_matrix"] = spd
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写 _pad_and_truncate 方法，跳过 tissue_id 和 spd_matrix 的 padding
        """
        # 临时移除不需要 padding 的字段
        tissue_ids = []
        spd_matrices = []
        
        for item in self.data:
            if "tissue_id" in item:
                tissue_ids.append(item.pop("tissue_id"))
            else:
                tissue_ids.append(None)
            
            if "spd_matrix" in item:
                spd_matrices.append(item.pop("spd_matrix"))
            else:
                spd_matrices.append(None)
        
        # 调用父类的 _pad_and_truncate
        super()._pad_and_truncate(pad_value)
        
        # 恢复 tissue_id
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
            
            # 恢复 spd_matrix
            if spd_matrices[i] is not None:
                item["spd_matrix"] = spd_matrices[i]
            else:
                # 默认: 线性距离
                spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
                for ii in range(self.max_length):
                    for jj in range(self.max_length):
                        spd[ii, jj] = min(abs(ii - jj), self.max_distance)
                item["spd_matrix"] = spd


# ============================================================================
# Part 4: Structure-Aware Attention with Spatial Encoding
# ============================================================================

class SpatialEncodingAttentionLayer(nn.Module):
    """
    带有 Graphormer Spatial Encoding 的 Attention 层
    
    核心思想:
    - 计算标准的 scaled dot-product attention
    - 添加基于 SPD (最短路径距离) 的可学习 bias
    
    公式:
        A_ij = (Q_i K_j^T) / √d + b_φ(vi, vj)
        Attention = Softmax(A) V
    
    与原始配对矩阵方法的区别:
    - 原始: 只区分 "配对" vs "非配对"
    - SPD: 细粒度的距离信息，能区分 "距离1" "距离2" ... "距离32+"
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        max_distance: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_distance = max_distance
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 🔑 Graphormer Spatial Encoding
        self.spatial_encoding = SpatialEncodingBias(
            num_heads=num_heads,
            max_distance=max_distance
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        spd_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) 1=有效, 0=padding
            spd_matrix: (batch, seq_len, seq_len) 最短路径距离矩阵
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(hidden_states)  # (batch, seq, hidden)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算 attention scores: (batch, heads, seq, seq)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 🔑 添加 Spatial Encoding bias (Graphormer-style)
        spatial_bias = self.spatial_encoding(spd_matrix)  # (batch, heads, seq, seq)
        attn_scores = attn_scores + spatial_bias
        
        # 添加 padding mask
        # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
        padding_mask = attention_mask[:, None, None, :].float()
        padding_mask = (1.0 - padding_mask) * -10000.0
        attn_scores = attn_scores + padding_mask
        
        # Softmax + Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 计算输出
        attn_output = torch.matmul(attn_probs, V)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 残差连接 + LayerNorm
        output = self.layer_norm(hidden_states + attn_output)
        
        return output


# ============================================================================
# Part 5: Model with Spatial Encoding
# ============================================================================

class OmniModelWithSpatialEncoding(OmniModelForSequenceClassification):
    """
    支持 Graphormer Spatial Encoding 的序列分类模型
    
    Spatial Encoding (空间编码):
    - 计算 RNA 结构图中任意两点的最短路径距离 (SPD)
    - 为每个距离值学习一个可学习的 bias
    - 将 bias 加到 attention scores 上
    
    公式:
        A_ij = (h_i W_Q)(h_j W_K)^T / √d + b_φ(vi, vj)
    
    其中 φ(vi, vj) 是节点 i 和 j 的最短路径距离
    """
    
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        # 保存参数
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithSPD)
        self.max_distance = kwargs.pop('max_distance', 32)
        
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        # 获取模型配置
        hidden_size = self.config.hidden_size
        num_attention_heads = getattr(self.config, 'num_attention_heads', 12)
        
        # 🔑 Spatial Encoding Attention 层
        self.structure_attention = SpatialEncodingAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            max_distance=self.max_distance,
            dropout=0.1
        )
        
        # Tissue 嵌入层
        self.tissue_embed_dim = hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,  # 9 个 tissue 类型
            embedding_dim=self.tissue_embed_dim
        )
        
        # 重新定义 classifier，输入维度为 hidden_size + tissue_embed_dim
        self.classifier = nn.Linear(
            hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        print(f"📐 Initialized SpatialEncodingAttentionLayer:")
        print(f"   - num_heads: {num_attention_heads}")
        print(f"   - max_distance: {self.max_distance}")
        print(f"   - spatial_bias shape: {self.structure_attention.spatial_encoding.spatial_bias.shape}")
    
    def forward(self, **inputs):
        """
        Forward pass with Spatial Encoding attention
        """
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        spd_matrix = inputs.pop("spd_matrix", None)
        
        # 保存 attention_mask 用于后续处理
        attention_mask = inputs.get("attention_mask", None)
        
        # 获取 backbone 的 last_hidden_state
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        
        # 🔑 应用 Spatial Encoding Attention 层
        if spd_matrix is not None and attention_mask is not None:
            device = last_hidden_state.device
            spd_matrix = spd_matrix.to(device)
            attention_mask = attention_mask.to(device)
            
            last_hidden_state = self.structure_attention(
                last_hidden_state, 
                attention_mask, 
                spd_matrix
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
        
        # Late Fusion: 拼接 pooled state 和 tissue embedding
        combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
        
        # Classification
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
# Part 6: Visualization and Analysis Utilities
# ============================================================================

def visualize_spd_matrix(structure: str, save_path: str = None):
    """
    可视化 SPD 矩阵
    
    Args:
        structure: 点括号表示法
        save_path: 保存路径 (如果为 None，则显示)
    """
    import matplotlib.pyplot as plt
    
    seq_len = len(structure)
    pair_edges = dot_bracket_to_edges(structure)
    spd = compute_shortest_path_distance(seq_len, pair_edges)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: SPD 矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(spd.numpy(), cmap='viridis', aspect='auto')
    ax1.set_title(f'Shortest Path Distance (SPD) Matrix\nStructure: {structure[:30]}...')
    ax1.set_xlabel('Position j')
    ax1.set_ylabel('Position i')
    plt.colorbar(im1, ax=ax1, label='SPD')
    
    # 右图: 线性距离 vs SPD 对比 (对于配对位置)
    ax2 = axes[1]
    linear_dist = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            linear_dist[i, j] = abs(i - j)
    
    # 差异: 线性距离 - SPD (正值表示 SPD 更短)
    diff = linear_dist - spd.float()
    im2 = ax2.imshow(diff.numpy(), cmap='RdBu', aspect='auto', vmin=-10, vmax=10)
    ax2.set_title('Linear Distance - SPD\n(Positive = SPD is shorter)')
    ax2.set_xlabel('Position j')
    ax2.set_ylabel('Position i')
    plt.colorbar(im2, ax=ax2, label='Difference')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved SPD visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_spd_example():
    """
    打印一个 SPD 计算的示例
    
    使用截图中的例子: ..((...))"..
    """
    # 使用截图中的例子
    structure = "..((...))"
    seq_len = len(structure)
    
    print(f"\n{'='*60}")
    print("SPD (Shortest Path Distance) 计算示例")
    print(f"{'='*60}")
    print(f"结构: {structure}")
    print(f"长度: {seq_len}")
    
    # 解析配对
    pair_edges = dot_bracket_to_edges(structure)
    print(f"\n配对边 (Edges):")
    for i, j in pair_edges:
        print(f"  - 位置 {i} 和 {j} 配对")
    
    # 计算 SPD
    spd = compute_shortest_path_distance(seq_len, pair_edges)
    
    print(f"\nSPD 矩阵:")
    print("    ", end="")
    for j in range(seq_len):
        print(f"{j:3}", end="")
    print()
    
    for i in range(seq_len):
        print(f"{i:3} ", end="")
        for j in range(seq_len):
            print(f"{spd[i,j].item():3}", end="")
        print()
    
    # 示例: 配对点距离
    print(f"\n关键距离示例:")
    for i, j in pair_edges:
        linear_dist = abs(j - i)
        graph_dist = spd[i, j].item()
        print(f"  - 位置 {i} 到 {j}: 线性距离={linear_dist}, 图距离={graph_dist}")


# ============================================================================
# Part 7: Training Script
# ============================================================================

if __name__ == "__main__":
    # 打印 SPD 示例
    print_spd_example()
    
    # Model configuration
    model_name_or_path = "yangheng/OmniGenome-52M"
    # model_name_or_path = "yangheng/OmniGenome-186M"
    
    # Label mapping for three-class classification
    label2id = {"0": 0, "1": 1, "2": 2}
    
    # Initialize tokenizer
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
    
    # Maximum distance for SPD
    max_distance = 32
    
    # Load datasets with SPD information
    print("\n📊 Loading datasets with SPD computation...")
    datasets = OmniDatasetWithSPD.from_hub(
        "/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10",
        tokenizer=tokenizer,
        max_length=512,
        max_distance=max_distance,
        label2id=label2id,
    )
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")
    
    # Verify SPD data is loaded
    sample = datasets["train"][0]
    print(f"\n📋 Sample data keys: {list(sample.keys())}")
    if "spd_matrix" in sample:
        spd = sample["spd_matrix"]
        print(f"  - spd_matrix shape: {spd.shape}")
        print(f"  - spd_matrix dtype: {spd.dtype}")
        print(f"  - spd_matrix range: [{spd.min().item()}, {spd.max().item()}]")
        
        # 统计距离分布
        unique, counts = torch.unique(spd, return_counts=True)
        print(f"  - Distance distribution (top 5):")
        sorted_idx = counts.argsort(descending=True)[:5]
        for idx in sorted_idx:
            print(f"      Distance {unique[idx].item()}: {counts[idx].item()} pairs")
    
    # Initialize model with Spatial Encoding
    print("\n🔧 Initializing model with Graphormer Spatial Encoding...")
    model = OmniModelWithSpatialEncoding(
        model_name_or_path,
        tokenizer,
        num_labels=len(label2id),
        dataset_class=OmniDatasetWithSPD,
        max_distance=max_distance,
    )
    
    # Print Spatial Encoding parameters
    print(f"\n📐 Spatial Encoding parameters:")
    se = model.structure_attention.spatial_encoding
    print(f"  - spatial_bias shape: {se.spatial_bias.shape}")
    print(f"  - Sample bias values (distance 0-5):")
    for d in range(min(6, max_distance + 1)):
        mean_bias = se.spatial_bias[:, d].mean().item()
        print(f"      Distance {d}: {mean_bias:.4f}")
    
    # Training configuration
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
    
    print("\n🎓 Starting training with Graphormer Spatial Encoding...")
    metrics = trainer.train(
        path_to_save="ogb_te_3class_finetuned_52M_seq_tissue_spd",
        dataset_class=OmniDatasetWithSPD
    )
    
    # Print final learned bias parameters
    print(f"\n📐 Spatial Encoding parameters (after training):")
    se = model.structure_attention.spatial_encoding
    print(f"  - Sample bias values (distance 0-5):")
    for d in range(min(6, max_distance + 1)):
        mean_bias = se.spatial_bias[:, d].mean().item()
        print(f"      Distance {d}: {mean_bias:.4f}")
    
    print('\n✅ Final Metrics:', metrics)
