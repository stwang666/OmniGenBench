# -*- coding: utf-8 -*-
"""triclass_seq_tissue_structure_backbone.py

三分类任务：序列 + tissue embedding + RNA 二级结构(点括号) 作为 attention bias 注入到 **backbone 每层 self-attention**。

- encoding_type=spd: 只注入 SPD 矩阵
- encoding_type=graphormer: 注入 SPD + edge_path_matrix（路径边类型，Graphormer 风格）
- encoding_type=pairing: 仅注入配对矩阵

用法示例：
    python triclass_seq_tissue_structure_backbone.py --encoding_type spd
    python triclass_seq_tissue_structure_backbone.py --encoding_type graphormer
    
数据目录需包含 train.csv / valid.csv / test.csv，且每行至少包含：
  - seq 或 sequence
  - label 或 labels
  - tissue
  - structure (dot-bracket)
"""

import argparse
import gc
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
import torch.nn as nn

# Ensure we import the repo-local `omnigenbench` package (not the pip-installed one).
# This is critical because we patched `omnigenbench/src/trainer/accelerate_trainer.py`
# in the repo to fix gradient accumulation + gradient clipping.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omnigenbench import (
    AccelerateTrainer,
    Trainer,
    ClassificationMetric,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    OmniTokenizer,
)

from structure_aware_backbone import (
    patch_backbone_with_structure,
    compute_shortest_path_distance,
    compute_spd_and_edge_paths,
    dot_bracket_to_edges,
    EDGE_TYPE_NONE,
    EDGE_TYPE_BACKBONE,
    EDGE_TYPE_PAIR,
)


def _ensure_1d_long(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.to(torch.long)
    if x.ndim == 0:
        x = x.unsqueeze(0)
    elif x.ndim > 1:
        x = x.flatten()[:1]
    return x


def compute_edge_paths_only(
    seq_len: int,
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32,
    max_path_length: int = 8,
) -> torch.Tensor:
    """只计算 Edge 路径矩阵（当 SPD 已经缓存时使用）。
    
    Returns:
        edge_path_matrix: uint8 (seq_len, seq_len, max_path_length)
    """
    from collections import deque
    import numpy as np
    
    # 构建邻接表（带边类型）
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(seq_len)]
    for i in range(seq_len - 1):
        adj[i].append((i + 1, EDGE_TYPE_BACKBONE))
        adj[i + 1].append((i, EDGE_TYPE_BACKBONE))
    for i, j in pair_edges:
        if 0 <= i < seq_len and 0 <= j < seq_len:
            adj[i].append((j, EDGE_TYPE_PAIR))
            adj[j].append((i, EDGE_TYPE_PAIR))
    
    edge_path = np.full((seq_len, seq_len, max_path_length), EDGE_TYPE_NONE, dtype=np.uint8)
    
    for start in range(seq_len):
        dist = [-1] * seq_len
        parent = [-1] * seq_len
        parent_edge = [EDGE_TYPE_NONE] * seq_len
        dist[start] = 0
        q = deque([start])
        
        while q:
            node = q.popleft()
            if dist[node] >= max_distance:
                continue
            for nb, et in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    parent_edge[nb] = et
                    q.append(nb)
        
        for end in range(seq_len):
            if dist[end] == -1:
                continue
            
            # 回溯路径，记录边类型
            if dist[end] > 0 and dist[end] <= max_path_length:
                path_edges: List[int] = []
                cur = end
                while (cur != start) and parent[cur] != -1:
                    path_edges.append(parent_edge[cur])
                    cur = parent[cur]
                # 反转路径 (从 start 到 end)
                path_edges = path_edges[::-1]
                for idx, et in enumerate(path_edges[:max_path_length]):
                    edge_path[start, end, idx] = et
    
    return torch.from_numpy(edge_path)


def _compute_edge_paths_worker(args: Tuple[str, int, int, int]) -> Tuple[int, torch.Tensor]:
    """多进程工作函数：计算单个结构的 Edge 路径矩阵。
    
    Args:
        args: (structure, max_length, max_distance, max_path_length)
    
    Returns:
        (index, edge_path_matrix): 索引和边路径矩阵
    """
    structure, max_length, max_distance, max_path_length = args
    pair_edges = dot_bracket_to_edges(structure)
    seq_len = len(structure)
    
    edge_paths = compute_edge_paths_only(seq_len, pair_edges, max_distance, max_path_length)
    
    # Padding 到 max_length
    if seq_len < max_length:
        padded_edge = torch.zeros((max_length, max_length, max_path_length), dtype=edge_paths.dtype)
        padded_edge[:seq_len, :seq_len, :] = edge_paths
        edge_paths = padded_edge
    elif seq_len > max_length:
        edge_paths = edge_paths[:max_length, :max_length, :]
    
    return edge_paths


def batch_compute_edge_paths_parallel(
    structures: List[str],
    max_length: int,
    max_distance: int = 32,
    max_path_length: int = 8,
    num_workers: Optional[int] = None,
) -> List[torch.Tensor]:
    """使用多进程批量计算 Edge 路径矩阵。
    
    Args:
        structures: 结构字符串列表
        max_length: 最大序列长度
        max_distance: 最大距离
        max_path_length: 最大路径长度
        num_workers: 进程数，None 表示使用 CPU 核心数
    
    Returns:
        edge_path_matrices: Edge 路径矩阵列表
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 保留一个核心
    
    if num_workers == 1 or len(structures) == 0:
        # 单进程模式
        results = []
        for structure in structures:
            edge_paths = _compute_edge_paths_worker((structure, max_length, max_distance, max_path_length))
            results.append(edge_paths)
        return results
    
    # 多进程模式
    args_list = [(s, max_length, max_distance, max_path_length) for s in structures]
    
    with Pool(num_workers) as pool:
        results = pool.map(_compute_edge_paths_worker, args_list)
    
    return results


class OmniDatasetWithStructure(OmniDatasetForSequenceClassification):
    """Dataset: tokenized seq + tissue_id + structure matrices.
    
    支持复用 Interleaved 版本创建的磁盘缓存。
    """
    
    tissue_mapping = {
        "anthers": 0,
        "flag_leaf": 1,
        "florets": 2,
        "grain_06DAA": 3,
        "grain_20DAA": 4,
        "lemma": 5,
        "roots_seedling": 6,
        "seedling_shoot": 7,
        "whole_spikelet": 8,
        # 兼容旧命名
        "root": 0,
        "seedling": 1,
        "leaf": 2,
        "FMI": 3,
        "FOD": 4,
        "Prophase-I-pollen": 5,
        "Tricellular-pollen": 6,
        "flag": 7,
        "grain": 8,
    }
    
    def __init__(
        self,
        dataset_name_or_path,
        tokenizer,
        max_length: int = 512,
        encoding_type: str = "spd",
        max_distance: int = 32,
        max_path_length: int = 8,
        cache_dir: Optional[str] = None,
        split_name: Optional[str] = None,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        self.encoding_type = encoding_type
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.max_length = max_length
        self.num_workers = num_workers
        self._structure_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # 磁盘缓存设置（复用 Interleaved 版本的缓存）
        if cache_dir is None:
            parent = os.path.dirname(dataset_name_or_path) if os.path.isfile(dataset_name_or_path) else dataset_name_or_path
            cache_dir = os.path.join(parent, ".spd_cache")
        self.cache_dir = cache_dir
        self.split_name = split_name
        self._disk_cache: Optional[Dict] = None  # 可以是整数或字符串 key
        self._sample_index = 0
        
        # 尝试加载磁盘缓存（只加载 SPD，不加载 Edge）
        self._disk_cache = self._load_disk_cache()
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)

        # 修正 tissue_id 维度
        for item in self.data:
            if "tissue_id" in item:
                item["tissue_id"] = _ensure_1d_long(item["tissue_id"])
        
        # 如果 encoding_type 是 graphormer，批量预计算 Edge 路径矩阵
        if self.encoding_type == "graphormer":
            self._precompute_edge_paths()

    def _get_cache_path(self) -> str:
        """生成与 Interleaved 版本兼容的缓存路径"""
        split = self.split_name or "unknown"
        # 使用新的缓存格式（没有 BUG 的版本）: spd_cache_{split}_len{max_length}.pt
        filename = f"spd_cache_{split}_len{self.max_length}.pt"
        return os.path.join(self.cache_dir, filename)

    def _load_disk_cache(self) -> Optional[Dict]:
        """加载 Interleaved 版本创建的磁盘缓存"""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            print(f"   ⚠️ 未找到磁盘缓存: {cache_path}")
            return None
        
        print(f"📂 Loading disk cache from: {cache_path}")
        try:
            raw_cache = torch.load(cache_path, weights_only=True)
            # 直接返回原始缓存，保持原有的 key 格式
            print(f"   ✅ Loaded {len(raw_cache)} cached structure matrices")
            return raw_cache
        except Exception as e:
            print(f"   ⚠️ Failed to load cache: {e}")
            return None

    def _save_disk_cache(self):
        """保存磁盘缓存（只保存 SPD，不保存 Edge）"""
        if not self._structure_cache:
            return
        
        cache_path = self._get_cache_path()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"💾 Saving disk cache to: {cache_path} (SPD only, Edge not cached)")
        try:
            # 将内存缓存转换为磁盘缓存格式（只保存 SPD）
            disk_cache = {}
            for cache_key, struct in self._structure_cache.items():
                # 从 cache_key 中提取 structure 字符串
                structure = cache_key.split("|")[0]
                import hashlib
                structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
                
                # 只保存 SPD，不保存 Edge
                cache_entry = {}
                if "spd_matrix" in struct:
                    cache_entry["spd"] = struct["spd_matrix"]
                if "pairing_matrix" in struct:
                    cache_entry["pairing"] = struct["pairing_matrix"]
                # 不保存 edge_path_matrix
            
            torch.save(disk_cache, cache_path)
            print(f"   ✅ Saved {len(disk_cache)} SPD matrices to disk (Edge not cached)")
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def _precompute_edge_paths(self):
        """批量预计算 Edge 路径矩阵（使用多进程）"""
        num_workers_str = str(self.num_workers) if self.num_workers is not None else "auto"
        print(f"\n🔄 Precomputing Edge paths with {num_workers_str} workers...")
        
        # 收集所有需要计算的结构
        structures_to_compute = []
        indices_to_compute = []
        
        for i, item in enumerate(self.data):
            # 从原始数据中获取 structure（可能是 "structure" 或其他字段名）
            structure = item.get("structure") or item.get("Structure") or item.get("dot_bracket")
            if structure is not None and isinstance(structure, str):
                # 检查是否已经有 Edge 路径矩阵
                if "edge_path_matrix" not in item:
                    structures_to_compute.append(structure)
                    indices_to_compute.append(i)
        
        if not structures_to_compute:
            print(f"   ✅ All Edge paths already computed or no structures found")
            return
        
        print(f"   📊 Computing Edge paths for {len(structures_to_compute)} structures...")
        
        # 批量计算
        try:
            edge_paths_list = batch_compute_edge_paths_parallel(
                structures_to_compute,
                self.max_length,
                self.max_distance,
                self.max_path_length,
                self.num_workers,
            )
            
            # 将结果存储到 data 中
            for idx, edge_paths in zip(indices_to_compute, edge_paths_list):
                self.data[idx]["edge_path_matrix"] = edge_paths
            
            print(f"   ✅ Precomputed {len(edge_paths_list)} Edge path matrices")
        except Exception as e:
            print(f"   ⚠️ Failed to precompute Edge paths: {e}")
            print(f"   💡 Edge paths will be computed on-the-fly during training")

    @classmethod
    def from_hub(
        cls,
        dataset_name_or_path,
        tokenizer,
        max_length: int = 512,
        encoding_type: str = "spd",
        max_distance: int = 32,
        max_path_length: int = 8,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        # 设置共享的缓存目录
        cache_dir = os.path.join(dataset_name_or_path, ".spd_cache")
        
        datasets = {}
        for split in ["train", "valid", "test"]:
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
                    cache_dir=cache_dir,
                    split_name=split,
                    num_workers=num_workers,
                    **kwargs,
                )
        return datasets

    def _pad_square(self, m: torch.Tensor, fill: int, max_length: int) -> torch.Tensor:
        L = m.shape[0]
        actual = min(L, max_length)
        out = torch.full((max_length, max_length), fill_value=fill, dtype=m.dtype)
        out[:actual, :actual] = m[:actual, :actual]
        return out
    
    def prepare_input(self, instance: dict, **kwargs) -> dict:
        labels = -100
        tissue_id = 0
        structure = None

        if isinstance(instance, dict):
            sequence = instance.get("seq") or instance.get("sequence")
            label = instance.get("labels")
            if label is None:
                label = instance.get("label")
            labels = label

            tissue_name = instance.get("tissue")
            if tissue_name is not None:
                tissue_id = int(self.tissue_mapping.get(tissue_name, 0))

            structure = instance.get("structure")
        else:
            sequence = instance

        # Tokenize sequence (both branches need this)
        tokenized = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for k in tokenized:
            tokenized[k] = tokenized[k].squeeze(0)

        if labels is not None:
            # Some OmniGenBench dataset loaders may not populate `self.label2id`
            # (you can see the warning "No label2id provided").
            # Our CSV already provides numeric labels (0/1/2), so fall back to int-cast.
            label2id = getattr(self, "label2id", None)
            if isinstance(label2id, dict) and label2id:
                labels = label2id.get(str(labels), -100)
            else:
                labels = int(labels)
                
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        tokenized["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)

        if structure is None:
            self._sample_index += 1
            return tokenized

        # 获取当前样本索引（用于磁盘缓存）
        current_index = self._sample_index
        self._sample_index += 1

        # 优先尝试从磁盘缓存加载
        struct = None
        if self._disk_cache is not None:
            # 尝试两种 key 格式：整数索引和结构哈希
            cache_key = None
            cached = None
            
            # 首先尝试整数索引（旧格式）
            if current_index in self._disk_cache:
                cache_key = current_index
                cached = self._disk_cache[current_index]
            else:
                # 尝试结构哈希（新格式）
                import hashlib
                structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
                if structure_hash in self._disk_cache:
                    cache_key = structure_hash
                    cached = self._disk_cache[structure_hash]
            
            if cached is not None:
                struct = {}
                if isinstance(cached, torch.Tensor):
                    struct["spd_matrix"] = cached
                elif isinstance(cached, dict):
                    if 'spd' in cached:
                        struct["spd_matrix"] = cached['spd']
                    elif 'spd_matrix' in cached:
                        struct["spd_matrix"] = cached['spd_matrix']
                    # 不加载 Edge（Edge 不缓存，实时计算）
        
        # 如果磁盘缓存未命中，尝试内存缓存（只缓存 SPD）
        if struct is None:
            cache_key = f"{structure}|{self.encoding_type}|{self.max_distance}|{self.max_path_length}|{self.max_length}"
            if cache_key in self._structure_cache:
                struct = self._structure_cache[cache_key].copy()  # 复制，避免修改缓存
        
        # 如果都没有，计算结构矩阵（只计算 SPD，Edge 在 _precompute_edge_paths 中批量计算）
        if struct is None:
            pair_edges = dot_bracket_to_edges(structure)
            L = len(structure)
            struct = {}
            if self.encoding_type == "spd":
                spd = compute_shortest_path_distance(L, pair_edges, self.max_distance)
                struct["spd_matrix"] = self._pad_square(spd, self.max_distance, self.max_length)
            elif self.encoding_type == "graphormer":
                # 只计算 SPD，Edge 在 _precompute_edge_paths 中批量计算
                spd = compute_shortest_path_distance(L, pair_edges, self.max_distance)
                struct["spd_matrix"] = self._pad_square(spd, self.max_distance, self.max_length)
                # Edge 路径矩阵应该在 _precompute_edge_paths 中已经计算好了
                # 如果没有，这里实时计算（作为后备）
                if "edge_path_matrix" not in tokenized:
                    edge = compute_edge_paths_only(L, pair_edges, self.max_distance, self.max_path_length)
                    P = edge.shape[-1]
                    padded_edge = torch.zeros((self.max_length, self.max_length, P), dtype=edge.dtype)
                    actual = min(L, self.max_length)
                    padded_edge[:actual, :actual, :] = edge[:actual, :actual, :]
                    struct["edge_path_matrix"] = padded_edge
            elif self.encoding_type == "pairing":
                # 简单配对矩阵
                pairing = torch.zeros((L, L), dtype=torch.float32)
                for i, j in pair_edges:
                    pairing[i, j] = 1.0
                    pairing[j, i] = 1.0
                struct["pairing_matrix"] = self._pad_square(pairing, 0, self.max_length)
            # 保存到内存缓存（只缓存 SPD，不缓存 Edge）
            cache_key = f"{structure}|{self.encoding_type}|{self.max_distance}|{self.max_path_length}|{self.max_length}"
            # 只保存 SPD 到缓存
            spd_cache = {"spd_matrix": struct.get("spd_matrix")}
            if "pairing_matrix" in struct:
                spd_cache["pairing_matrix"] = struct["pairing_matrix"]
            self._structure_cache[cache_key] = spd_cache

        tokenized.update(struct)
        
        # 如果 encoding_type 是 graphormer，确保有 edge_path_matrix
        if self.encoding_type == "graphormer" and "edge_path_matrix" not in tokenized:
            # 尝试从 data 中获取（应该在 _precompute_edge_paths 中已经计算好了）
            edge_path_found = False
            if current_index < len(self.data) and "edge_path_matrix" in self.data[current_index]:
                tokenized["edge_path_matrix"] = self.data[current_index]["edge_path_matrix"]
                edge_path_found = True
            else:
                # 如果索引不匹配（可能因为 shuffle），尝试通过结构匹配
                import hashlib
                structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
                for item in self.data:
                    if "edge_path_matrix" in item:
                        # 检查结构是否匹配（通过比较结构字符串）
                        item_structure = item.get("structure") or item.get("Structure") or item.get("dot_bracket")
                        if item_structure == structure:
                            tokenized["edge_path_matrix"] = item["edge_path_matrix"]
                            edge_path_found = True
                            break
            
            # 如果仍然没有找到，实时计算（后备方案）
            if not edge_path_found:
                pair_edges = dot_bracket_to_edges(structure)
                L = len(structure)
                edge = compute_edge_paths_only(L, pair_edges, self.max_distance, self.max_path_length)
                P = edge.shape[-1]
                padded_edge = torch.zeros((self.max_length, self.max_length, P), dtype=edge.dtype)
                actual = min(L, self.max_length)
                padded_edge[:actual, :actual, :] = edge[:actual, :actual, :]
                tokenized["edge_path_matrix"] = padded_edge
        
        return tokenized
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写 padding 方法，处理结构矩阵（spd_matrix, edge_path_matrix, pairing_matrix）
        这些矩阵不应该被 padding，因为它们已经是固定大小的方阵
        """
        # 临时移除结构相关的键和 tissue_id
        structure_keys = ['spd_matrix', 'edge_path_matrix', 'pairing_matrix']
        saved_data = {key: [] for key in structure_keys + ['tissue_id']}
        
        for item in self.data:
            for key in structure_keys + ['tissue_id']:
                if key in item:
                    saved_data[key].append(item.pop(key))
                else:
                    saved_data[key].append(None)
        
        # 调用父类的 _pad_and_truncate 处理其他字段
        super()._pad_and_truncate(pad_value)
        
        # 恢复结构矩阵和 tissue_id
        for i, item in enumerate(self.data):
            # 恢复 tissue_id
            if saved_data['tissue_id'][i] is not None:
                tid = saved_data['tissue_id'][i]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim == 1:
                    item["tissue_id"] = tid
                else:
                    item["tissue_id"] = tid.flatten()[:1]
            else:
                item["tissue_id"] = torch.tensor([0], dtype=torch.long)
            
            # 恢复结构矩阵（如果存在）
            for key in structure_keys:
                if saved_data[key][i] is not None:
                    item[key] = saved_data[key][i]
                # 如果某个样本没有某个结构矩阵，根据 encoding_type 设置默认值
                elif key == 'spd_matrix' and key not in item:
                    # 默认：线性距离
                    spd = torch.zeros(self.max_length, self.max_length, dtype=torch.uint8)
                    for ii in range(self.max_length):
                        for jj in range(self.max_length):
                            spd[ii, jj] = min(abs(ii - jj), self.max_distance)
                    item[key] = spd
                elif key == 'edge_path_matrix' and key not in item and self.encoding_type == "graphormer":
                    # 默认：全 NONE
                    item[key] = torch.full(
                        (self.max_length, self.max_length, self.max_path_length),
                        EDGE_TYPE_NONE,
                        dtype=torch.uint8
                    )
                elif key == 'pairing_matrix' and key not in item and self.encoding_type == "pairing":
                    item[key] = torch.zeros(
                        (self.max_length, self.max_length),
                        dtype=torch.float32
                    )

    def __del__(self):
        """析构函数：在对象销毁时保存缓存"""
        if hasattr(self, '_structure_cache') and self._structure_cache:
            self._save_disk_cache()


class OmniModelWithStructureBackbone(OmniModelForSequenceClassification):
    """Patch backbone self-attn and add tissue embedding."""
    
    def __init__(
        self,
        config_or_model,
        tokenizer,
        *args,
        encoding_type: str = "spd",
        max_distance: int = 32,
        max_path_length: int = 8,
        share_bias_across_layers: bool = False,
        layers_to_patch: Optional[List[int]] = None,
        init_scale: float = 0.1,
        **kwargs,
    ):
        self.dataset_class = kwargs.pop("dataset_class", OmniDatasetWithStructure)
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        
        hidden_size = self.config.hidden_size
        num_heads = getattr(self.config, "num_attention_heads", None)
        
        self.patched_backbone = patch_backbone_with_structure(
            backbone=self.model,
            structure_bias_type=encoding_type,
            num_heads=num_heads,
            max_distance=max_distance,
            max_path_length=max_path_length,
            share_across_layers=share_bias_across_layers,
            layers_to_patch=layers_to_patch,
            init_scale=init_scale,
        )
        
        self.tissue_embed_dim = hidden_size // 4
        self.tissue_embedding = nn.Embedding(9, self.tissue_embed_dim)
        self.classifier = nn.Linear(hidden_size + self.tissue_embed_dim, self.config.num_labels)
    
    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        
        spd_matrix = inputs.pop("spd_matrix", None)
        edge_path_matrix = inputs.pop("edge_path_matrix", None)
        pairing_matrix = inputs.pop("pairing_matrix", None)
        
        structure_kwargs = {}
        if spd_matrix is not None:
            structure_kwargs["spd_matrix"] = spd_matrix
        if edge_path_matrix is not None:
            structure_kwargs["edge_path_matrix"] = edge_path_matrix
        if pairing_matrix is not None:
            structure_kwargs["pairing_matrix"] = pairing_matrix
        
        if structure_kwargs:
            self.patched_backbone.set_structure_info(**structure_kwargs)
        else:
            self.patched_backbone.clear_structure_info()
        
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        self.patched_backbone.clear_structure_info()
        
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        pooled = self.pooler(inputs, last_hidden_state)
        
        if tissue_id is None:
            tissue_id = torch.zeros((pooled.shape[0],), dtype=torch.long, device=pooled.device)
        else:
            tissue_id = tissue_id.to(pooled.device)
            if tissue_id.ndim > 1:
                tissue_id = tissue_id.squeeze(-1)

        tissue_embed = self.tissue_embedding(tissue_id)
        combined = torch.cat([pooled, tissue_embed], dim=-1)
        logits = self.classifier(combined)  # 注意：不要 softmax，loss 用 CrossEntropy

        # 直接在模型中计算 loss，避免依赖外部 loss_function
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits, "last_hidden_state": last_hidden_state, "labels": labels}
        

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="yangheng/OmniGenome-52M")
    p.add_argument("--data_dir", type=str, default="/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10")
    p.add_argument("--output_dir", type=str, default="ogb_te_3class_finetuned_52M_backbone")

    p.add_argument("--encoding_type", type=str, default="spd", choices=["spd", "graphormer", "pairing"])
    p.add_argument("--max_distance", type=int, default=32)
    p.add_argument("--max_path_length", type=int, default=8)
    p.add_argument("--share_bias", action="store_true")
    p.add_argument("--init_scale", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--structure_lr", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--no_fp16", action="store_true", help="禁用混合精度训练 (用于诊断 NaN 问题)")
    p.add_argument("--num_workers", type=int, default=None, help="多进程计算 Edge 路径的进程数，None 表示自动（CPU核心数-1）")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    torch.cuda.empty_cache()
    gc.collect()

    # OmniTokenizer expects a *base tokenizer object*.
    # If you pass a string path directly (e.g. "yangheng/OmniGenome-52M"),
    # OmniTokenizer.base_tokenizer becomes a string and calling it will crash:
    # TypeError: 'str' object is not callable.
    tokenizer = OmniTokenizer.from_pretrained(args.model)
    
    datasets = OmniDatasetWithStructure.from_hub(
        args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        encoding_type=args.encoding_type,
        max_distance=args.max_distance,
        max_path_length=args.max_path_length,
        num_workers=args.num_workers,
    )

    # OmniDataset may not populate label2id if not explicitly provided.
    # Our CSV labels are numeric (0/1/2), so fall back to a fixed mapping.
    label2id = getattr(datasets["train"], "label2id", None)
    if not isinstance(label2id, dict) or not label2id:
        label2id = {"0": 0, "1": 1, "2": 2}
    model = OmniModelWithStructureBackbone(
        args.model,
        tokenizer,
        num_labels=len(label2id),
        label2id=label2id,
        encoding_type=args.encoding_type,
        max_distance=args.max_distance,
        max_path_length=args.max_path_length,
        share_bias_across_layers=args.share_bias,
        init_scale=args.init_scale,
        dataset_class=OmniDatasetWithStructure,
    )
    
    # differential LR
    structure_param_ids = {id(p) for p in model.patched_backbone.structure_bias_modules.parameters()}
    structure_params = []
    other_params = []
    for _, p in model.named_parameters():
        (structure_params if id(p) in structure_param_ids else other_params).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.learning_rate},
            {"params": structure_params, "lr": args.structure_lr},
        ],
        weight_decay=0.01,
    )

    metric_functions = [
        ClassificationMetric().accuracy_score,
        ClassificationMetric(average="macro").f1_score,
    ]

    # autocast="no" 禁用混合精度，用于诊断 NaN 问题
    autocast_setting = "no" if args.no_fp16 else "float16"
    
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("valid"),
        test_dataset=datasets.get("test"),
        compute_metrics=metric_functions,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        optimizer=optimizer,
        autocast=autocast_setting,
    )
    
    out = f"{args.output_dir}_{args.encoding_type}"
    print(f"\n🎓 Starting training...")
    print(f"   Output directory: {out}")
    print(f"   Mixed precision: {autocast_setting}")
    if args.no_fp16:
        print("   ⚠️ 混合精度已禁用 (--no_fp16)，训练速度会变慢但更稳定")
    
    metrics = trainer.train(path_to_save=out, dataset_class=OmniDatasetWithStructure)
    print(metrics)
    
    # 训练结束后，显式保存所有数据集的缓存
    print("\n💾 Saving structure caches...")
    for split_name, dataset in datasets.items():
        if hasattr(dataset, '_save_disk_cache'):
            print(f"   Saving {split_name} cache...")
            dataset._save_disk_cache()


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()
