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
from typing import Dict, List, Optional

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
        **kwargs,
    ):
        self.encoding_type = encoding_type
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.max_length = max_length
        self._structure_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # 磁盘缓存设置（复用 Interleaved 版本的缓存）
        if cache_dir is None:
            parent = os.path.dirname(dataset_name_or_path) if os.path.isfile(dataset_name_or_path) else dataset_name_or_path
            cache_dir = os.path.join(parent, ".spd_cache")
        self.cache_dir = cache_dir
        self.split_name = split_name
        self._disk_cache: Optional[Dict] = None  # 可以是整数或字符串 key
        self._sample_index = 0
        self._cache_update_count = 0  # 跟踪缓存更新次数
        self._cache_save_interval = 100  # 每 100 次更新保存一次
        
        # 尝试加载磁盘缓存
        self._disk_cache = self._load_disk_cache()
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)

        # 修正 tissue_id 维度
        for item in self.data:
            if "tissue_id" in item:
                item["tissue_id"] = _ensure_1d_long(item["tissue_id"])

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
            return {}
        
        print(f"📂 Loading disk cache from: {cache_path}")
        try:
            raw_cache = torch.load(cache_path, weights_only=True)
            # 直接返回原始缓存，保持原有的 key 格式
            print(f"   ✅ Loaded {len(raw_cache)} cached structure matrices")
            return raw_cache
        except Exception as e:
            print(f"   ⚠️ Failed to load cache: {e}")
            return {}
    
    def _save_disk_cache(self):
        """保存更新后的磁盘缓存"""
        if self._disk_cache is None:
            return
        
        cache_path = self._get_cache_path()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            torch.save(self._disk_cache, cache_path)
            print(f"💾 Saved updated cache ({len(self._disk_cache)} items) to: {cache_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save cache: {e}")
    
    def _save_cache(self):
        """保存结构矩阵到磁盘缓存（包括 SPD 和 edge 信息）"""
        cache_path = self._get_cache_path()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 检查是否已有缓存
        if os.path.exists(cache_path):
            # 加载现有缓存
            try:
                existing_cache = torch.load(cache_path, weights_only=True)
            except:
                existing_cache = {}
        else:
            existing_cache = {}
        
        import hashlib
        cache_updated = False
        
        # 遍历数据，计算并保存结构矩阵
        for i, item in enumerate(self.data):
            structure = item.get("structure")
            if structure is None:
                continue
            
            # 使用结构哈希作为 key（与加载时一致）
            structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
            
            # 如果已存在且包含所需信息，跳过
            if structure_hash in existing_cache:
                cached_item = existing_cache[structure_hash]
                has_spd = isinstance(cached_item, torch.Tensor) or (isinstance(cached_item, dict) and ('spd' in cached_item or 'spd_matrix' in cached_item))
                has_edge = isinstance(cached_item, dict) and 'edge' in cached_item
                
                # 如果 encoding_type 是 graphormer 但缓存中没有 edge，需要重新计算
                if self.encoding_type == "graphormer" and not has_edge:
                    # 需要重新计算
                    pass
                elif has_spd and (self.encoding_type != "graphormer" or has_edge):
                    # 已有完整缓存，跳过
                    continue
            
            # 计算结构矩阵
            pair_edges = dot_bracket_to_edges(structure)
            L = len(structure)
            
            cache_item = {}
            
            if self.encoding_type == "spd":
                spd = compute_shortest_path_distance(L, pair_edges, self.max_distance)
                cache_item['spd_matrix'] = self._pad_square(spd, self.max_distance, self.max_length)
            elif self.encoding_type == "graphormer":
                spd, edge = compute_spd_and_edge_paths(L, pair_edges, self.max_distance, self.max_path_length)
                cache_item['spd_matrix'] = self._pad_square(spd, self.max_distance, self.max_length)
                # edge: (L,L,P)
                P = edge.shape[-1]
                padded_edge = torch.zeros((self.max_length, self.max_length, P), dtype=edge.dtype)
                actual = min(L, self.max_length)
                padded_edge[:actual, :actual, :] = edge[:actual, :actual, :]
                cache_item['edge_path_matrix'] = padded_edge
            
            if cache_item:
                # 转换为与 Interleaved 版本兼容的格式
                saved_item = {}
                if 'spd_matrix' in cache_item:
                    saved_item['spd'] = cache_item['spd_matrix']
                if 'edge_path_matrix' in cache_item:
                    saved_item['edge'] = cache_item['edge_path_matrix']
                
                existing_cache[structure_hash] = saved_item
                cache_updated = True
        
        # 合并 _disk_cache 中的更新（如果有）
        if self._disk_cache is not None and self._cache_update_count > 0:
            # 将 _disk_cache 中的更新合并到 existing_cache
            for key, value in self._disk_cache.items():
                if key not in existing_cache or not isinstance(existing_cache.get(key), dict):
                    existing_cache[key] = value if isinstance(value, dict) else {'spd': value}
                elif isinstance(value, dict) and 'edge' in value:
                    # 确保 edge 信息被保存
                    if not isinstance(existing_cache[key], dict):
                        existing_cache[key] = {'spd': existing_cache[key]}
                    existing_cache[key]['edge'] = value['edge']
            cache_updated = True
        
        # 保存更新后的缓存
        if cache_updated:
            print(f"💾 Saving cache to: {cache_path}")
            torch.save(existing_cache, cache_path)
            print(f"   ✅ Saved {len(existing_cache)} cached structure matrices")
            # 更新 _disk_cache 引用
            self._disk_cache = existing_cache
            self._cache_update_count = 0

    @classmethod
    def from_hub(
        cls,
        dataset_name_or_path,
        tokenizer,
        max_length: int = 512,
        encoding_type: str = "spd",
        max_distance: int = 32,
        max_path_length: int = 8,
        **kwargs,
    ):
        # 设置共享的缓存目录
        cache_dir = os.path.join(dataset_name_or_path, ".spd_cache")
        
        datasets = {}
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(dataset_name_or_path, f"{split}.csv")
            if os.path.exists(split_path):
                print(f"\n📊 Loading {split} split...")
                dataset = cls(
                    split_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    encoding_type=encoding_type,
                    max_distance=max_distance,
                    max_path_length=max_path_length,
                    cache_dir=cache_dir,
                    split_name=split,
                    **kwargs,
                )
                # 保存缓存（包括 SPD 和 edge 信息）
                dataset._save_cache()
                datasets[split] = dataset
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
                    if 'edge' in cached and self.encoding_type == "graphormer":
                        struct["edge_path_matrix"] = cached['edge']
                
                # 检查 graphormer 是否需要计算 edge（缓存中有 SPD 但没有 edge）
                if self.encoding_type == "graphormer" and "edge_path_matrix" not in struct:
                    # 需要计算 edge 信息
                    pair_edges = dot_bracket_to_edges(structure)
                    L = len(structure)
                    _, edge = compute_spd_and_edge_paths(L, pair_edges, self.max_distance, self.max_path_length)
                    # edge: (L,L,P)
                    P = edge.shape[-1]
                    padded_edge = torch.zeros((self.max_length, self.max_length, P), dtype=edge.dtype)
                    actual = min(L, self.max_length)
                    padded_edge[:actual, :actual, :] = edge[:actual, :actual, :]
                    struct["edge_path_matrix"] = padded_edge
                    
                    # 保存 edge 到磁盘缓存
                    if cache_key is not None and self._disk_cache is not None:
                        import hashlib
                        if isinstance(cache_key, str):
                            structure_hash = cache_key
                        else:
                            structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
                        
                        # 更新缓存
                        if structure_hash not in self._disk_cache:
                            self._disk_cache[structure_hash] = {}
                        if not isinstance(self._disk_cache[structure_hash], dict):
                            self._disk_cache[structure_hash] = {'spd': self._disk_cache[structure_hash]}
                        
                        self._disk_cache[structure_hash]['edge'] = padded_edge
                        
                        # 定期保存到磁盘
                        self._cache_update_count += 1
                        if self._cache_update_count % self._cache_save_interval == 0:
                            self._save_disk_cache()
        
        # 如果磁盘缓存未命中，尝试内存缓存
        if struct is None:
            cache_key = f"{structure}|{self.encoding_type}|{self.max_distance}|{self.max_path_length}|{self.max_length}"
            if cache_key in self._structure_cache:
                struct = self._structure_cache[cache_key]
        
        # 如果都没有，计算结构矩阵
        if struct is None:
            pair_edges = dot_bracket_to_edges(structure)
            L = len(structure)
            struct = {}
            if self.encoding_type == "spd":
                spd = compute_shortest_path_distance(L, pair_edges, self.max_distance)
                struct["spd_matrix"] = self._pad_square(spd, self.max_distance, self.max_length)
            elif self.encoding_type == "graphormer":
                spd, edge = compute_spd_and_edge_paths(L, pair_edges, self.max_distance, self.max_path_length)
                struct["spd_matrix"] = self._pad_square(spd, self.max_distance, self.max_length)
                # edge: (L,L,P)
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
            # 保存到内存缓存（只在计算新结构时）
            cache_key = f"{structure}|{self.encoding_type}|{self.max_distance}|{self.max_path_length}|{self.max_length}"
            self._structure_cache[cache_key] = struct

        tokenized.update(struct)
        return tokenized
    
    def __del__(self):
        """析构时保存所有待保存的缓存"""
        if hasattr(self, '_cache_update_count') and self._cache_update_count > 0:
            try:
                self._save_disk_cache()
            except:
                pass  # 忽略析构时的错误


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


if __name__ == "__main__":
    warnings.filterwarnings("default")
    main()

