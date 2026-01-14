# Step 1: Data Preparation
import torch
import gc
import sys
import os
import numpy as np
import pickle
from typing import Optional, Dict, List
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
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

# 导入Graphormer相关模块
sys.path.insert(0, os.path.dirname(__file__))
from rna_pyg import rna_structure_to_graph_with_edge_types, NUM_EDGE_TYPES
from data.wrapper import preprocess_item
from graphormer_layers import GraphAttnBias

# 独立函数用于多进程计算结构信息
def _compute_structure_info_worker(args):
    """
    工作函数：计算单个样本的结构信息（用于多进程）
    
    Args:
        args: (sequence, structure, max_spatial_pos) 元组
    
    Returns:
        tuple: (cache_key, structure_info) 或 (cache_key, None) 如果失败
    """
    sequence, structure, max_spatial_pos = args
    
    if structure is None:
        return (f"{sequence}_{structure}", None)
    
    cache_key = f"{sequence}_{structure}"
    
    try:
        # 1. 将RNA转换为PyG图
        graph = rna_structure_to_graph_with_edge_types(sequence, structure)
        
        # 2. 使用Graphormer的preprocess_item计算结构信息
        item = graph
        item.idx = 0  # 临时idx
        
        # 3. 调用preprocess_item
        processed_item = preprocess_item(item)
        
        # 4. 提取结构信息
        structure_info = {
            'spatial_pos': processed_item.spatial_pos,  # (N, N)
            'edge_input': processed_item.edge_input,    # (N, N, max_dist, edge_dim)
            'attn_edge_type': processed_item.attn_edge_type,  # (N, N, edge_dim)
            'attn_bias': processed_item.attn_bias,      # (N+1, N+1)
            'x': processed_item.x,                      # (N, 1)
        }
        
        # 限制spatial_pos的最大值
        structure_info['spatial_pos'] = torch.clamp(
            structure_info['spatial_pos'], 
            max=max_spatial_pos
        )
        
        return (cache_key, structure_info)
        
    except Exception as e:
        return (cache_key, None)

# 尝试导入Cython算法，如果失败则使用Python fallback
try:
    from data import algos
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("⚠️ Cython algos not available, using Python fallback")

model_name_or_path = "yangheng/OmniGenome-52M"
# model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
# dataset_name = "translation_efficiency_prediction"

# 自定义数据集类：处理tissue字段并转换为0-8编号，同时计算Graphormer结构信息
class OmniDatasetWithTissue(OmniDatasetForSequenceClassification):
    """
    支持tissue信息和结构信息的数据集类。
    将tissue名称映射为0-8的编号，并计算Graphormer结构信息（SPD、edge paths等）。
    
    预计算结构信息的位置：
    1. __init__() 方法中，调用 super().__init__() 后
    2. 如果 use_structure=True，会调用 self._precompute_structure_info()
    3. _precompute_structure_info() 使用多进程并行计算所有样本的结构信息
    4. 计算结果存储在 self._structure_cache 字典中
    5. 后续在 prepare_input() 中通过 _compute_structure_info() 从缓存读取
    """
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, 
                 use_structure=True, max_spatial_pos=32, multi_hop_max_dist=5,
                 structure_cache_file=None, split=None, **kwargs):
        # 在调用super().__init__()之前初始化tissue2id
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        
        # 结构信息相关参数
        self.use_structure = use_structure
        self.max_spatial_pos = max_spatial_pos
        self.multi_hop_max_dist = multi_hop_max_dist
        
        # 缓存结构信息（按需加载，不一次性加载到内存）
        self._structure_cache = {}
        self._precomputed = False  # 标记是否已预计算
        self._cache_file_loaded = False  # 标记缓存文件是否已加载（用于延迟加载）
        
        # 处理structure_cache_file：如果是字典，暂时保存，等待from_hub中处理
        if isinstance(structure_cache_file, dict):
            # 字典形式的缓存文件将在from_hub中处理
            self.structure_cache_file = None
            self._pending_cache_file_dict = structure_cache_file
        else:
            self.structure_cache_file = structure_cache_file
            self._pending_cache_file_dict = None
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 在父类__init__完成后，确保所有tissue_id都是1维张量
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
        
        # ============================================================
        # 🔥 结构信息加载策略（延迟加载）
        # 不在初始化时加载缓存文件，而是在需要时（prepare_input）按需加载
        # 这样可以避免一次性加载所有结构信息到内存
        # ============================================================
        if self.use_structure and self._pending_cache_file_dict is None:
            if self.structure_cache_file and os.path.exists(self.structure_cache_file):
                print(f"\n📂 检测到预计算的结构信息文件: {self.structure_cache_file}")
                print(f"   将在需要时按需加载（延迟加载，节省内存）\n")
            elif self.structure_cache_file:
                print(f"\n⚠️ 指定的缓存文件不存在: {self.structure_cache_file}")
                print(f"   将进行实时计算...\n")
            else:
                print(f"\n🚀 未指定缓存文件，将进行实时计算...\n")
    
    def _compute_structure_info_single(self, sequence: str, structure: str = None):
        """
        计算单个样本的Graphormer结构信息（内部方法，用于并行计算）
        
        Args:
            sequence: RNA序列
            structure: 点括号结构（可选）
        
        Returns:
            tuple: (cache_key, structure_info) 或 (cache_key, None) 如果失败
        """
        if not self.use_structure or structure is None:
            return (f"{sequence}_{structure}", None)
        
        cache_key = f"{sequence}_{structure}"
        
        try:
            # 1. 将RNA转换为PyG图
            graph = rna_structure_to_graph_with_edge_types(sequence, structure)
            
            # 2. 使用Graphormer的preprocess_item计算结构信息
            item = graph
            item.idx = 0  # 临时idx
            
            # 3. 调用preprocess_item
            processed_item = preprocess_item(item)
            
            # 4. 提取结构信息
            structure_info = {
                'spatial_pos': processed_item.spatial_pos,  # (N, N)
                'edge_input': processed_item.edge_input,    # (N, N, max_dist, edge_dim)
                'attn_edge_type': processed_item.attn_edge_type,  # (N, N, edge_dim)
                'attn_bias': processed_item.attn_bias,      # (N+1, N+1)
                'x': processed_item.x,                      # (N, 1)
            }
            
            # 限制spatial_pos的最大值
            structure_info['spatial_pos'] = torch.clamp(
                structure_info['spatial_pos'], 
                max=self.max_spatial_pos
            )
            
            return (cache_key, structure_info)
            
        except Exception as e:
            print(f"⚠️ Error computing structure info for {sequence[:20]}...: {e}")
            return (cache_key, None)
    
    def _compute_structure_info(self, sequence: str, structure: str = None):
        """
        计算Graphormer结构信息（按需从磁盘缓存加载或实时计算）
        
        这个方法在 prepare_input() 中被调用，用于获取结构信息。
        逻辑：
        1. 如果缓存文件存在但未加载，先加载缓存文件（延迟加载）
        2. 从内存缓存中读取（如果已加载）
        3. 如果缓存中没有，尝试实时计算
        
        Args:
            sequence: RNA序列
            structure: 点括号结构（可选）
        
        Returns:
            dict: 包含 spatial_pos, edge_input, attn_edge_type, attn_bias, x
        """
        if not self.use_structure or structure is None:
            return None
        
        # ============================================================
        # 🔥 延迟加载：如果缓存文件存在但未加载，现在加载
        # ============================================================
        if not self._cache_file_loaded and self.structure_cache_file and os.path.exists(self.structure_cache_file):
            print(f"\n📂 首次需要结构信息，正在从磁盘加载缓存: {self.structure_cache_file}")
            self._load_structure_cache_from_file()
            print(f"✅ 加载完成！缓存了 {len(self._structure_cache)} 个样本的结构信息\n")
        
        cache_key = f"{sequence}_{structure}"
        
        # 优先从内存缓存读取
        if cache_key in self._structure_cache:
            return self._structure_cache[cache_key]
        
        # 如果已加载缓存但缓存中没有，说明这个样本没有结构信息或计算失败
        if self._cache_file_loaded:
            # 缓存已加载但找不到，可能是新样本或计算失败
            # 尝试实时计算一次
            _, structure_info = self._compute_structure_info_single(sequence, structure)
            if structure_info:
                self._structure_cache[cache_key] = structure_info
            return structure_info
        
        # 没有缓存文件，实时计算（这种情况应该避免，因为会拖慢数据加载）
        if len(self._structure_cache) == 0:
            print(f"⚠️ 警告：没有缓存文件，正在实时计算结构信息（这会导致数据加载变慢）")
        _, structure_info = self._compute_structure_info_single(sequence, structure)
        if structure_info:
            self._structure_cache[cache_key] = structure_info
        return structure_info
    
    def _load_structure_cache_from_file(self):
        """从文件加载预计算的结构信息（延迟加载）"""
        try:
            with open(self.structure_cache_file, 'rb') as f:
                self._structure_cache = pickle.load(f)
            self._precomputed = True
            self._cache_file_loaded = True
            print(f"   成功加载 {len(self._structure_cache)} 个样本的结构信息")
        except Exception as e:
            print(f"   ⚠️ 加载结构信息文件失败: {e}")
            print(f"   将进行实时计算...")
            self._precomputed = False
            self._cache_file_loaded = False
            self._structure_cache = {}
    
    def _precompute_structure_info(self, num_workers=None):
        """
        ============================================================
        🔥 预计算所有样本的结构信息（使用多进程加速）
        ============================================================
        
        这是预计算的核心方法，执行流程：
        1. 扫描数据源（self.examples 或 self.data），收集所有需要计算的 (sequence, structure) 对
        2. 使用多进程池（Pool）并行调用 _compute_structure_info_worker 计算结构信息
        3. 将计算结果存入 self._structure_cache 字典
        4. 设置 self._precomputed = True 标记预计算完成
        
        Args:
            num_workers: 并行进程数，默认为 CPU 核心数的 1/4 到 1/2
        """
        if self._precomputed:
            return
        
        print(f"🔄 预计算结构信息（使用多进程加速）...")
        
        # ============================================================
        # 步骤1: 收集所有需要计算结构信息的样本
        # ============================================================
        # 注意：需要从原始数据源收集，而不是从已处理的 self.data
        tasks = []
        # 尝试从 self.examples 或原始数据源获取
        data_source = getattr(self, 'examples', None) or getattr(self, 'data', None) or []
        
        print(f"   正在扫描数据源（共 {len(data_source)} 个样本）...")
        for idx, item in enumerate(data_source):
            if isinstance(item, dict):
                # 尝试多种可能的字段名
                sequence = (item.get("seq") or item.get("sequence") or 
                           item.get("text") or item.get("dna") or item.get("rna") or "")
                structure = (item.get("structure") or item.get("ss") or 
                            item.get("Structure") or item.get("STRUCTURE") or None)
                
                if structure and sequence:
                    tasks.append((sequence, structure))
            # 每处理1000个样本打印一次进度
            if (idx + 1) % 1000 == 0:
                print(f"   📋 已扫描 {idx + 1}/{len(data_source)} 个样本，找到 {len(tasks)} 个需要计算结构信息的样本...")
        
        if not tasks:
            print("   ⚠️ 没有找到需要计算结构信息的样本")
            self._precomputed = True
            return
        
        print(f"   📊 找到 {len(tasks)} 个样本需要计算结构信息")
        
        # ============================================================
        # 步骤2: 使用多进程并行计算
        # ============================================================
        # 根据CPU核心数和任务特性自动调整进程数
        # 结构信息计算是CPU密集型，可以充分利用多核
        if num_workers is None:
            # 优先使用SLURM分配的核心数，如果没有则使用物理核心数
            # SLURM_CPUS_PER_TASK 是SLURM实际分配给任务的核心数
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus:
                total_cpus = int(slurm_cpus)
                print(f"    从SLURM环境变量检测到 {total_cpus} 个CPU核心（实际分配）")
            else:
                total_cpus = cpu_count()
                print(f"    从系统检测到 {total_cpus} 个CPU核心（物理核心数）")
            
            # 对于CPU密集型任务，可以使用更多进程
            # 但避免过度并行导致上下文切换开销和内存不足
            # 使用 CPU核心数的 1/2 到 3/4，但不超过实际分配的核心数
            # 同时设置上限避免OOM（每个进程可能占用50-200MB内存）
            num_workers = min(total_cpus, max(16, int(total_cpus * 0.75)))
            print(f"    使用 {num_workers} 个进程并行计算（限制在分配核心数的75%以避免OOM）")
        
        print(f"   🔧 使用 {num_workers} 个进程并行计算...")
        
        # 准备任务参数（包含 max_spatial_pos）
        task_args = [(seq, struct, self.max_spatial_pos) for seq, struct in tasks]
        
        # 创建进程池并计算（使用进度条显示）
        import time
        start_time = time.time()
        
        # 使用默认的进程池
        with Pool(processes=num_workers) as pool:
            results = []
            # 使用 imap_unordered 可以更快看到结果
            for idx, result in enumerate(pool.imap_unordered(_compute_structure_info_worker, task_args)):
                results.append(result)
                if (idx + 1) % 1000 == 0 or (idx + 1) == len(task_args):
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    remaining = (len(task_args) - idx - 1) / rate if rate > 0 else 0
                    print(f"   ⏳ 进度: {idx + 1}/{len(task_args)} ({100*(idx+1)/len(task_args):.1f}%) "
                          f"[{elapsed:.1f}s, {rate:.1f} it/s, 剩余 {remaining/60:.1f} 分钟]")
        
        # ============================================================
        # 步骤3: 将结果存入缓存
        # ============================================================
        cached_count = 0
        for cache_key, structure_info in results:
            if structure_info is not None:
                self._structure_cache[cache_key] = structure_info
                cached_count += 1
        
        total_time = time.time() - start_time
        self._precomputed = True
        print(f"   ✅ 预计算完成！成功计算 {cached_count}/{len(tasks)} 个样本的结构信息")
        print(f"   ⏱️  总耗时: {total_time/60:.1f} 分钟，平均速度: {len(tasks)/total_time:.1f} it/s")
        
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据，包括tissue信息和结构信息。
        
        在这个方法中，通过 _compute_structure_info() 从预计算的缓存中读取结构信息。
        """
        if (self.use_structure and 
        not self._precomputed and 
        hasattr(self, 'examples') and len(self.examples) > 0 and len(self._structure_cache) == 0):  # 确保是第一次调用
            print(f"\n检测到数据加载, 开始预计算结构信息...")
            print(f"   当前 examples 数量: {len(self.examples)}")
            self._precompute_structure_info()
            print(f"预计算完成！缓存了 {len(self._structure_cache)} 个样本的结构信息\n")

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
            
            # 获取tissue信息
            tissue_name = instance.get("tissue", None)
            if tissue_name:
                tissue_id = self.tissue2id.get(tissue_name, 0)
            
            # 获取结构信息
            structure = instance.get("structure", None) or instance.get("ss", None)

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
        
        # 添加tissue_id
        if tissue_id is not None:
            tokenized_inputs["tissue_id"] = torch.tensor([tissue_id], dtype=torch.long)
        else:
            tokenized_inputs["tissue_id"] = torch.tensor([0], dtype=torch.long)
        
        # 计算并添加结构信息（从预计算的缓存中读取）
        if self.use_structure and structure:
            structure_info = self._compute_structure_info(sequence, structure)
            if structure_info:
                tokenized_inputs["spatial_pos"] = structure_info["spatial_pos"]
                tokenized_inputs["edge_input"] = structure_info["edge_input"]
                tokenized_inputs["attn_edge_type"] = structure_info["attn_edge_type"]
                tokenized_inputs["attn_bias"] = structure_info["attn_bias"]
                tokenized_inputs["graph_x"] = structure_info["x"]
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写_pad_and_truncate方法，跳过tissue_id和结构信息的padding处理。
        """
        # 临时移除tissue_id和结构信息
        tissue_ids = []
        structure_infos = []
        
        for item in self.data:
            if "tissue_id" in item:
                tissue_ids.append(item.pop("tissue_id"))
            else:
                tissue_ids.append(None)
            
            structure_info = {}
            for key in ["spatial_pos", "edge_input", "attn_edge_type", "attn_bias", "graph_x"]:
                if key in item:
                    structure_info[key] = item.pop(key)
            structure_infos.append(structure_info if structure_info else None)
        
        # 调用父类的_pad_and_truncate处理其他字段
        super()._pad_and_truncate(pad_value)
        
        # 恢复tissue_id和结构信息
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
            
            # 恢复结构信息（不需要padding，因为每个样本的结构信息大小不同）
            if structure_infos[i] is not None:
                for key, value in structure_infos[i].items():
                    item[key] = value
    
    @classmethod
    def from_hub(cls, dataset_name_or_path, tokenizer, max_length=None, 
                 structure_cache_file=None, **kwargs):
        """
        重写from_hub方法，支持为每个split传递对应的缓存文件
        
        Args:
            dataset_name_or_path: 数据集路径
            tokenizer: tokenizer
            max_length: 最大长度
            structure_cache_file: 可以是单个文件路径，也可以是字典 {split: file_path}
            **kwargs: 其他参数
        """
        # 如果structure_cache_file是字典，需要在调用父类from_hub之前处理
        # 但父类的from_hub会调用__init__，所以我们在__init__中已经处理了
        # 这里只需要确保传递正确的参数
        
        # 调用父类的from_hub方法获取数据集
        # 注意：这里需要将structure_cache_file传递给每个split的__init__
        # 但由于from_hub内部会为每个split创建实例，我们需要在创建时传递不同的参数
        # 最简单的方法是：先调用父类，然后为每个split设置对应的缓存文件
        
        # 如果structure_cache_file是字典，先传递字典给__init__，然后在from_hub中处理
        if isinstance(structure_cache_file, dict):
            kwargs['structure_cache_file'] = structure_cache_file
        else:
            kwargs['structure_cache_file'] = structure_cache_file
        
        # 调用父类的from_hub方法获取数据集
        datasets = super().from_hub(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 如果structure_cache_file是字典，为每个split的数据集设置对应的缓存文件路径
        # 注意：不在这里加载，而是在需要时按需加载（延迟加载）
        if isinstance(structure_cache_file, dict):
            for split, dataset in datasets.items():
                if hasattr(dataset, '_pending_cache_file_dict') and dataset._pending_cache_file_dict:
                    if split in dataset._pending_cache_file_dict:
                        cache_file = dataset._pending_cache_file_dict[split]
                        dataset.structure_cache_file = cache_file
                        if dataset.use_structure and os.path.exists(cache_file):
                            print(f"📂 {split}数据集：检测到缓存文件 {cache_file}，将在需要时按需加载\n")
                        elif dataset.use_structure:
                            print(f"⚠️ {split}数据集的缓存文件不存在: {cache_file}，将进行实时计算\n")
                    # 清除临时字典
                    dataset._pending_cache_file_dict = None
        
        return datasets


# ============================================================================
# Graphormer Attention Wrapper: 使用forward hook注入结构bias
# ============================================================================

class GraphormerAttentionWrapper(nn.Module):
    """
    包装预训练模型的attention层，注入Graphormer-style的结构bias
    
    工作原理：
    1. Hook是PyTorch的机制，允许在forward/backward时插入自定义逻辑
    2. 我们直接重写forward方法，复制原始attention层的逻辑
    3. 在计算attention scores后，softmax之前，加上结构bias
    4. 参考structure_aware_backbone.py的实现方式
    5. 支持Pre-LN结构（Layer Normalization在attention之前）
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        graph_attn_bias: GraphAttnBias,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.graph_attn_bias = graph_attn_bias
        self.layer_idx = layer_idx
        self._batched_data: Optional[Dict[str, torch.Tensor]] = None
        
    def set_batched_data(self, batched_data: Dict[str, torch.Tensor]):
        """设置结构信息"""
        self._batched_data = batched_data
    
    def clear_batched_data(self):
        """清除结构信息"""
        self._batched_data = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        """
        重写forward方法，在计算attention scores后加上结构bias
        
        参考structure_aware_backbone.py的实现
        支持Pre-LN结构（如果原始attention有layer_norm，会在计算Q/K/V之前应用）
        """
        attn = self.original_attention
        
        # 检查是否有Pre-LN（某些模型在attention内部有layer_norm）
        # 如果有，先应用layer_norm
        if hasattr(attn, 'layer_norm') or hasattr(attn, 'LayerNorm'):
            layer_norm = getattr(attn, 'layer_norm', None) or getattr(attn, 'LayerNorm', None)
            if layer_norm is not None:
                hidden_states = layer_norm(hidden_states)
        
        # 计算Q, K, V
        mixed_query_layer = attn.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = attn.transpose_for_scores(attn.key(encoder_hidden_states))
            value_layer = attn.transpose_for_scores(attn.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = attn.transpose_for_scores(attn.key(hidden_states))
            value_layer = attn.transpose_for_scores(attn.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = attn.transpose_for_scores(attn.key(hidden_states))
            value_layer = attn.transpose_for_scores(attn.value(hidden_states))
        
        query_layer = attn.transpose_for_scores(mixed_query_layer)
        
        if getattr(attn, "is_decoder", False):
            past_key_value = (key_layer, value_layer)
        
        # Scale
        query_layer = query_layer * (attn.attention_head_size ** -0.5)

        # Rotary / relative position if exists
        if getattr(attn, "position_embedding_type", None) == "rotary" and hasattr(attn, "rotary_embeddings"):
            query_layer, key_layer = attn.rotary_embeddings(query_layer, key_layer)
        
        # 计算attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # 注入Graphormer结构bias
        if self._batched_data is not None:
            try:
                # 计算Graphormer attention bias
                graph_attn_bias = self.graph_attn_bias(self._batched_data)
                # graph_attn_bias: (batch, num_heads, seq_len+1, seq_len+1)
                # 去掉graph token维度: (batch, num_heads, seq_len, seq_len)
                structure_bias = graph_attn_bias[:, :, 1:, 1:]
                
                # 确保形状匹配
                if structure_bias.shape == attention_scores.shape:
                    attention_scores = attention_scores + structure_bias.to(attention_scores.dtype)
            except Exception as e:
                print(f"⚠️ Error injecting structure bias in layer {self.layer_idx}: {e}")

        # 应用attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attn.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 应用到values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        context_layer = attn.dense(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if getattr(attn, "is_decoder", False):
            outputs = outputs + (past_key_value,)
        
        return outputs


def patch_backbone_with_graphormer(
    model: nn.Module,
    num_heads: int,
    num_edges: int = NUM_EDGE_TYPES,
    num_spatial: int = 33,  # 0-32
    num_edge_dis: int = 6,  # 0-5 (multi_hop_max_dist=5时)
    max_spatial_pos: int = 32,
    multi_hop_max_dist: int = 5,
    edge_type: str = "multi_hop",
    share_bias_across_layers: bool = True,
) -> nn.Module:
    """
    为模型的每一层attention注入Graphormer结构bias，并修改为Pre-LN结构
    
    Args:
        model: 预训练模型
        num_heads: attention head数量
        num_edges: 边类型数量（默认5：BACKBONE, GC, AU, GU, OTHER）
        num_spatial: 空间位置编码数量（默认33：0-32）
        num_edge_dis: 边距离编码数量（默认6：0-5，对应multi_hop_max_dist=5）
        max_spatial_pos: 最大空间位置
        multi_hop_max_dist: 多跳最大距离
        edge_type: 边类型（"multi_hop" 或 "single_hop"）
        share_bias_across_layers: 是否在所有层共享同一个GraphAttnBias
    
    Returns:
        包装后的模型
    """
    # 创建GraphAttnBias模块
    if share_bias_across_layers:
        graph_attn_bias = GraphAttnBias(
            num_heads=num_heads,
            num_atoms=4,  # A, C, G, U
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            hidden_dim=model.config.hidden_size if hasattr(model, 'config') else 768,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=1,  # 不影响参数数量
        )
    else:
        graph_attn_bias = None
    
    # 找到所有attention层并包装
    attention_layers = []
    layer_idx = 0
    
    # 遍历模型找到attention层
    # HuggingFace模型通常在encoder.layers[i].attention.self中
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
        if hasattr(encoder, 'layer'):
            layers = encoder.layer
        elif hasattr(encoder, 'layers'):
            layers = encoder.layers
        else:
            layers = None
        
        if layers is not None:
            for i, layer in enumerate(layers):
                if hasattr(layer, 'attention'):
                    # 获取原始attention模块
                    attn_module = layer.attention
                    if hasattr(attn_module, 'self'):
                        original_self_attn = attn_module.self
                    else:
                        original_self_attn = attn_module
                    
                    # 创建或获取GraphAttnBias
                    if not share_bias_across_layers:
                        graph_attn_bias = GraphAttnBias(
                            num_heads=num_heads,
                            num_atoms=4,
                            num_edges=num_edges,
                            num_spatial=num_spatial,
                            num_edge_dis=num_edge_dis,
                            hidden_dim=model.config.hidden_size if hasattr(model, 'config') else 768,
                            edge_type=edge_type,
                            multi_hop_max_dist=multi_hop_max_dist,
                            n_layers=1,
                        )
                    
                    # 创建包装器
                    wrapper = GraphormerAttentionWrapper(
                        original_attention=original_self_attn,
                        graph_attn_bias=graph_attn_bias,
                        layer_idx=layer_idx,
                    )
                    
                    # 使用setattr正确替换模块
                    if hasattr(attn_module, 'self'):
                        setattr(attn_module, 'self', wrapper)
                    else:
                        setattr(layer, 'attention', wrapper)
                    
                    attention_layers.append((f"encoder.layer.{i}.attention", wrapper))
                    layer_idx += 1

    
    print(f" Patched {len(attention_layers)} attention layers with Graphormer bias")
    
    return model


# 自定义模型类：添加tissue嵌入层和Graphormer结构bias
class OmniModelForSequenceClassificationWithTissue(OmniModelForSequenceClassification):
    """
    支持tissue嵌入的序列分类模型。
    将tissue嵌入拼接到last_hidden_state（在pooler之前，每个token位置都会包含tissue信息）。
    """
    
    def __init__(self, config_or_model, tokenizer, *args, 
                 use_structure=True, max_spatial_pos=32, multi_hop_max_dist=5, **kwargs):
        # 保存 dataset_class 参数
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithTissue)
        
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        # Tissue embedding
        self.tissue_embed_dim = self.config.hidden_size // 4
        self.tissue_embedding = nn.Embedding(
            num_embeddings=9,  # 0-8共9个tissue
            embedding_dim=self.tissue_embed_dim
        )
        
        # 重新定义classifier
        self.classifier = nn.Linear(
            self.config.hidden_size + self.tissue_embed_dim,
            self.config.num_labels
        )
        
        # Graphormer结构bias
        self.use_structure = use_structure
        if self.use_structure:
            num_heads = getattr(self.config, 'num_attention_heads', 12)
            num_encoder_layers = getattr(self.config, 'num_hidden_layers', 
                                         getattr(self.config, 'num_layers', 12))
            
            # 为backbone注入Graphormer结构bias
            print(f"\n Patching backbone with Graphormer structure encoding...")
            self.patched_backbone = patch_backbone_with_graphormer(
                model=self,
                num_heads=num_heads,
                num_edges=NUM_EDGE_TYPES,
                num_spatial=max_spatial_pos + 1,  # 0到max_spatial_pos
                num_edge_dis=multi_hop_max_dist + 1,  # 0到multi_hop_max_dist
                max_spatial_pos=max_spatial_pos,
                multi_hop_max_dist=multi_hop_max_dist,
                edge_type="multi_hop",
                share_bias_across_layers=True,  # 共享参数以减少参数量
            )
            
            # 统计新增参数
            structure_params = sum(p.numel() for name, p in self.named_parameters() 
                                 if 'graph_attn_bias' in name and p.requires_grad)
            print(f"    Graphormer structure encoding parameters: {structure_params:,}")
        else:
            self.patched_backbone = None

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        
        # 提取结构信息并设置到attention层
        structure_kwargs = {}
        if self.use_structure and self.patched_backbone is not None:
            spatial_pos = inputs.pop("spatial_pos", None)
            edge_input = inputs.pop("edge_input", None)
            attn_edge_type = inputs.pop("attn_edge_type", None)
            attn_bias = inputs.pop("attn_bias", None)
            graph_x = inputs.pop("graph_x", None)
            
            if spatial_pos is not None:
                # 构建batched_data
                batch_size = spatial_pos.shape[0] if spatial_pos.ndim > 2 else 1
                if spatial_pos.ndim == 2:
                    spatial_pos = spatial_pos.unsqueeze(0)
                
                batched_data = {
                    'spatial_pos': spatial_pos.to(self.device),
                    'attn_bias': attn_bias.to(self.device) if attn_bias is not None 
                               else torch.zeros(spatial_pos.shape[0], spatial_pos.shape[1]+1, 
                                               spatial_pos.shape[1]+1, device=self.device),
                    'x': graph_x.to(self.device) if graph_x is not None 
                        else torch.zeros(spatial_pos.shape[0], spatial_pos.shape[1], 1, 
                                       dtype=torch.long, device=self.device),
                }
                
                if edge_input is not None:
                    if edge_input.ndim == 4:
                        edge_input = edge_input.unsqueeze(0)
                    batched_data['edge_input'] = edge_input.to(self.device)
                if attn_edge_type is not None:
                    if attn_edge_type.ndim == 3:
                        attn_edge_type = attn_edge_type.unsqueeze(0)
                    batched_data['attn_edge_type'] = attn_edge_type.to(self.device)
                
                # 设置结构信息到所有attention层
                for name, module in self.named_modules():
                    if isinstance(module, GraphormerAttentionWrapper):
                        module.set_batched_data(batched_data)
        
        # 1. 获取序列的 last_hidden_state (不拼接 Tissue)
        # Shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        
        # 2. 直接池化序列特征 (使用原始 hidden_size，无需切片)
        # Shape: (batch_size, hidden_size)
        pooled_state = self.pooler(inputs, last_hidden_state)
        
        # 3. 获取 Tissue Embedding (无需扩展到 seq_len)
        if tissue_id is not None:
            if tissue_id.ndim > 1:
                tissue_id = tissue_id.squeeze(-1)
            # Shape: (batch_size, tissue_embed_dim)
            tissue_embed = self.tissue_embedding(tissue_id)
        else:
            batch_size = last_hidden_state.shape[0]
            tissue_embed = torch.zeros(batch_size, self.tissue_embed_dim, device=last_hidden_state.device)

        # 4. 在池化后进行拼接 (Late Fusion)
        # Shape: (batch_size, hidden_size + tissue_embed_dim)
        combined_features = torch.cat([pooled_state, tissue_embed], dim=-1)
        
        # 5. 分类
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            # labels shape: [batch, 1]
            # Flatten for CrossEntropyLoss
            logits_flat = logits.view(-1, self.config.num_labels)  # [batch, num_classes]
            labels_flat = labels.view(-1)  # [batch * 1]

            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = self.loss_fn(logits_flat, labels_flat)
        
        # 清除结构信息
        if self.use_structure and self.patched_backbone is not None:
            for name, module in self.named_modules():
                if isinstance(module, GraphormerAttentionWrapper):
                    module.clear_batched_data()
        
        outputs = {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": last_hidden_state, 
        }
        return outputs
    
    # def forward(self, **inputs):
    #     """
    #     Forward pass with tissue embedding.
    #     在pooler之前将tissue嵌入拼接到last_hidden_state的每个token位置。
    #     """
    #     labels = inputs.pop("labels", None)
    #     tissue_id = inputs.pop("tissue_id", None)
        
    #     # 获取last_hidden_state: (batch_size, seq_len, hidden_size)
    #     last_hidden_state = self.last_hidden_state_forward(**inputs)
    #     last_hidden_state = self.dropout(last_hidden_state)
    #     last_hidden_state = self.activation(last_hidden_state)
        
    #     # 获取tissue嵌入并扩展到每个token位置
    #     if tissue_id is not None:
    #         # 处理tissue_id可能是[batch_size, 1]的情况（DataLoader批处理后的形状）
    #         if tissue_id.ndim > 1:
    #             tissue_id = tissue_id.squeeze(-1)  # 压缩为[batch_size]
    #         tissue_embed = self.tissue_embedding(tissue_id)  # (batch_size, tissue_embed_dim)
    #         # 扩展到每个token位置: (batch_size, 1, tissue_embed_dim) -> (batch_size, seq_len, tissue_embed_dim)
    #         seq_len = last_hidden_state.shape[1]
    #         tissue_embed_expanded = tissue_embed.unsqueeze(1).expand(-1, seq_len, -1)
    #         # 拼接到last_hidden_state的每个token位置
    #         last_hidden_state = torch.cat([last_hidden_state, tissue_embed_expanded], dim=-1)  # (batch_size, seq_len, hidden_size + tissue_embed_dim)
    #     else:
    #         # 如果没有tissue_id，使用零向量
    #         batch_size, seq_len = last_hidden_state.shape[0], last_hidden_state.shape[1]
    #         device = last_hidden_state.device
    #         zero_tissue_embed = torch.zeros(batch_size, seq_len, self.tissue_embed_dim, device=device)
    #         last_hidden_state = torch.cat([last_hidden_state, zero_tissue_embed], dim=-1)
        
    #     # 池化操作：需要修改pooler来处理扩展后的hidden_size
    #     # 为了兼容现有的pooler，我们创建一个临时的inputs，但pooler内部会使用原始的hidden_size部分
    #     # 实际上，我们需要自定义pooler逻辑
    #     # 简单方法：先对原始部分进行池化，然后拼接tissue嵌入
    #     # 但这样tissue信息不会影响pooling过程
        
    #     # 更好的方法：创建一个自定义的pooling，或者修改pooler的输入
    #     # 为了简化，我们使用原始hidden_size部分进行pooling，然后拼接tissue嵌入
    #     original_hidden_size = self.config.hidden_size
    #     original_hidden_state = last_hidden_state[:, :, :original_hidden_size]
    #     pooled_state = self.pooler(inputs, original_hidden_state)  # (batch_size, hidden_size)
        
    #     # 从拼接后的hidden_state中提取tissue嵌入（取第一个token位置的tissue嵌入，因为所有token的tissue嵌入相同）
    #     tissue_embed_pooled = last_hidden_state[:, 0, original_hidden_size:]  # (batch_size, tissue_embed_dim)
        
    #     # 拼接pooled state和tissue嵌入
    #     last_hidden_state = torch.cat([pooled_state, tissue_embed_pooled], dim=-1)  # (batch_size, hidden_size + tissue_embed_dim)
        
    #     # 分类
    #     logits = self.classifier(last_hidden_state)
    #     logits = self.softmax(logits)
        
    #     outputs = {
    #         "logits": logits,
    #         "last_hidden_state": last_hidden_state,
    #         "labels": labels,
    #     }
    #     return outputs


# # 自定义数据集类：使用已有的结构信息，而不是自动预测
# class OmniDatasetWithExistingStructure(OmniDatasetForSequenceClassification):
#     """
#     使用数据中已有的结构信息的数据集类。
    # 如果数据中包含 'structure' 字段，直接使用；否则回退到自动预测。
    # """
    
    # def _preprocessing(self):
    #     """
    #     重写预处理方法：优先使用数据中已有的结构信息。
    #     """
    #     for idx, ex in enumerate(self.examples):
    #         # 处理不同的序列字段名（SEQ, seq, sequence, text）
    #         if "SEQ" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["SEQ"]
    #             del self.examples[idx]["SEQ"]
    #         if "seq" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["seq"]
    #             del self.examples[idx]["seq"]
    #         if "text" in self.examples[idx]:
    #             self.examples[idx]["sequence"] = self.examples[idx]["text"]
    #             del self.examples[idx]["text"]

    #         if "sequence" not in self.examples[idx]:
    #             import warnings
    #             warnings.warn("The 'sequence' field is missing in the raw dataset.")
        
    #     if len(self.examples) > 0 and "sequence" in self.examples[0]:
    #         sequences = [ex["sequence"] for ex in self.examples]
    #         if self.structure_in:
    #             # 检查数据中是否已经包含结构信息（支持不同的大小写）
    #             has_structure = False
    #             structure_key = None
    #             for key in ["structure", "Structure", "STRUCTURE"]:
    #                 if key in self.examples[0]:
    #                     has_structure = True
    #                     structure_key = key
    #                     break
                
    #             if has_structure:
    #                 # 使用已有的结构信息
    #                 for idx, ex in enumerate(self.examples):
    #                     structure = ex.get(structure_key, "")
    #                     sequence = ex["sequence"]
    #                     self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"
    #             else:
    #                 # 如果没有结构信息，则自动预测（回退到原始行为）
    #                 structures = self.rna2structure.fold(sequences)
    #                 for idx, (sequence, structure) in enumerate(zip(sequences, structures)):
    #                     self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"



# Model and Tokenizer

# We define the label mapping in the training
label2id = {"0": 0, "1": 1, "2": 2}  # 0/1: 原始标签, 2: 填充的空label样本

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

# ============================================================
# 🔥 预计算结构信息的触发点
# ============================================================
# 当调用 OmniDatasetWithTissue.from_hub() 时：
# 1. 会创建 train/valid/test 三个数据集实例
# 2. 每个实例在 __init__() 中会调用 _precompute_structure_info()
# 3. 预计算会使用多进程并行计算所有样本的结构信息
# 4. 计算结果存储在 self._structure_cache 中
# ============================================================
print(f"\n{'='*60}")
print(f"🚀 开始加载数据集并预计算结构信息...")
print(f"{'='*60}\n")

# Load datasets
# 使用自定义数据集类的 from_hub 方法，它会自动使用已有的结构信息
# 注意：当 structure_in=True 时，需要增加 max_length 以容纳序列+结构信息
# 如果序列长度约512，结构信息长度也约512，建议设置 max_length=1024 或更大
# 但要注意不能超过模型的最大位置嵌入限制（model.config.max_position_embeddings）
# 预计算结构信息文件路径（如果存在，将直接加载，避免训练时计算）
# 为train/valid/test分别指定预计算文件
structure_cache_files = {
    'train': 'structure_cache_train.pkl',
    'valid': 'structure_cache_valid.pkl',
    'test': 'structure_cache_test.pkl',
}

# 检查哪些文件存在
available_cache_files = {
    split: path for split, path in structure_cache_files.items() 
    if os.path.exists(path)
}

if available_cache_files:
    print(f"📂 找到预计算结构信息文件: {list(available_cache_files.keys())}")
else:
    print(f"⚠️ 未找到预计算结构信息文件，将在加载数据集时实时计算")

datasets = OmniDatasetWithTissue.from_hub(
    '/home/u5cs/stwang.u5cs/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/backbone_graphormer_main', # 指定三分类数据目录（使用相对路径）
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
    use_structure=True,  # 启用Graphormer结构编码
    max_spatial_pos=32,  # 最大空间位置距离
    multi_hop_max_dist=5,  # 多跳最大距离
    structure_cache_file=available_cache_files,  # 传递字典，为每个split指定对应的缓存文件
)

print(f"\n{'='*60}")
print(f"✅ 数据集加载完成！")
print(f"{'='*60}\n")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")
    if hasattr(dataset, '_structure_cache'):
        print(f"    └─ 缓存了 {len(dataset._structure_cache)} 个样本的结构信息")

# Step 2: Model Initialization
# === Model Initialization ===
# We support all genomic foundation models from Hugging Face Hub.
model = OmniModelForSequenceClassificationWithTissue(
    model_name_or_path,
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Three-class classification
    dataset_class=OmniDatasetWithTissue,  # 传递自定义数据集类，用于模型保存和加载
    use_structure=True,  # 启用Graphormer结构编码
    max_spatial_pos=32,
    multi_hop_max_dist=5,
)

# 统计新增参数
def count_structure_parameters(model):
    """统计Graphormer结构编码新增的参数数量"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计GraphAttnBias的参数
    structure_params = 0
    for name, module in model.named_modules():
        if isinstance(module, GraphAttnBias):
            structure_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    return {
        'total_trainable_params': total_params,
        'structure_encoding_params': structure_params,
        'other_params': total_params - structure_params,
    }

param_stats = count_structure_parameters(model)
print(f"\n📊 Parameter Statistics:")
print(f"   Total trainable: {param_stats['total_trainable_params']:,}")
print(f"   Structure encoding: {param_stats['structure_encoding_params']:,}")
print(f"   Other: {param_stats['other_params']:,}")

# Step 3: Model Training
metric_functions = [
    ClassificationMetric().accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric(average='macro').f1_score]


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
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # 早停和最佳模型配置
    # early_stopping=True,
    # patience=8,  # 8个epoch验证集准确率不提升就停止
    monitor='valid_accuracy_score',  # 监控验证集准确率
    load_best_model_at_end=True,  # 训练结束自动加载最佳模型
)
print("🎓 Starting training...")

# metrics = trainer.train()
# trainer.save_model("ogb_te_finetuned")

# # trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save="ogb_te_3class_finetuned_52M_seq_tissue_new20251205_log10", dataset_class=OmniDatasetWithTissue)
print('Final Metrics:', metrics)


# # Step 4: Model Inference and Interpretation
# inference_model = ModelHub.load("yangheng/ogb_te_finetuned")

# sample_sequences = {
#     "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
#     "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
#     "Random sequence": "AUGC" * (128 // 4),
# }
# for seq_name, sequence in sample_sequences.items():
#     outputs = inference_model.inference(sequence)

#     # —— Result Interpretation ——
#     prediction = outputs['predictions']
#     confidence = outputs['confidence']
#     print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")
