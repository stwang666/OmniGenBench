# Step 1: Data Preparation
import torch
import gc
import sys
import os
import numpy as np
import pickle
import hashlib
from typing import Optional, Dict, List
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import threading
torch.cuda.empty_cache()
gc.collect()

# 导入LMDB
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False
    print("⚠️ LMDB not available, please install: pip install lmdb")

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


# 尝试导入Cython算法，如果失败则使用Python fallback
try:
    from data import algos
    HAS_CYTHON = True
    print("✅ Cython algos 已加载，将使用加速版本")
except ImportError:
    HAS_CYTHON = False
    print("⚠️ Cython algos not available, using Python fallback")
    print("   提示：运行 'cd data && bash build_cython.sh' 来编译Cython加速模块")

model_name_or_path = "yangheng/OmniGenome-52M"
# model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
# dataset_name = "translation_efficiency_prediction"


# ============================================================================
# LMDB缓存管理器：用于高效存储和读取结构信息
# ============================================================================

class LMDBStructureCache:
    """
    使用LMDB存储结构信息的缓存管理器。
    LMDB是内存映射数据库，性能高且只占用一个文件，避免磁盘配额问题。
    """
    
    def __init__(self, cache_path: str = "./structure_cache.lmdb", map_size: int = 50 * 1024**3):
        """
        初始化LMDB缓存
        
        Args:
            cache_path: LMDB数据库文件路径
            map_size: 数据库最大大小（默认50GB）
        """
        if not HAS_LMDB:
            raise ImportError("LMDB is required. Install with: pip install lmdb")
        
        self.cache_path = cache_path
        self.map_size = map_size
        self.env = None
        self._open_env()
    
    def _open_env(self):
        """打开LMDB环境"""
        if self.env is None:
            # 确保目录存在
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # 打开LMDB环境（readonly=False允许写入）
            self.env = lmdb.open(
                self.cache_path,
                map_size=self.map_size,
                readonly=False,
                create=True,
                lock=True,
            )
    
    def _serialize(self, obj: Dict) -> bytes:
        """序列化结构信息（包含torch张量）"""
        # 将torch张量转换为numpy数组以便序列化
        serializable_obj = {}
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                serializable_obj[key] = {
                    '_type': 'torch.Tensor',
                    'data': value.cpu().numpy(),
                    'dtype': str(value.dtype),
                    'shape': list(value.shape),
                }
            else:
                serializable_obj[key] = value
        return pickle.dumps(serializable_obj)
    
    def _deserialize(self, data: bytes) -> Dict:
        """反序列化结构信息（恢复torch张量）"""
        obj = pickle.loads(data)
        # 恢复torch张量
        for key, value in obj.items():
            if isinstance(value, dict) and value.get('_type') == 'torch.Tensor':
                numpy_array = value['data']
                dtype_str = value['dtype']
                # 转换dtype字符串为torch dtype
                if 'int' in dtype_str:
                    if '64' in dtype_str:
                        tensor = torch.from_numpy(numpy_array).long()
                    else:
                        tensor = torch.from_numpy(numpy_array).long()
                elif 'float' in dtype_str:
                    if '64' in dtype_str:
                        tensor = torch.from_numpy(numpy_array).double()
                    else:
                        tensor = torch.from_numpy(numpy_array).float()
                else:
                    tensor = torch.from_numpy(numpy_array)
                obj[key] = tensor
        return obj
    
    def _get_key(self, cache_key: str) -> bytes:
        """将字符串键转换为bytes"""
        # 使用hash确保键长度一致
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return key_hash.encode()
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """从LMDB获取结构信息"""
        if self.env is None:
            self._open_env()
        
        key_bytes = self._get_key(cache_key)
        with self.env.begin() as txn:
            data = txn.get(key_bytes)
            if data is None:
                return None
            return self._deserialize(data)
    
    def put(self, cache_key: str, structure_info: Dict):
        """将结构信息存储到LMDB"""
        if self.env is None:
            self._open_env()
        
        key_bytes = self._get_key(cache_key)
        data = self._serialize(structure_info)
        with self.env.begin(write=True) as txn:
            txn.put(key_bytes, data)
    
    def contains(self, cache_key: str) -> bool:
        """检查缓存中是否存在该键"""
        if self.env is None:
            self._open_env()
        
        key_bytes = self._get_key(cache_key)
        with self.env.begin() as txn:
            return txn.get(key_bytes) is not None
    
    def close(self):
        """关闭LMDB环境"""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# ============================================================================
# 多进程计算辅助函数（需要在类外部定义，以便pickle）
# ============================================================================

def _compute_structure_info_worker(args):
    """
    多进程工作函数：计算单个样本的结构信息
    
    注意：此函数需要在模块级别定义，以便multiprocessing可以pickle它。
    函数内部会重新导入必要的模块，确保在子进程中也能正常工作。
    
    Args:
        args: tuple (sequence, structure, max_spatial_pos)
    
    Returns:
        tuple: (cache_key, structure_info) 或 (cache_key, None) 如果失败
    """
    # 在函数内部导入，确保子进程能访问
    import sys
    import os
    import torch
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from rna_pyg import rna_structure_to_graph_with_edge_types
    from data.wrapper import preprocess_item
    
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
        print(f"⚠️ Error computing structure info for {sequence[:20]}...: {e}")
        return (cache_key, None)


# 自定义数据集类：处理tissue字段并转换为0-8编号，同时计算Graphormer结构信息
class OmniDatasetWithTissue(OmniDatasetForSequenceClassification):
    """
    支持tissue信息和结构信息的数据集类。
    将tissue名称映射为0-8的编号，并计算Graphormer结构信息（SPD、edge paths等）。
    使用多进程并行计算并缓存到LMDB，充分利用多核CPU加速，避免磁盘配额问题。
    """
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, 
                 use_structure=True, max_spatial_pos=32, multi_hop_max_dist=5, 
                 cache_path=None, num_workers=None, **kwargs):
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
        
        # 多进程参数
        if num_workers is None:
            # 默认使用CPU核心数，但至少保留1个核心给系统
            num_workers = max(1, cpu_count() - 1)
        self.num_workers = num_workers
        
        # 使用LMDB缓存替代内存缓存，避免磁盘配额问题和内存占用过大
        if self.use_structure:
            if cache_path is None:
                # 默认缓存路径：在数据集目录下创建structure_cache.lmdb
                if isinstance(dataset_name_or_path, str):
                    cache_path = os.path.join(dataset_name_or_path, "structure_cache.lmdb")
                else:
                    cache_path = "./structure_cache.lmdb"
            self._lmdb_cache = LMDBStructureCache(cache_path=cache_path, map_size=50 * 1024**3)
            print(f"📦 使用LMDB缓存: {cache_path}")
        else:
            self._lmdb_cache = None
        
        self._precomputed = False  # 标记是否已预计算
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 在父类__init__完成后，确保所有tissue_id都是1维张量
        for item in self.data:
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
        
        # 如果启用结构信息，单进程顺序预计算并缓存
        if self.use_structure:
            print(f"\n{'='*60}")
            print(f"🔄 开始预计算结构信息（单进程顺序计算）...")
            print(f"   - 数据集大小: {len(self.data)} 样本")
            print(f"   - Cython加速: {'✅ 已启用' if HAS_CYTHON else '❌ 未启用'}")
            print(f"{'='*60}\n")
            self._precompute_structure_info()
    
    def close_cache(self):
        """关闭LMDB缓存连接"""
        if hasattr(self, '_lmdb_cache') and self._lmdb_cache is not None:
            self._lmdb_cache.close()
    
    def __del__(self):
        """析构函数，确保关闭LMDB缓存"""
        self.close_cache()
    
    def _compute_structure_info_single(self, sequence: str, structure: str = None):
        """
        计算单个样本的Graphormer结构信息
        
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
        从LMDB缓存读取结构信息
        
        Args:
            sequence: RNA序列
            structure: 点括号结构（可选）
        
        Returns:
            dict: 包含 spatial_pos, edge_input, attn_edge_type, attn_bias, x
        """
        if not self.use_structure or structure is None:
            return None
        
        cache_key = f"{sequence}_{structure}"
        
        # 从LMDB缓存读取（预计算后应该都在缓存中）
        if self._lmdb_cache is not None:
            cached_info = self._lmdb_cache.get(cache_key)
            if cached_info is not None:
                return cached_info
            
            # 如果预计算已完成但缓存中没有，说明这个样本计算失败或被跳过
            if self._precomputed:
                return None
            
            # 如果预计算未完成，尝试延迟计算并保存（用于增量场景）
            _, structure_info = self._compute_structure_info_single(sequence, structure)
            if structure_info is not None:
                try:
                    self._lmdb_cache.put(cache_key, structure_info)
                except Exception as e:
                    print(f"⚠️ 保存到LMDB缓存失败: {e}")
            return structure_info
        
        # 如果没有LMDB缓存，直接计算（回退模式，不推荐）
        _, structure_info = self._compute_structure_info_single(sequence, structure)
        return structure_info
    
    def _precompute_structure_info(self):
        """
        多进程并行预计算所有样本的结构信息并缓存到LMDB
        使用LMDB避免多进程OOM和磁盘配额问题（LMDB是单文件，高效且节省空间）
        使用多进程充分利用多核CPU加速计算
        """
        if self._precomputed:
            return
        
        if self._lmdb_cache is None:
            print("⚠️ LMDB缓存未初始化，跳过预计算")
            self._precomputed = True
            return
        
        print(f"📊 收集待处理任务...")
        tasks = []
        for item in self.data:
            if isinstance(item, dict):
                sequence = item.get("seq") or item.get("sequence", "")
                structure = item.get("structure") or item.get("ss", None)
                if structure:
                    tasks.append((sequence, structure))
        
        if not tasks:
            print("⚠️ 没有找到需要处理的样本")
            self._precomputed = True
            return
        
        print(f"   📊 找到 {len(tasks)} 个样本需要计算结构信息")
        
        # 先检查哪些已经缓存
        print(f"   🔍 检查缓存状态...")
        tasks_to_compute = []
        skipped_count = 0
        
        for sequence, structure in tasks:
            cache_key = f"{sequence}_{structure}"
            if self._lmdb_cache.contains(cache_key):
                skipped_count += 1
            else:
                tasks_to_compute.append((sequence, structure))
        
        print(f"   - 已缓存: {skipped_count}")
        print(f"   - 需要计算: {len(tasks_to_compute)}")
        
        if not tasks_to_compute:
            print(f"   ✅ 所有样本已缓存，跳过计算")
            self._precomputed = True
            return
        
        # 使用多进程并行计算
        print(f"   🔧 使用 {self.num_workers} 个进程并行计算并保存到LMDB缓存...")
        
        import time
        start_time = time.time()
        computed_count = 0
        
        # 准备任务参数（添加max_spatial_pos）
        task_args = [
            (seq, struct, self.max_spatial_pos) 
            for seq, struct in tasks_to_compute
        ]
        
        # 使用多进程池并行计算
        # 注意：LMDB写入需要加锁，但LMDB本身是线程安全的，我们使用进程池
        # 每个进程会创建自己的LMDB连接，LMDB内部会处理并发写入
        with Pool(processes=self.num_workers) as pool:
            # 使用imap_unordered以获得更好的进度显示
            results = list(tqdm(
                pool.imap_unordered(_compute_structure_info_worker, task_args),
                total=len(task_args),
                desc="计算结构信息"
            ))
        
        # 批量写入LMDB（在主进程中，避免多进程写入冲突）
        print(f"   💾 保存计算结果到LMDB缓存...")
        for cache_key, structure_info in tqdm(results, desc="保存到缓存"):
            if structure_info is not None:
                try:
                    self._lmdb_cache.put(cache_key, structure_info)
                    computed_count += 1
                except Exception as e:
                    print(f"⚠️ 保存到LMDB缓存失败 ({cache_key[:20]}...): {e}")
        
        elapsed_time = time.time() - start_time
        self._precomputed = True
        print(f"\n✅ 预计算完成！")
        print(f"   - 总任务数: {len(tasks)}")
        print(f"   - 从缓存加载: {skipped_count}")
        print(f"   - 新计算: {computed_count}")
        print(f"   - 成功缓存: {skipped_count + computed_count}")
        print(f"   - 耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        if computed_count > 0:
            print(f"   - 计算速度: {computed_count/elapsed_time:.2f} 样本/秒")
            print(f"   - 加速比: 约 {self.num_workers}x (使用{self.num_workers}个进程)")
        print(f"{'='*60}\n")
        
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据，包括tissue信息和结构信息。
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
        
        # 计算并添加结构信息（从内存缓存读取，已预计算）
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

# Load datasets
# 使用自定义数据集类的 from_hub 方法，它会自动使用已有的结构信息
# 注意：当 structure_in=True 时，需要增加 max_length 以容纳序列+结构信息
# 如果序列长度约512，结构信息长度也约512，建议设置 max_length=1024 或更大
# 但要注意不能超过模型的最大位置嵌入限制（model.config.max_position_embeddings）
datasets = OmniDatasetWithTissue.from_hub(
    '/projects/u5cs/stwang/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/backbone_graphormer_main', # 指定三分类数据目录（使用相对路径）
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
    use_structure=True,  # 启用Graphormer结构编码
    max_spatial_pos=32,  # 最大空间位置距离
    multi_hop_max_dist=5,  # 多跳最大距离
)
# 检查数据集是否加载成功
if datasets is None:
    raise ValueError("❌ 数据集加载失败，datasets为None。请检查数据路径和参数。")

print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    if dataset is None:
        print(f"  ⚠️ {split}: dataset为None，跳过")
        continue
    print(f"  - {split}: {len(dataset)} samples")

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
    device=torch.device("cuda:0"),
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
