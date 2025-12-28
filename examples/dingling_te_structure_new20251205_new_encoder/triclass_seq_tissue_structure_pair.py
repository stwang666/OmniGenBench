# -*- coding: utf-8 -*-
"""
RNA Translation Efficiency Classification with Structure-Aware Attention

This script implements a three-class classification model that incorporates:
1. Tissue embedding (9 tissue types)
2. RNA secondary structure information via attention bias (Graphormer-style)

The structure information is injected into the attention mechanism using learnable
bias parameters, following the approach described in Graphormer paper.
"""

# Step 1: Data Preparation
import torch
import gc
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
from typing import Optional, List, Dict, Tuple
import re


# ============================================================================
# Part 1: Structure Parsing Utilities
# ============================================================================

def dot_bracket_to_pairing_matrix(structure: str, max_length: int = None) -> torch.Tensor:
    """
    将点括号表示法转换为配对矩阵
    
    Args:
        structure: 点括号表示法，如 "(((...)))"
            '(' 和 ')' 表示配对
            '.' 表示非配对
        max_length: 最大长度，用于padding
    
    Returns:
        pairing_matrix: (seq_len, seq_len) 的 0/1 矩阵
            M[i,j] = 1 表示位置 i 和 j 配对
            M[i,j] = 0 表示位置 i 和 j 不配对
    
    Example:
        >>> structure = "((...))"
        >>> matrix = dot_bracket_to_pairing_matrix(structure)
        >>> # 位置 0 和 6 配对, 位置 1 和 5 配对
    """
    seq_len = len(structure)
    target_len = max_length if max_length is not None else seq_len
    
    pairing_matrix = torch.zeros(target_len, target_len)
    
    # 使用栈来匹配括号
    stack = []
    for i, char in enumerate(structure):
        if i >= target_len:
            break
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                # 对称填充
                pairing_matrix[i, j] = 1
                pairing_matrix[j, i] = 1
        # '.' 不做处理，保持为 0
    
    return pairing_matrix


def batch_dot_bracket_to_pairing_matrix(
    structures: List[str], 
    max_length: int
) -> torch.Tensor:
    """
    批量将点括号转换为配对矩阵
    
    Args:
        structures: 结构列表
        max_length: 统一的最大长度
    
    Returns:
        pairing_matrices: (batch_size, max_length, max_length)
    """
    matrices = []
    for struct in structures:
        matrix = dot_bracket_to_pairing_matrix(struct, max_length)
        matrices.append(matrix)
    return torch.stack(matrices)


# ============================================================================
# Part 2: Structure Attention Bias Module (Graphormer-style)
# ============================================================================

class StructureAttentionBias(nn.Module):
    """
    将 RNA 配对矩阵转换为可学习的 Attention 偏置
    
    类似 Graphormer 的 Spatial Encoding，使用三类可学习偏置：
    - 配对位置 (paired): 位置 i 和 j 形成碱基对
    - 非配对位置 (unpaired): 位置 i 和 j 不形成碱基对
    - 对角线/自身 (self): 位置 i 对自身的注意力
    
    每个 attention head 有独立的偏置参数，允许不同的 head 学习不同的结构偏好。
    
    公式:
        Attention(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + B_struct) V
    
    其中 B_struct[i,j] = 
        - paired_bias   如果 M[i,j] = 1 (i和j配对)
        - unpaired_bias 如果 M[i,j] = 0 且 i != j
        - self_bias     如果 i == j
    """
    
    def __init__(self, num_heads: int, init_paired: float = 0.1, init_unpaired: float = 0.0):
        """
        初始化结构注意力偏置模块
        
        Args:
            num_heads: 注意力头的数量
            init_paired: 配对位置偏置的初始值（正值鼓励关注配对位置）
            init_unpaired: 非配对位置偏置的初始值
        """
        super().__init__()
        self.num_heads = num_heads
        
        # 三类可学习偏置，每个 head 有独立的参数
        # Shape: (num_heads,)
        self.paired_bias = nn.Parameter(torch.full((num_heads,), init_paired))
        self.unpaired_bias = nn.Parameter(torch.full((num_heads,), init_unpaired))
        self.self_bias = nn.Parameter(torch.zeros(num_heads))
        
    def forward(self, pairing_matrix: torch.Tensor) -> torch.Tensor:
        """
        将配对矩阵转换为 attention bias
        
        Args:
            pairing_matrix: (batch_size, seq_len, seq_len) 
                            1 表示配对，0 表示非配对
        
        Returns:
            attention_bias: (batch_size, num_heads, seq_len, seq_len)
                           可以直接加到 attention scores 上
        """
        batch_size, seq_len, _ = pairing_matrix.shape
        device = pairing_matrix.device
        
        # 创建偏置矩阵
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        bias = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, 
            device=device, dtype=self.paired_bias.dtype
        )
        
        # 对角线 mask: (1, 1, seq_len, seq_len)
        diag_mask = torch.eye(seq_len, device=device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)
        
        # 配对位置 mask: (batch_size, 1, seq_len, seq_len)
        paired_mask = (pairing_matrix == 1).unsqueeze(1)
        
        # 非配对位置 mask (排除对角线): (batch_size, 1, seq_len, seq_len)
        unpaired_mask = (pairing_matrix == 0).unsqueeze(1) & ~diag_mask
        
        # 应用可学习偏置
        # self.paired_bias: (num_heads,) -> (1, num_heads, 1, 1)
        bias = bias + self.self_bias.view(1, -1, 1, 1) * diag_mask.float()
        bias = bias + self.paired_bias.view(1, -1, 1, 1) * paired_mask.float()
        bias = bias + self.unpaired_bias.view(1, -1, 1, 1) * unpaired_mask.float()
        
        return bias
    
    def extra_repr(self) -> str:
        return f'num_heads={self.num_heads}'


# ============================================================================
# Part 3: Custom Dataset with Structure Information
# ============================================================================

class OmniDatasetWithTissueAndStructure(OmniDatasetForSequenceClassification):
    """
    支持 tissue 和 structure 信息的数据集类
    
    从 CSV 中读取:
    - sequence: RNA 序列
    - structure: 点括号表示法的二级结构
    - tissue: 组织类型名称
    - label: 分类标签
    
    返回:
    - input_ids, attention_mask: tokenized 序列
    - tissue_id: 组织类型编号 (0-8)
    - pairing_matrix: 配对矩阵 (seq_len, seq_len)
    - labels: 分类标签
    """
    
    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        # 在调用 super().__init__() 之前初始化映射
        self.tissues = [
            'root', 'seedling', 'leaf', 'FMI', 'FOD',
            'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
        ]
        self.tissue2id = {t: i for i, t in enumerate(self.tissues)}
        self.max_length = max_length
        
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)
        
        # 后处理：确保 tissue_id 和 pairing_matrix 的维度正确
        for item in self.data:
            # 确保 tissue_id 是 1D
            if "tissue_id" in item:
                tid = item["tissue_id"]
                if tid.ndim == 0:
                    item["tissue_id"] = tid.unsqueeze(0)
                elif tid.ndim > 1:
                    item["tissue_id"] = tid.flatten()[:1]
    
    def prepare_input(self, instance, **kwargs):
        """
        准备输入数据，包括 tissue 和 structure 信息
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
        
        # 添加 pairing_matrix (配对矩阵)
        if structure is not None:
            pairing_matrix = dot_bracket_to_pairing_matrix(structure, self.max_length)
            tokenized_inputs["pairing_matrix"] = pairing_matrix
        else:
            # 如果没有结构信息，使用全零矩阵（相当于没有结构偏置）
            tokenized_inputs["pairing_matrix"] = torch.zeros(
                self.max_length, self.max_length
            )
        
        return tokenized_inputs
    
    def _pad_and_truncate(self, pad_value=0):
        """
        重写 _pad_and_truncate 方法，跳过 tissue_id 和 pairing_matrix 的 padding
        """
        # 临时移除不需要 padding 的字段
        tissue_ids = []
        pairing_matrices = []
        
        for item in self.data:
            if "tissue_id" in item:
                tissue_ids.append(item.pop("tissue_id"))
            else:
                tissue_ids.append(None)
            
            if "pairing_matrix" in item:
                pairing_matrices.append(item.pop("pairing_matrix"))
            else:
                pairing_matrices.append(None)
        
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
            
            # 恢复 pairing_matrix
            if pairing_matrices[i] is not None:
                item["pairing_matrix"] = pairing_matrices[i]
            else:
                item["pairing_matrix"] = torch.zeros(
                    self.max_length, self.max_length
                )


# ============================================================================
# Part 4: Model with Structure-Aware Attention
# ============================================================================

class StructureAwareAttentionLayer(nn.Module):
    """
    额外的 Structure-Aware Attention 层
    
    在 backbone 的输出上应用一个带有结构偏置的 attention 层。
    这种方式不需要修改 backbone 模型内部，但仍然可以利用结构信息。
    
    公式:
        Attention(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + B_struct) V
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # 结构偏置模块
        self.structure_bias = StructureAttentionBias(
            num_heads=num_heads,
            init_paired=0.5,     # 配对位置给较大正偏置
            init_unpaired=0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        pairing_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) 1=有效, 0=padding
            pairing_matrix: (batch, seq_len, seq_len) 配对矩阵
        
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
        
        # 🔑 添加结构偏置 (Graphormer-style)
        structure_bias = self.structure_bias(pairing_matrix)  # (batch, heads, seq, seq)
        attn_scores = attn_scores + structure_bias
        
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


class OmniModelWithTissueAndStructure(OmniModelForSequenceClassification):
    """
    支持 tissue 嵌入和结构注意力偏置的序列分类模型
    
    结构信息注入方式:
    - 在 backbone 输出上添加一个额外的 Structure-Aware Attention 层
    - 该层使用 Graphormer-style 的可学习偏置
    - 不修改 backbone 模型内部，兼容性更好
    
    Tissue 嵌入方式 (Late Fusion):
    - 在 pooler 之后将 tissue embedding 与序列特征拼接
    """
    
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        # 保存 dataset_class 参数
        self.dataset_class = kwargs.pop('dataset_class', OmniDatasetWithTissueAndStructure)
        
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        
        # 获取模型配置
        hidden_size = self.config.hidden_size
        num_attention_heads = getattr(self.config, 'num_attention_heads', 12)
        
        # 🔑 Structure-Aware Attention 层
        self.structure_attention = StructureAwareAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
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
        
        print(f"📐 Initialized StructureAwareAttentionLayer with {num_attention_heads} heads")
        print(f"   - paired_bias (init): {self.structure_attention.structure_bias.paired_bias.data.mean().item():.4f}")
        print(f"   - unpaired_bias (init): {self.structure_attention.structure_bias.unpaired_bias.data.mean().item():.4f}")
    
    def forward(self, **inputs):
        """
        Forward pass with structure-aware attention and tissue embedding
        """
        labels = inputs.pop("labels", None)
        tissue_id = inputs.pop("tissue_id", None)
        pairing_matrix = inputs.pop("pairing_matrix", None)
        
        # 保存 attention_mask 用于后续处理
        attention_mask = inputs.get("attention_mask", None)
        
        # 获取 backbone 的 last_hidden_state
        # 这里不修改 attention_mask，直接传入原始的 2D mask
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        
        # 🔑 应用 Structure-Aware Attention 层
        if pairing_matrix is not None and attention_mask is not None:
            # 确保在正确的设备上
            device = last_hidden_state.device
            pairing_matrix = pairing_matrix.to(device)
            attention_mask = attention_mask.to(device)
            
            # 通过结构感知的 attention 层
            last_hidden_state = self.structure_attention(
                last_hidden_state, 
                attention_mask, 
                pairing_matrix
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
# Part 5: Training Script
# ============================================================================

if __name__ == "__main__":
    # Model configuration
    model_name_or_path = "yangheng/OmniGenome-52M"
    # model_name_or_path = "yangheng/OmniGenome-186M"
    
    # Label mapping for three-class classification
    label2id = {"0": 0, "1": 1, "2": 2}
    
    # Initialize tokenizer
    tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)
    
    # Load datasets with structure information
    print("📊 Loading datasets...")
    datasets = OmniDatasetWithTissueAndStructure.from_hub(
        "/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/split_label_all_together_log10",
        tokenizer=tokenizer,
        max_length=512,
        label2id=label2id,
    )
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    for split, dataset in datasets.items():
        print(f"  - {split}: {len(dataset)} samples")
    
    # Verify structure data is loaded
    sample = datasets["train"][0]
    print(f"\n📋 Sample data keys: {list(sample.keys())}")
    if "pairing_matrix" in sample:
        pm = sample["pairing_matrix"]
        num_pairs = (pm > 0).sum().item() // 2
        print(f"  - pairing_matrix shape: {pm.shape}")
        print(f"  - Number of base pairs: {num_pairs}")
    
    # Initialize model with structure-aware attention
    print("\n🔧 Initializing model with structure-aware attention...")
    model = OmniModelWithTissueAndStructure(
        model_name_or_path,
        tokenizer,
        num_labels=len(label2id),
        dataset_class=OmniDatasetWithTissueAndStructure,
    )
    
    # Print structure bias parameters (now inside structure_attention layer)
    print(f"\n📐 Structure Attention Bias parameters (from StructureAwareAttentionLayer):")
    sb = model.structure_attention.structure_bias
    print(f"  - paired_bias (init): {sb.paired_bias.data.mean().item():.4f}")
    print(f"  - unpaired_bias (init): {sb.unpaired_bias.data.mean().item():.4f}")
    print(f"  - self_bias (init): {sb.self_bias.data.mean().item():.4f}")
    
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
    
    print("\n🎓 Starting training with structure-aware attention...")
    metrics = trainer.train(
        path_to_save="ogb_te_3class_finetuned_52M_seq_tissue_structure_pair",
        dataset_class=OmniDatasetWithTissueAndStructure
    )
    
    # Print final learned bias parameters
    print(f"\n📐 Structure Attention Bias parameters (after training):")
    sb = model.structure_attention.structure_bias
    print(f"  - paired_bias: {sb.paired_bias.data.mean().item():.4f}")
    print(f"  - unpaired_bias: {sb.unpaired_bias.data.mean().item():.4f}")
    print(f"  - self_bias: {sb.self_bias.data.mean().item():.4f}")
    
    print('\n✅ Final Metrics:', metrics)
