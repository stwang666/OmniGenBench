# -*- coding: utf-8 -*-
"""
集成示例：如何将结构编码改进应用到现有代码

这个文件展示了如何修改现有的数据集和模型类来支持：
1. 改进的SPD bias（更好的距离衰减）
2. 配对类型编码（GC、AU、GU等）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from structure_encoding_improvements import (
    dot_bracket_to_pairing_matrix_with_types,
    CombinedStructureBias,
    ImprovedSPDStructureBias,
    BasePairTypeBias,
)


# ============================================================================
# 示例1：修改数据集类以支持配对类型编码
# ============================================================================

def modify_dataset_prepare_input_example():
    """
    示例：如何修改 OmniDatasetWithSPD.prepare_input() 方法
    
    原始代码只使用 structure，现在需要同时使用 sequence 和 structure
    """
    
    # 原始代码（只使用structure）：
    # structure = instance.get("structure", None)
    # pair_edges = dot_bracket_to_edges(structure)
    # spd = compute_shortest_path_distance(seq_len, pair_edges, self.max_distance)
    
    # 改进后的代码（同时使用sequence和structure）：
    def prepare_input_improved(self, instance, **kwargs):
        # ... 其他代码 ...
        
        sequence = instance.get("sequence") or instance.get("seq")
        structure = instance.get("structure")
        
        if structure is not None and sequence is not None:
            # 1. 计算SPD矩阵（原有功能）
            from structure_aware_backbone import (
                dot_bracket_to_edges,
                compute_shortest_path_distance,
            )
            pair_edges = dot_bracket_to_edges(structure)
            seq_len = len(structure)
            spd = compute_shortest_path_distance(seq_len, pair_edges, self.max_distance)
            
            # 2. 计算配对类型矩阵（新增功能）
            pairing_matrix, pairing_type_matrix, pairing_stability_matrix = \
                dot_bracket_to_pairing_matrix_with_types(
                    structure=structure,
                    sequence=sequence,
                    max_length=self.max_length
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
                
                # 配对类型矩阵也需要padding（但已经是max_length大小了）
                # 如果序列被截断，需要处理
                if len(sequence) > self.max_length:
                    pairing_matrix = pairing_matrix[:self.max_length, :self.max_length]
                    pairing_type_matrix = pairing_type_matrix[:self.max_length, :self.max_length]
                    pairing_stability_matrix = pairing_stability_matrix[:self.max_length, :self.max_length]
            
            tokenized_inputs["spd_matrix"] = spd
            tokenized_inputs["pairing_matrix"] = pairing_matrix
            tokenized_inputs["pairing_type_matrix"] = pairing_type_matrix
            tokenized_inputs["pairing_stability_matrix"] = pairing_stability_matrix
        
        return tokenized_inputs
    
    return prepare_input_improved


# ============================================================================
# 示例2：修改模型类以使用组合结构编码
# ============================================================================

class ExampleModelWithCombinedBias(nn.Module):
    """
    示例：如何在模型中使用组合结构编码
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_distance: int = 32,
        use_pair_type: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_pair_type = use_pair_type
        
        # 使用组合结构编码
        self.structure_bias = CombinedStructureBias(
            num_heads=num_heads,
            max_distance=max_distance,
            use_pair_type=use_pair_type,
        )
        
        # 其他层...
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = (hidden_size // num_heads) ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spd_matrix: Optional[torch.Tensor] = None,
        pairing_matrix: Optional[torch.Tensor] = None,
        pairing_type_matrix: Optional[torch.Tensor] = None,
        pairing_stability_matrix: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 🔑 添加组合结构bias（SPD + 配对类型）
        if spd_matrix is not None:
            structure_bias = self.structure_bias(
                spd_matrix,
                pairing_matrix,
                pairing_type_matrix,
                pairing_stability_matrix,
            )
            attn_scores = attn_scores + structure_bias
        
        # Padding mask
        padding_mask = attention_mask[:, None, None, :].float()
        padding_mask = (1.0 - padding_mask) * -10000.0
        attn_scores = attn_scores + padding_mask
        
        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Output
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return attn_output


# ============================================================================
# 示例3：只使用改进的SPD（不添加配对类型）
# ============================================================================

class ExampleModelWithImprovedSPDOnly(nn.Module):
    """
    示例：只使用改进的SPD bias（不添加配对类型编码）
    
    这是最简单的集成方式，只需要替换SPDStructureBias为ImprovedSPDStructureBias
    """
    
    def __init__(self, hidden_size: int, num_heads: int, max_distance: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 使用改进的SPD bias（更好的初始化和温度控制）
        self.spd_bias = ImprovedSPDStructureBias(
            num_heads=num_heads,
            max_distance=max_distance,
            init_scale=0.1,
            decay_type='exponential',  # 或 'inverse'
            temperature=1.0,  # 可以调整来控制softmax锐度
        )
        
        # 其他层...
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = (hidden_size // num_heads) ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spd_matrix: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 🔑 添加改进的SPD bias
        if spd_matrix is not None:
            spd_bias = self.spd_bias(spd_matrix)
            attn_scores = attn_scores + spd_bias
        
        # Padding mask
        padding_mask = attention_mask[:, None, None, :].float()
        padding_mask = (1.0 - padding_mask) * -10000.0
        attn_scores = attn_scores + padding_mask
        
        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Output
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return attn_output


# ============================================================================
# 示例4：验证配对类型提取
# ============================================================================

def test_pairing_type_extraction():
    """测试配对类型提取功能"""
    
    # 测试数据
    structure = "(((...)))"  # 位置0-7配对，位置1-6配对
    sequence = "AUGCAUGC"    # A-U, U-G, G-C, C-A, A-U, U-G, G-C
    
    # 提取配对类型
    pairing_matrix, pairing_type_matrix, pairing_stability_matrix = \
        dot_bracket_to_pairing_matrix_with_types(
            structure=structure,
            sequence=sequence,
            max_length=8
        )
    
    print("=" * 60)
    print("配对类型提取测试")
    print("=" * 60)
    print(f"结构: {structure}")
    print(f"序列: {sequence}")
    print()
    
    # 检查配对位置
    pair_positions = torch.nonzero(pairing_matrix)
    print("配对位置:")
    for pos in pair_positions:
        i, j = pos[0].item(), pos[1].item()
        if i < j:  # 只显示上三角
            pair_type_id = pairing_type_matrix[i, j].item()
            stability = pairing_stability_matrix[i, j].item()
            
            # 配对类型名称
            type_names = {0: "GC", 1: "AU", 2: "GU", 3: "OTHER"}
            type_name = type_names.get(pair_type_id, "UNKNOWN")
            
            # 实际碱基对
            base_pair = sequence[i] + sequence[j]
            
            print(f"  位置 {i}-{j}: {base_pair} ({type_name}), 稳定性={stability:.2f}")
    
    print()
    print("✅ 配对类型提取测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    test_pairing_type_extraction()
    
    # 创建示例模型
    print("\n创建示例模型...")
    model = ExampleModelWithImprovedSPDOnly(
        hidden_size=768,
        num_heads=12,
        max_distance=32
    )
    print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")



