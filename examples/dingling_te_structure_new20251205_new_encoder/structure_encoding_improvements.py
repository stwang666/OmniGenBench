# -*- coding: utf-8 -*-
"""
Structure Encoding Improvements

解决两个问题：
1. SPD bias 的影响方向：确保距离越近的bias越大，符合生物学直觉
2. 配对类型编码：区分GC、AU、GU等不同配对类型的稳定性

改进点：
- 添加配对类型编码（Base Pair Type Encoding）
- 优化SPD bias初始化策略
- 提供可选的配对稳定性权重
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
import numpy as np


# ============================================================================
# Part 1: Base Pair Type Encoding (碱基对类型编码)
# ============================================================================

# 碱基对稳定性权重（基于氢键数量和自由能）
# GC: 3个氢键，最稳定，ΔG ≈ -3.4 kcal/mol
# AU: 2个氢键，中等稳定，ΔG ≈ -2.1 kcal/mol  
# GU: 2个氢键（wobble配对），较不稳定，ΔG ≈ -1.3 kcal/mol
BASE_PAIR_STABILITY = {
    'GC': 1.0,   # 最稳定，作为基准
    'CG': 1.0,
    'AU': 0.62,  # 2/3.4 ≈ 0.62
    'UA': 0.62,
    'GU': 0.38,  # 1.3/3.4 ≈ 0.38
    'UG': 0.38,
    'AT': 0.62,  # 同AU
    'TA': 0.62,
    'GT': 0.38,  # 同GU
    'TG': 0.38,
}

# 碱基对类型ID映射
BASE_PAIR_TYPE_ID = {
    'GC': 0, 'CG': 0,
    'AU': 1, 'UA': 1, 'AT': 1, 'TA': 1,
    'GU': 2, 'UG': 2, 'GT': 2, 'TG': 2,
    'OTHER': 3,  # 其他配对（较少见）
}


def get_base_pair_type(seq: str, i: int, j: int) -> Tuple[str, float]:
    """
    获取位置 i 和 j 的碱基对类型和稳定性权重
    
    Args:
        seq: RNA序列
        i, j: 配对位置
    
    Returns:
        (pair_type, stability_weight): 配对类型字符串和稳定性权重
    """
    if i >= len(seq) or j >= len(seq):
        return 'OTHER', 0.0
    
    pair = seq[i] + seq[j]
    
    if pair in BASE_PAIR_STABILITY:
        return pair, BASE_PAIR_STABILITY[pair]
    else:
        return 'OTHER', 0.0


def dot_bracket_to_pairing_matrix_with_types(
    structure: str,
    sequence: str,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将点括号表示法转换为配对矩阵和配对类型矩阵
    
    Args:
        structure: 点括号表示法，如 "(((...)))"
        sequence: RNA序列，如 "AUGCAUGC"
        max_length: 最大长度，用于padding
    
    Returns:
        pairing_matrix: (seq_len, seq_len) 0/1矩阵，表示是否配对
        pairing_type_matrix: (seq_len, seq_len) 配对类型矩阵
            - 0: GC/CG (最稳定)
            - 1: AU/UA/AT/TA (中等稳定)
            - 2: GU/UG/GT/TG (较不稳定)
            - 3: 其他或非配对
        pairing_stability_matrix: (seq_len, seq_len) 稳定性权重矩阵
    """
    seq_len = len(structure)
    target_len = max_length if max_length is not None else seq_len
    
    pairing_matrix = torch.zeros(target_len, target_len, dtype=torch.float32)
    pairing_type_matrix = torch.full(
        (target_len, target_len), BASE_PAIR_TYPE_ID['OTHER'], dtype=torch.long
    )
    pairing_stability_matrix = torch.zeros(target_len, target_len, dtype=torch.float32)
    
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
                # 标记配对
                pairing_matrix[i, j] = 1.0
                pairing_matrix[j, i] = 1.0
                
                # 获取碱基对类型和稳定性
                pair_type, stability = get_base_pair_type(sequence, j, i)
                type_id = BASE_PAIR_TYPE_ID.get(pair_type, BASE_PAIR_TYPE_ID['OTHER'])
                
                pairing_type_matrix[i, j] = type_id
                pairing_type_matrix[j, i] = type_id
                pairing_stability_matrix[i, j] = stability
                pairing_stability_matrix[j, i] = stability
    
    return pairing_matrix, pairing_type_matrix, pairing_stability_matrix


# ============================================================================
# Part 2: Improved SPD Bias with Distance Decay
# ============================================================================

class ImprovedSPDStructureBias(nn.Module):
    """
    改进的SPD结构偏置
    
    关键改进：
    1. 确保距离越近，bias越大（符合生物学直觉：距离近更可能形成motif）
    2. 使用更合理的衰减函数
    3. 添加可选的温度缩放，控制softmax的锐度
    """
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 32,
        init_scale: float = 0.1,
        decay_type: str = 'exponential',  # 'exponential' or 'inverse'
        temperature: float = 1.0,  # 温度缩放，控制softmax锐度
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.temperature = temperature
        self.decay_type = decay_type
        
        # 可学习的距离bias
        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
        self._init_bias(init_scale)
    
    def _init_bias(self, scale: float) -> None:
        """
        初始化bias：距离越近，bias越大
        
        策略：
        - 距离0（自身）：最大bias
        - 距离1（直接相邻）：较大bias
        - 距离2+：指数衰减
        
        注意：这些bias会在softmax之前加到attention scores上，
        softmax会进一步放大距离近的位置的attention权重。
        """
        with torch.no_grad():
            for d in range(self.max_distance + 1):
                if d == 0:
                    # 自身：最大bias
                    self.spatial_bias[:, d] = 1.0 * scale
                elif d == 1:
                    # 直接相邻：较大bias（容易形成motif）
                    self.spatial_bias[:, d] = 0.5 * scale
                elif d == 2:
                    # 距离2：中等bias
                    self.spatial_bias[:, d] = 0.3 * scale
                else:
                    # 距离3+：指数衰减
                    if self.decay_type == 'exponential':
                        # 指数衰减：exp(-alpha * (d - 2))
                        alpha = 0.2
                        self.spatial_bias[:, d] = 0.2 * scale * float(np.exp(-alpha * (d - 2)))
                    else:
                        # 逆函数衰减：1 / (1 + beta * (d - 2))
                        beta = 0.3
                        self.spatial_bias[:, d] = 0.2 * scale / (1.0 + beta * (d - 2))
    
    def forward(self, spd_matrix: torch.Tensor) -> torch.Tensor:
        """
        将SPD矩阵转换为attention bias
        
        注意：bias会在softmax之前加到attention scores上。
        由于softmax的指数性质，即使bias的差异较小，也会被放大。
        
        例如：
        - 距离1的bias = 0.5，距离3的bias = 0.1
        - 在softmax后，距离1的attention权重会显著大于距离3
        """
        bsz, L, _ = spd_matrix.shape
        device = spd_matrix.device
        
        # 确保距离值在有效范围内
        spd_clamped = spd_matrix.to(torch.long).clamp(0, self.max_distance)
        
        # 应用温度缩放
        spatial_bias = self.spatial_bias.to(device) / self.temperature
        
        # 索引查表
        spd_flat = spd_clamped.view(-1)
        bias_table = spatial_bias.t()  # (D+1, H)
        bias_flat = bias_table[spd_flat]  # (B*L*L, H)
        
        bias = bias_flat.view(bsz, L, L, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        
        return bias


# ============================================================================
# Part 3: Base Pair Type Encoding Bias
# ============================================================================

class BasePairTypeBias(nn.Module):
    """
    基于配对类型的attention bias
    
    核心思想：
    - GC配对（3个氢键）最稳定，给较大的正bias
    - AU配对（2个氢键）中等稳定，给中等bias
    - GU配对（2个氢键，wobble）较不稳定，给较小的bias
    - 非配对位置：零bias或负bias
    
    公式：
        A_ij = (Q_i K_j^T) / √d + b_spatial(SPD) + b_pair_type(pair_type)
    """
    
    def __init__(
        self,
        num_heads: int,
        num_pair_types: int = 4,  # GC, AU, GU, OTHER
        init_scale: float = 0.1,
        use_stability_weights: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_pair_types = num_pair_types
        self.use_stability_weights = use_stability_weights
        
        # 每种配对类型的可学习bias
        # shape: (num_heads, num_pair_types)
        self.pair_type_bias = nn.Parameter(torch.zeros(num_heads, num_pair_types))
        self._init_bias(init_scale)
    
    def _init_bias(self, scale: float) -> None:
        """
        初始化配对类型bias
        
        策略（基于稳定性）：
        - GC (type 0): 最稳定，bias = 0.5 * scale
        - AU (type 1): 中等稳定，bias = 0.3 * scale
        - GU (type 2): 较不稳定，bias = 0.1 * scale
        - OTHER (type 3): 非配对或其他，bias = 0.0
        """
        with torch.no_grad():
            # GC/CG: 最稳定
            self.pair_type_bias[:, 0] = 0.5 * scale
            # AU/UA/AT/TA: 中等稳定
            self.pair_type_bias[:, 1] = 0.3 * scale
            # GU/UG/GT/TG: 较不稳定
            self.pair_type_bias[:, 2] = 0.1 * scale
            # OTHER: 非配对
            self.pair_type_bias[:, 3] = 0.0
    
    def forward(
        self,
        pairing_matrix: torch.Tensor,
        pairing_type_matrix: torch.Tensor,
        pairing_stability_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        将配对类型矩阵转换为attention bias
        
        Args:
            pairing_matrix: (B, L, L) 0/1矩阵，表示是否配对
            pairing_type_matrix: (B, L, L) 配对类型ID (0=GC, 1=AU, 2=GU, 3=OTHER)
            pairing_stability_matrix: (B, L, L) 可选的稳定性权重矩阵
        
        Returns:
            bias: (B, H, L, L) attention bias
        """
        bsz, L, _ = pairing_matrix.shape
        device = pairing_matrix.device
        
        # 将pair_type_bias移到正确设备
        pair_type_bias = self.pair_type_bias.to(device)  # (H, T)
        
        # 索引查表：根据配对类型获取bias
        type_flat = pairing_type_matrix.view(-1)  # (B*L*L,)
        bias_flat = pair_type_bias.t()[type_flat]  # (B*L*L, H)
        bias = bias_flat.view(bsz, L, L, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        
        # 如果使用稳定性权重，进一步缩放
        if self.use_stability_weights and pairing_stability_matrix is not None:
            stability = pairing_stability_matrix.unsqueeze(-1)  # (B, L, L, 1)
            stability = stability.permute(0, 3, 1, 2)  # (B, 1, L, L)
            # 将稳定性权重作为额外的缩放因子
            bias = bias * (0.5 + 0.5 * stability)  # 在0.5-1.0之间缩放
        
        # 只对配对位置应用bias（非配对位置保持为0）
        pairing_mask = pairing_matrix.unsqueeze(1)  # (B, 1, L, L)
        bias = bias * pairing_mask
        
        return bias


# ============================================================================
# Part 4: Combined Structure Encoding (SPD + Base Pair Type)
# ============================================================================

class CombinedStructureBias(nn.Module):
    """
    组合的结构编码：SPD + 配对类型
    
    公式：
        A_ij = (Q_i K_j^T) / √d + b_spatial(SPD_ij) + b_pair_type(pair_type_ij)
    
    优势：
    1. SPD编码捕获空间距离信息（距离越近，bias越大）
    2. 配对类型编码捕获稳定性信息（GC > AU > GU）
    3. 两者结合提供更丰富的结构信息
    """
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 32,
        init_scale: float = 0.1,
        use_pair_type: bool = True,
    ):
        super().__init__()
        self.use_pair_type = use_pair_type
        
        # SPD编码
        self.spd_bias = ImprovedSPDStructureBias(
            num_heads=num_heads,
            max_distance=max_distance,
            init_scale=init_scale,
        )
        
        # 配对类型编码（可选）
        if use_pair_type:
            self.pair_type_bias = BasePairTypeBias(
                num_heads=num_heads,
                num_pair_types=4,
                init_scale=init_scale,
            )
    
    def forward(
        self,
        spd_matrix: torch.Tensor,
        pairing_matrix: Optional[torch.Tensor] = None,
        pairing_type_matrix: Optional[torch.Tensor] = None,
        pairing_stability_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        组合SPD和配对类型bias
        
        Args:
            spd_matrix: (B, L, L) 最短路径距离矩阵
            pairing_matrix: (B, L, L) 配对矩阵（0/1）
            pairing_type_matrix: (B, L, L) 配对类型矩阵
            pairing_stability_matrix: (B, L, L) 稳定性权重矩阵
        
        Returns:
            total_bias: (B, H, L, L) 组合的attention bias
        """
        # SPD bias
        total_bias = self.spd_bias(spd_matrix)
        
        # 配对类型bias（如果启用）
        if self.use_pair_type and pairing_matrix is not None and pairing_type_matrix is not None:
            pair_bias = self.pair_type_bias(
                pairing_matrix,
                pairing_type_matrix,
                pairing_stability_matrix,
            )
            total_bias = total_bias + pair_bias
        
        return total_bias


# ============================================================================
# Part 5: Usage Example
# ============================================================================

def example_usage():
    """
    使用示例
    """
    # 示例数据
    structure = "(((...)))"
    sequence = "AUGCAUGC"
    max_length = 512
    
    # 1. 计算配对类型矩阵
    pairing_matrix, pairing_type_matrix, pairing_stability_matrix = \
        dot_bracket_to_pairing_matrix_with_types(structure, sequence, max_length)
    
    print("配对矩阵形状:", pairing_matrix.shape)
    print("配对类型矩阵形状:", pairing_type_matrix.shape)
    print("稳定性矩阵形状:", pairing_stability_matrix.shape)
    
    # 2. 创建组合结构bias
    num_heads = 12
    combined_bias = CombinedStructureBias(
        num_heads=num_heads,
        max_distance=32,
        use_pair_type=True,
    )
    
    # 3. 计算bias（需要SPD矩阵，这里用示例）
    batch_size = 2
    seq_len = len(structure)
    spd_matrix = torch.randint(0, 10, (batch_size, max_length, max_length))
    
    # 扩展配对矩阵到batch维度
    pairing_matrix_batch = pairing_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    pairing_type_matrix_batch = pairing_type_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    pairing_stability_matrix_batch = pairing_stability_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    
    total_bias = combined_bias(
        spd_matrix,
        pairing_matrix_batch,
        pairing_type_matrix_batch,
        pairing_stability_matrix_batch,
    )
    
    print("组合bias形状:", total_bias.shape)  # (B, H, L, L)
    print("✅ 使用示例完成")


if __name__ == "__main__":
    example_usage()



