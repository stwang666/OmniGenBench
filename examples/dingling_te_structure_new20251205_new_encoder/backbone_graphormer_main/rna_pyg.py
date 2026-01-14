# -*- coding: utf-8 -*-
"""
RNA to Graph Conversion Utilities

独立模块：将RNA序列和结构转换为PyG图格式（带边类型）
"""

import torch
from torch_geometric.data import Data
from typing import List, Tuple


# ============================================================================
# 边类型定义
# ============================================================================

EDGE_TYPE_BACKBONE = 0   # 骨架边 (i -> i+1)
EDGE_TYPE_GC = 1         # GC配对 (3个氢键)
EDGE_TYPE_AU = 2         # AU配对 (2个氢键)
EDGE_TYPE_GU = 3         # GU配对 (1个氢键，wobble)
EDGE_TYPE_OTHER = 4      # 其他配对（如UU等，较少见）

NUM_EDGE_TYPES = 5  # 总共5种边类型


# ============================================================================
# 辅助函数
# ============================================================================

def dot_bracket_to_edges(structure: str) -> List[Tuple[int, int]]:
    """
    将点括号结构转换为配对边列表
    
    Args:
        structure: 点括号结构字符串（如 "((..))"）
    
    Returns:
        配对边列表，每个元素为 (i, j) 表示位置i和j配对
    """
    edges: List[Tuple[int, int]] = []
    stack: List[int] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            edges.append((j, i))
    return edges


def get_pair_type(seq: str, i: int, j: int) -> int:
    """
    根据序列确定配对类型（边类型）
    
    Args:
        seq: RNA序列
        i, j: 配对位置
    
    Returns:
        边类型ID (EDGE_TYPE_GC, EDGE_TYPE_AU, EDGE_TYPE_GU, EDGE_TYPE_OTHER)
    """
    pair = (seq[i].upper(), seq[j].upper())
    
    # 标准配对
    if pair in [('G', 'C'), ('C', 'G')]:
        return EDGE_TYPE_GC
    elif pair in [('A', 'U'), ('U', 'A')]:
        return EDGE_TYPE_AU
    elif pair in [('G', 'U'), ('U', 'G')]:
        return EDGE_TYPE_GU
    else:
        return EDGE_TYPE_OTHER


# ============================================================================
# RNA结构转图（带边类型）
# ============================================================================

def rna_structure_to_graph_with_edge_types(
    sequence: str,
    structure: str,
) -> Data:
    """
    将RNA序列和二级结构转换为PyG图格式（带边类型）
    
    图的构建：
    - 节点：每个核苷酸位置作为一个节点
    - 节点特征：核苷酸类型编码 (A=0, C=1, G=2, U=3)
    - 边类型1：骨架边（相邻核苷酸之间的连接，i -> i+1）
    - 边类型2-5：配对边（根据配对类型：GC, AU, GU, OTHER）
    
    Args:
        sequence: RNA序列（如 "ACGUACGU"）
        structure: 点括号结构（如 "((..))"），长度必须与序列一致
    
    Returns:
        PyG Data对象，包含：
        - x: 节点特征矩阵 (N, 1)，N为序列长度
        - edge_index: 边索引矩阵 (2, E)，E为边数
        - edge_attr: 边属性矩阵 (E, 1)，每个元素为边类型ID
    
    Example:
        >>> seq = "ACGUACGU"
        >>> struct = "((..))"
        >>> graph = rna_structure_to_graph_with_edge_types(seq, struct)
        >>> print(graph.x.shape)  # (8, 1)
        >>> print(graph.edge_index.shape)  # (2, E)
        >>> print(graph.edge_attr.shape)  # (E, 1)
    """
    seq_len = len(sequence)
    assert len(structure) == seq_len, f"序列长度({seq_len})和结构长度({len(structure)})必须一致"
    
    # 1. 节点特征：核苷酸编码
    nuc_to_id = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    node_features = torch.tensor([
        [nuc_to_id.get(seq.upper(), 0)] for seq in sequence
    ], dtype=torch.long)
    
    # 2. 构建边和边类型
    edge_list = []
    edge_type_list = []
    
    # 2.1 骨架边（i -> i+1）
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])  # 无向图，双向边
        edge_type_list.append(EDGE_TYPE_BACKBONE)
        edge_type_list.append(EDGE_TYPE_BACKBONE)
    
    # 2.2 配对边（从点括号结构提取）
    pair_edges = dot_bracket_to_edges(structure)
    for i, j in pair_edges:
        if 0 <= i < seq_len and 0 <= j < seq_len:
            pair_type = get_pair_type(sequence, i, j)
            
            edge_list.append([i, j])
            edge_list.append([j, i])  # 无向图
            edge_type_list.append(pair_type)
            edge_type_list.append(pair_type)
    
    # 3. 转换为tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_type_list, dtype=torch.long).unsqueeze(-1)  # [num_edges, 1]
    
    # 4. 创建PyG Data对象
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,  # 边类型，用于embedding
    )
    
    return data


# ============================================================================
# 批量处理
# ============================================================================

def batch_rna_structure_to_graph(
    sequences: List[str],
    structures: List[str],
) -> List[Data]:
    """
    批量将RNA序列和结构转换为图
    
    Args:
        sequences: RNA序列列表
        structures: 点括号结构列表，长度必须与sequences一致
    
    Returns:
        PyG Data对象列表
    """
    assert len(sequences) == len(structures), "序列和结构列表长度必须一致"
    
    graphs = []
    for seq, struct in zip(sequences, structures):
        graph = rna_structure_to_graph_with_edge_types(seq, struct)
        graphs.append(graph)
    
    return graphs