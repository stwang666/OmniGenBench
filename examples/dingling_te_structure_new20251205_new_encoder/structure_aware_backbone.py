# -*- coding: utf-8 -*-
"""structure_aware_backbone.py

把 RNA 二级结构(点括号)编码成 **attention bias**，并直接 patch OmniGenome/Transformer backbone 的 self-attention。

目标是实现类似 Graphormer 的：
  A_ij = (QK^T)/sqrt(d) + b_phi(SPD_ij) + c_ij(edge-path)

- SPDStructureBias: 只用最短路径距离 SPD -> 每个(head, distance)一个可学习 scalar
- GraphormerStructureBias: SPD + Edge encoding (路径上每条边的类型 + 位置权重)
- PairingStructureBias: 仅区分配对/不配对

注意：如果检测到 FlashAttention 路径(某些模型会有 flash_attn_func)，当前 wrapper 会回退到原始 attention，
即不注入结构 bias。
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Edge types for path encoding
EDGE_TYPE_NONE = 0
EDGE_TYPE_BACKBONE = 1
EDGE_TYPE_PAIR = 2
NUM_EDGE_TYPES = 3


def dot_bracket_to_edges(structure: str) -> List[Tuple[int, int]]:
    """Convert dot-bracket to pairing edges."""
    edges: List[Tuple[int, int]] = []
    stack: List[int] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
                j = stack.pop()
                edges.append((j, i))
    return edges


def compute_shortest_path_distance(
    seq_len: int, 
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32,
) -> torch.Tensor:
    """Compute SPD matrix on RNA graph (backbone + pairing). Returns uint8 (seq_len, seq_len)."""
    adj: List[List[int]] = [[] for _ in range(seq_len)]
    for i in range(seq_len - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    for i, j in pair_edges:
        if 0 <= i < seq_len and 0 <= j < seq_len:
            adj[i].append(j)
            adj[j].append(i)
    
    spd = np.full((seq_len, seq_len), max_distance, dtype=np.uint8)
    for start in range(seq_len):
        dist = [-1] * seq_len
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            if dist[node] >= max_distance:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        for end in range(seq_len):
            if dist[end] != -1:
                spd[start, end] = min(dist[end], max_distance)
    return torch.from_numpy(spd)


def compute_spd_and_edge_paths(
    seq_len: int,
    pair_edges: List[Tuple[int, int]],
    max_distance: int = 32,
    max_path_length: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute SPD + edge-path matrix.
    
    Returns:
      - spd: uint8 (L, L)
      - edge_path: uint8 (L, L, P) where P=max_path_length, each entry in {NONE,BACKBONE,PAIR}
    """
    # adjacency with edge types
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(seq_len)]
    for i in range(seq_len - 1):
        adj[i].append((i + 1, EDGE_TYPE_BACKBONE))
        adj[i + 1].append((i, EDGE_TYPE_BACKBONE))
    for i, j in pair_edges:
        if 0 <= i < seq_len and 0 <= j < seq_len:
            adj[i].append((j, EDGE_TYPE_PAIR))
            adj[j].append((i, EDGE_TYPE_PAIR))
    
    spd = np.full((seq_len, seq_len), max_distance, dtype=np.uint8)
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
            d = min(dist[end], max_distance)
            spd[start, end] = d
                
            # reconstruct path edges (start -> end)
            path_edges: List[int] = []
            cur = end
            while (cur != start) and parent[cur] != -1:
                path_edges.append(parent_edge[cur])
                cur = parent[cur]
            # reverse to start->end
            path_edges = path_edges[::-1]
            for idx, et in enumerate(path_edges[:max_path_length]):
                edge_path[start, end, idx] = et
    
    return torch.from_numpy(spd), torch.from_numpy(edge_path)


class StructureBiasBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """Return bias (B, H, L, L)"""
    
    @abstractmethod
    def get_required_inputs(self) -> List[str]:
        ...


class SPDStructureBias(StructureBiasBase):
    """Learnable scalar bias per (head, distance)."""
    
    def __init__(self, num_heads: int, max_distance: int = 32, init_scale: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
        self._init_bias(init_scale)
    
    def _init_bias(self, scale: float) -> None:
        with torch.no_grad():
            for d in range(self.max_distance + 1):
                if d == 0:
                    self.spatial_bias[:, d] = 0.5 * scale
                elif d == 1:
                    self.spatial_bias[:, d] = 0.3 * scale
                elif d == 2:
                    self.spatial_bias[:, d] = 0.2 * scale
                else:
                    self.spatial_bias[:, d] = 0.1 * scale * float(np.exp(-0.1 * (d - 2)))

    def get_required_inputs(self) -> List[str]:
        return ["spd_matrix"]
    
    def forward(self, spd_matrix: torch.Tensor, **_: Any) -> torch.Tensor:
        # spd_matrix: (B, L, L)
        bsz, L, _ = spd_matrix.shape
        
        # 关键修复：正确处理不同的数据类型
        # spd_matrix 可能是 uint8（从缓存加载）或其他类型
        # 先转换为 long，然后再 clamp
        if spd_matrix.dtype == torch.uint8:
            # 从缓存加载的是 uint8，直接转 long
            spd = spd_matrix.to(torch.long)
        else:
            # 其他类型，先 clamp 再转 long
            spd = torch.clamp(spd_matrix, min=0, max=self.max_distance).to(torch.long)
        
        # 再次确保范围正确
        spd = torch.clamp(spd, min=0, max=self.max_distance)
        
        # 强制在 FP32 下进行索引操作
        bias_table = self.spatial_bias.float().t()  # (D+1, H) in FP32
        
        # 确保 bias_table 大小正确
        if bias_table.shape[0] != self.max_distance + 1:
            print(f"[SPD ERROR] bias_table has wrong size: {bias_table.shape}, expected ({self.max_distance + 1}, {self.num_heads})")
            # 返回零 bias
            return torch.zeros((bsz, self.num_heads, L, L), device=spd_matrix.device, dtype=torch.float32)
        
        # 安全索引
        try:
            spd_flat = spd.reshape(-1)
            # 使用 F.embedding 替代直接索引，更安全
            import torch.nn.functional as F
            bias_flat = F.embedding(spd_flat, bias_table)  # (B*L*L, H)
        except Exception as e:
            print(f"[SPD ERROR] Indexing failed: {e}")
            try:
                print(f"[SPD ERROR] spd shape: {spd.shape}, dtype: {spd.dtype}")
                print(f"[SPD ERROR] bias_table shape: {bias_table.shape}")
            except:
                pass
            # 回退到零 bias
            return torch.zeros((bsz, self.num_heads, L, L), device=spd_matrix.device, dtype=torch.float32)
            
        bias = bias_flat.view(bsz, L, L, self.num_heads).permute(0, 3, 1, 2)
        return bias


class GraphormerStructureBias(StructureBiasBase):
    """Graphormer-style: spatial bias + edge-path encoding."""
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 32,
        max_path_length: int = 8,
        init_scale: float = 0.1,
        use_spatial: bool = True,
        use_edge: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.use_spatial = use_spatial
        self.use_edge = use_edge
        
        if use_spatial:
            self.spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
            self._init_spatial_bias(init_scale)
        if use_edge:
            # (E, H)
            self.edge_type_embedding = nn.Parameter(torch.zeros(NUM_EDGE_TYPES, num_heads))
            self.position_weights = nn.Parameter(torch.ones(max_path_length, num_heads))
            self._init_edge_bias(init_scale)
    
    def _init_spatial_bias(self, scale: float) -> None:
        with torch.no_grad():
            for d in range(self.max_distance + 1):
                if d == 0:
                    self.spatial_bias[:, d] = 0.5 * scale
                elif d == 1:
                    self.spatial_bias[:, d] = 0.3 * scale
                elif d == 2:
                    self.spatial_bias[:, d] = 0.2 * scale
                else:
                    self.spatial_bias[:, d] = 0.1 * scale * float(np.exp(-0.1 * (d - 2)))
    
    def _init_edge_bias(self, scale: float) -> None:
        with torch.no_grad():
            # NONE=0 -> 0, backbone/pair small non-zero
            self.edge_type_embedding[EDGE_TYPE_NONE].fill_(0.0)
            self.edge_type_embedding[EDGE_TYPE_BACKBONE].fill_(0.05 * scale)
            self.edge_type_embedding[EDGE_TYPE_PAIR].fill_(0.10 * scale)
            # position_weights default 1

    def get_required_inputs(self) -> List[str]:
        req = ["spd_matrix"] if self.use_spatial else []
        if self.use_edge:
            req.append("edge_path_matrix")
        return req

    def forward(self, spd_matrix: torch.Tensor, edge_path_matrix: Optional[torch.Tensor] = None, **_: Any) -> torch.Tensor:
        # spd_matrix: (B, L, L)
        bsz, L, _ = spd_matrix.shape
        device = spd_matrix.device
        dtype = torch.float32

        total_bias = torch.zeros((bsz, self.num_heads, L, L), device=device, dtype=dtype)

        if self.use_spatial:
            # 使用 .clamp() 而非 .clamp_() 避免 in-place 修改
            spd = spd_matrix.clone().to(torch.long).clamp(0, self.max_distance)
            # 强制 FP32 避免 AMP 下索引操作出 NaN
            bias_table = self.spatial_bias.float().t()  # (D+1, H)
            spatial_flat = bias_table[spd.reshape(-1)]
            spatial = spatial_flat.view(bsz, L, L, self.num_heads).permute(0, 3, 1, 2)
            total_bias = total_bias + spatial
        
        if self.use_edge and edge_path_matrix is not None:
            # edge_path_matrix: (B, L, L, P)
            P = edge_path_matrix.shape[-1]
            P = min(P, self.max_path_length)
            edge_ids = edge_path_matrix[..., :P].to(torch.long)

            # 强制 FP32 避免 AMP 下索引操作出 NaN
            edge_emb = self.edge_type_embedding.float()[edge_ids]
            # pos_w: (P, H) -> (1,1,1,P,H)
            pos_w = self.position_weights[:P].float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted = edge_emb * pos_w

            # mask NONE edges
            mask = (edge_ids != EDGE_TYPE_NONE).unsqueeze(-1)
            weighted = weighted * mask
            denom = mask.sum(dim=-2).clamp(min=1)  # (B,L,L,1)
            edge_bias = weighted.sum(dim=-2) / denom  # (B,L,L,H)
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            total_bias = total_bias + edge_bias

        return total_bias


class PairingStructureBias(StructureBiasBase):
    def __init__(self, num_heads: int, init_paired: float = 0.1, init_unpaired: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.paired_bias = nn.Parameter(torch.full((num_heads,), float(init_paired)))
        self.unpaired_bias = nn.Parameter(torch.full((num_heads,), float(init_unpaired)))

    def get_required_inputs(self) -> List[str]:
        return ["pairing_matrix"]
    
    def forward(self, pairing_matrix: torch.Tensor, **_: Any) -> torch.Tensor:
        # pairing_matrix: (B, L, L) 0/1
        bsz, L, _ = pairing_matrix.shape
        paired = pairing_matrix.to(torch.bool)
        # 强制 FP32 避免 AMP 下出 NaN
        bias = torch.where(
            paired.unsqueeze(1),
            self.paired_bias.float().view(1, self.num_heads, 1, 1),
            self.unpaired_bias.float().view(1, self.num_heads, 1, 1),
        )
        return bias
    

class StructureAwareAttentionWrapper(nn.Module):
    """Wrap a self-attention module and inject structure bias into attention scores."""
    
    def __init__(self, original_attention: nn.Module, structure_bias: StructureBiasBase, layer_idx: int = 0):
        super().__init__()
        self.original_attention = original_attention
        self.structure_bias = structure_bias
        self.layer_idx = layer_idx
        self._structure_info: Optional[Dict[str, torch.Tensor]] = None
    
    def set_structure_info(self, **kwargs):
        self._structure_info = kwargs
    
    def clear_structure_info(self):
        self._structure_info = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        attn = self.original_attention
        
        # If model exposes flash attention path, fall back to original (no bias)
        if hasattr(attn, "flash_attn_func") and getattr(attn, "flash_attn_func") is not None:
            return attn(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        # We assume BERT-style self attention interface
        if not (hasattr(attn, "query") and hasattr(attn, "key") and hasattr(attn, "value")):
            # Unknown attention implementation
            return attn(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        
        # Debug input
        if not getattr(self, "_input_debug_printed", False):
            if not torch.isfinite(hidden_states).all():
                print(f"[Attn NaN Debug] Input hidden_states has nan/inf!")
                self._input_debug_printed = True

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
        
        # scale
        query_layer = query_layer * (attn.attention_head_size ** -0.5)

        # rotary / relative position if exists
        if getattr(attn, "position_embedding_type", None) == "rotary" and hasattr(attn, "rotary_embeddings"):
            query_layer, key_layer = attn.rotary_embeddings(query_layer, key_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # === NaN Debug (only print once) ===
        if not getattr(self, "_nan_debug_printed", False):
            if not torch.isfinite(attention_scores).all():
                print(f"[Attn NaN Debug] after matmul: attn_scores has nan/inf")
                print(f"[Attn NaN Debug]   query_layer has nan/inf: {not torch.isfinite(query_layer).all()}")
                print(f"[Attn NaN Debug]   key_layer has nan/inf: {not torch.isfinite(key_layer).all()}")
                self._nan_debug_printed = True

        # Inject structure bias
        if self._structure_info is not None:
            bias = self.structure_bias(**self._structure_info)
            
            # Debug: 检查 bias 本身
            if not getattr(self, "_bias_debug_printed", False):
                if not torch.isfinite(bias).all():
                    print(f"[Attn NaN Debug] structure bias itself has nan/inf!")
                    self._bias_debug_printed = True
                elif bias.abs().max() > 100:
                    print(f"[Attn NaN Debug] structure bias too large: max={bias.abs().max().item()}")
                    self._bias_debug_printed = True
            
            attention_scores = attention_scores + bias.to(attention_scores.dtype)
            
            # Debug bias
            if not getattr(self, "_nan_debug_printed", False):
                if not torch.isfinite(attention_scores).all():
                    print(f"[Attn NaN Debug] after +bias: attn_scores has nan/inf")
                    self._nan_debug_printed = True

        # mask
        if attention_mask is not None:
            # Debug: attention_mask 通常是 0（保留） 或 -10000/-inf（屏蔽）
            if not getattr(self, "_nan_debug_printed", False):
                if not torch.isfinite(attention_mask).all():
                    print(f"[Attn NaN Debug] attention_mask has inf")
            attention_scores = attention_scores + attention_mask
        
            if not getattr(self, "_nan_debug_printed", False):
                try:
                    if not torch.isfinite(attention_scores).all():
                        print(f"[Attn NaN Debug] after +mask: attn_scores has nan/inf")
                        self._nan_debug_printed = True
                except Exception as e:
                    print(f"[Attn NaN Debug] Error checking attention_scores: {e}")
                    self._nan_debug_printed = True

        # stable softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Debug: 如果 softmax 输入全是 -inf（整行被 mask），输出会是 nan
        if not getattr(self, "_nan_debug_printed", False):
            if not torch.isfinite(attention_probs).all():
                print(f"[Attn NaN Debug] after softmax: attention_probs has nan/inf")
                # 检查是否有整行都是 -inf 的情况
                if attention_mask is not None:
                    row_all_masked = (attention_mask <= -1e9).all(dim=-1)  # (B, H, L)
                    if row_all_masked.any():
                        print(f"[Attn NaN Debug] Found {row_all_masked.sum().item()} rows where all positions are masked!")
                self._nan_debug_printed = True
        attention_probs = attn.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if getattr(attn, "is_decoder", False):
            outputs = outputs + (past_key_value,)
        return outputs


class StructureAwareBackbone(nn.Module):
    """A wrapper that patches attention layers in-place and provides set/clear structure_info."""
    
    def __init__(
        self,
        backbone: nn.Module,
        structure_bias_class: Type[StructureBiasBase],
        structure_bias_kwargs: Optional[Dict[str, Any]] = None,
        share_across_layers: bool = False,
        layers_to_patch: Optional[List[int]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.structure_bias_kwargs = structure_bias_kwargs or {}
        self.share_across_layers = share_across_layers
        
        self.attention_wrappers: List[StructureAwareAttentionWrapper] = []
        self._current_structure_info: Optional[Dict[str, torch.Tensor]] = None
    
        encoder_layers = self._get_encoder_layers(backbone)
        if layers_to_patch is None:
            layers_to_patch = list(range(len(encoder_layers)))
        
        # create bias modules
        if share_across_layers:
            shared_bias = structure_bias_class(**self.structure_bias_kwargs)
            bias_list = [shared_bias] * len(layers_to_patch)
            self.structure_bias_modules = nn.ModuleList([shared_bias])
        else:
            bias_list = [structure_bias_class(**self.structure_bias_kwargs) for _ in layers_to_patch]
            self.structure_bias_modules = nn.ModuleList(bias_list)
        
        for patch_i, layer_i in enumerate(layers_to_patch):
            layer = encoder_layers[layer_i]
            self_attn = self._find_self_attention(layer)
            if self_attn is None:
                continue
            wrapper = StructureAwareAttentionWrapper(
                original_attention=self_attn,
                structure_bias=bias_list[patch_i],
                layer_idx=layer_i,
            )
            self.attention_wrappers.append(wrapper)
            
            # replace
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                layer.attention.self = wrapper
            elif hasattr(layer, "self_attn"):
                layer.self_attn = wrapper
            elif hasattr(layer, "attn"):
                layer.attn = wrapper
            elif hasattr(layer, "attention"):
                layer.attention = wrapper
            
    def _get_encoder_layers(self, backbone: nn.Module) -> List[nn.Module]:
        # Try common HuggingFace layouts
        candidates = [
            ("encoder.layer",),
            ("bert.encoder.layer",),
            ("base_model.encoder.layer",),
            ("encoder.layers",),
            ("layers",),
            ("layer",),
        ]
        for path_parts in candidates:
            obj = backbone
            ok = True
            for part in path_parts[0].split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and isinstance(obj, (list, nn.ModuleList)):
                return list(obj)
        raise ValueError("Could not locate encoder layers in backbone")

    def _find_self_attention(self, layer: nn.Module) -> Optional[nn.Module]:
        if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
            return layer.attention.self
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "attn"):
            return layer.attn
        if hasattr(layer, "attention"):
            return layer.attention
        return None
    
    def set_structure_info(self, **kwargs):
        self._current_structure_info = kwargs
        for w in self.attention_wrappers:
            w.set_structure_info(**kwargs)
    
    def clear_structure_info(self):
        self._current_structure_info = None
        for w in self.attention_wrappers:
            w.clear_structure_info()
    
    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)


def patch_backbone_with_structure(
    backbone: nn.Module,
    structure_bias_type: str = "spd",
    num_heads: Optional[int] = None,
    max_distance: int = 32,
    max_path_length: int = 8,
    share_across_layers: bool = False,
    layers_to_patch: Optional[List[int]] = None,
    **kwargs,
) -> StructureAwareBackbone:
    # infer heads
    if num_heads is None:
        if hasattr(backbone, "config"):
            num_heads = getattr(backbone.config, "num_attention_heads", None)
        if num_heads is None:
            raise ValueError("Could not infer num_heads; please pass num_heads")

    t = structure_bias_type.lower()
    if t == "spd":
        bias_cls = SPDStructureBias
        bias_kwargs = {"num_heads": num_heads, "max_distance": max_distance, **kwargs}
    elif t == "graphormer":
        bias_cls = GraphormerStructureBias
        bias_kwargs = {
            "num_heads": num_heads,
            "max_distance": max_distance,
            "max_path_length": max_path_length,
            **kwargs,
        }
    elif t == "pairing":
        bias_cls = PairingStructureBias
        bias_kwargs = {"num_heads": num_heads, **kwargs}
    else:
        raise ValueError(f"Unknown structure_bias_type: {structure_bias_type}")
    
    return StructureAwareBackbone(
        backbone=backbone,
        structure_bias_class=bias_cls,
        structure_bias_kwargs=bias_kwargs,
        share_across_layers=share_across_layers,
        layers_to_patch=layers_to_patch,
    )

