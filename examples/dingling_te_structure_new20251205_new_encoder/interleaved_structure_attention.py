# -*- coding: utf-8 -*-
"""
Interleaved Structure-Aware Attention Module

This module provides reusable components for interleaved structure attention:
- SPDStructureBias: SPD-based attention bias
- GraphormerStructureBias: Graphormer-style bias with edge encoding
- InterleavedStructureAttentionLayer: Single attention layer with structure bias
- InterleavedStructureWrapper: Wrapper for models with interleaved attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Any

# Edge type constants
EDGE_TYPE_NONE = 0
EDGE_TYPE_BACKBONE = 1
EDGE_TYPE_PAIR = 2


def count_trainable_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SPDStructureBias(nn.Module):
    """
    SPD-based structure bias for attention.
    
    Maps shortest-path distances to learnable per-head biases.
    Total parameters: num_heads * (max_distance + 1)
    For 24 heads and max_distance=32: 24 * 33 = 792 parameters
    """
    
    def __init__(self, num_heads: int, max_distance: int = 32, init_scale: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Learnable bias table: (num_heads, max_distance + 1)
        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
        self._init_bias(init_scale)
    
    def _init_bias(self, scale: float):
        """Initialize with distance-decay pattern"""
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
    
    def forward(self, spd_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            spd_matrix: (B, L, L) tensor of shortest-path distances
        Returns:
            bias: (B, num_heads, L, L) attention bias
        """
        bsz, L, _ = spd_matrix.shape
        
        # Clamp distances and convert to long for indexing
        spd = spd_matrix.to(torch.long).clamp_(0, self.max_distance)
        
        # 强制 FP32 避免 AMP 下索引操作出 NaN
        bias_table = self.spatial_bias.float().t()  # (max_distance + 1, num_heads)
        
        # Index into bias table
        bias_flat = bias_table[spd.reshape(-1)]  # (B*L*L, num_heads)
        bias = bias_flat.view(bsz, L, L, self.num_heads).permute(0, 3, 1, 2)
        
        return bias
    

class GraphormerStructureBias(nn.Module):
    """
    Graphormer-style structure bias with SPD + edge-path encoding.
    
    Features:
    - Spatial encoding: learnable bias per SPD distance
    - Edge encoding: learnable embeddings for edge types along shortest path
    """
    
    def __init__(
        self, 
        num_heads: int, 
        max_distance: int = 32,
        max_path_length: int = 8,
        init_scale: float = 0.1,
        use_spatial: bool = True,
        use_edge: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.max_path_length = max_path_length
        self.use_spatial = use_spatial
        self.use_edge = use_edge
        
        # Spatial bias (same as SPDStructureBias)
        if use_spatial:
            self.spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
            self._init_spatial_bias(init_scale)
        
        # Edge type embeddings: (num_edge_types, num_heads)
        if use_edge:
            self.edge_type_embedding = nn.Parameter(torch.zeros(3, num_heads))
            # Position weights for path positions
            self.position_weights = nn.Parameter(torch.ones(max_path_length, num_heads))
            self._init_edge_bias(init_scale)
    
    def _init_spatial_bias(self, scale: float):
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
    
    def _init_edge_bias(self, scale: float):
        with torch.no_grad():
            self.edge_type_embedding[EDGE_TYPE_NONE].fill_(0.0)
            self.edge_type_embedding[EDGE_TYPE_BACKBONE].fill_(0.05 * scale)
            self.edge_type_embedding[EDGE_TYPE_PAIR].fill_(0.10 * scale)
    
    def forward(
        self, 
        spd_matrix: torch.Tensor,
        edge_path_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            spd_matrix: (B, L, L) shortest-path distances
            edge_path_matrix: (B, L, L, max_path_length) edge types along paths
        Returns:
            bias: (B, num_heads, L, L)
        """
        bsz, L, _ = spd_matrix.shape
        device = spd_matrix.device
        dtype = torch.float32
        
        total_bias = torch.zeros((bsz, self.num_heads, L, L), device=device, dtype=dtype)
        
        # Spatial encoding
        if self.use_spatial:
            spd = spd_matrix.to(torch.long).clamp_(0, self.max_distance)
            # 强制 FP32 避免 AMP 下索引操作出 NaN
            bias_table = self.spatial_bias.float().t()
            spatial_flat = bias_table[spd.reshape(-1)]
            spatial = spatial_flat.view(bsz, L, L, self.num_heads).permute(0, 3, 1, 2)
            total_bias = total_bias + spatial
        
        # Edge encoding
        if self.use_edge and edge_path_matrix is not None:
            P = edge_path_matrix.shape[-1]
            P = min(P, self.max_path_length)
            edge_ids = edge_path_matrix[..., :P].to(torch.long)
            
            # 强制 FP32 避免 AMP 下索引操作出 NaN
            edge_emb = self.edge_type_embedding.float()[edge_ids]  # (B, L, L, P, H)
            pos_w = self.position_weights[:P].float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted = edge_emb * pos_w
            
            # Mask and average
            mask = (edge_ids != EDGE_TYPE_NONE).unsqueeze(-1)
            weighted = weighted * mask
            denom = mask.sum(dim=-2).clamp(min=1)
            edge_bias = weighted.sum(dim=-2) / denom
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            total_bias = total_bias + edge_bias
        
        return total_bias
    

class InterleavedStructureAttentionLayer(nn.Module):
    """
    Single attention layer with structure bias injection.
    
    Can optionally:
    - Copy Q/K/V weights from backbone
    - Use shared projections across layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        structure_bias_module: nn.Module,
        dropout: float = 0.1,
        init_from_backbone: Optional[nn.Module] = None,
        shared_projections: Optional[Dict[str, nn.Module]] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.structure_bias = structure_bias_module
        
        # Use shared projections if provided, else create own
        if shared_projections is not None:
            self.q_proj = shared_projections['q_proj']
            self.k_proj = shared_projections['k_proj']
            self.v_proj = shared_projections['v_proj']
            self.out_proj = shared_projections['out_proj']
            self._owns_projections = False
        else:
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
            self._owns_projections = True
        
            # Initialize from backbone if provided
            if init_from_backbone is not None:
                self._init_from_backbone(init_from_backbone)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def _init_from_backbone(self, backbone_layer: nn.Module):
        """Copy Q/K/V weights from backbone attention layer"""
        copied = False
        
        # Try BERT-style
        if hasattr(backbone_layer, 'attention'):
            attn = backbone_layer.attention
            if hasattr(attn, 'self'):
                self_attn = attn.self
                if hasattr(self_attn, 'query'):
                    self.q_proj.load_state_dict(self_attn.query.state_dict())
                    self.k_proj.load_state_dict(self_attn.key.state_dict())
                    self.v_proj.load_state_dict(self_attn.value.state_dict())
                    if hasattr(attn, 'output') and hasattr(attn.output, 'dense'):
                        self.out_proj.load_state_dict(attn.output.dense.state_dict())
                    copied = True
        
        # Try ESM/RoBERTa style
        if not copied and hasattr(backbone_layer, 'self_attn'):
            self_attn = backbone_layer.self_attn
            if hasattr(self_attn, 'q_proj'):
                self.q_proj.load_state_dict(self_attn.q_proj.state_dict())
                self.k_proj.load_state_dict(self_attn.k_proj.state_dict())
                self.v_proj.load_state_dict(self_attn.v_proj.state_dict())
                if hasattr(self_attn, 'out_proj'):
                    self.out_proj.load_state_dict(self_attn.out_proj.state_dict())
                copied = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_bias: Optional[torch.Tensor] = None,
        **structure_kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, H)
            attention_mask: (B, L) or (B, 1, 1, L)
            precomputed_bias: Optional precomputed structure bias (B, num_heads, L, L)
            **structure_kwargs: Arguments for structure_bias module (spd_matrix, etc.)
        """
        residual = hidden_states
        bsz, seq_len, _ = hidden_states.shape
        
        # Q, K, V projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Add structure bias
        if precomputed_bias is not None:
            attn_scores = attn_scores + precomputed_bias.to(attn_scores.dtype)
        elif structure_kwargs:
        structure_bias = self.structure_bias(**structure_kwargs)
            attn_scores = attn_scores + structure_bias.to(attn_scores.dtype)
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, L) -> (B, 1, 1, L)
                mask = attention_mask[:, None, None, :]
                mask = (1.0 - mask.float()) * -10000.0
            else:
                mask = attention_mask
            attn_scores = attn_scores + mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Residual + LayerNorm
        output = self.layer_norm(output + residual)
        
        return output


class InterleavedStructureWrapper(nn.Module):
    """
    Wrapper that adds interleaved structure attention to a backbone model.
    
    This wrapper inserts structure attention layers between backbone layers.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        structure_bias_class: type,
        structure_bias_kwargs: Dict[str, Any],
        hidden_size: int,
        num_heads: int,
        share_bias: bool = False,
        share_qkv: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        
        # Find encoder layers
        self.encoder_layers = self._find_encoder_layers(backbone)
        num_layers = len(self.encoder_layers)
        
        # Create structure bias modules
        if share_bias:
            shared_bias = structure_bias_class(num_heads=num_heads, **structure_bias_kwargs)
            self.structure_biases = nn.ModuleList([shared_bias] * num_layers)
        else:
            self.structure_biases = nn.ModuleList([
                structure_bias_class(num_heads=num_heads, **structure_bias_kwargs)
                for _ in range(num_layers)
            ])
        
        # Create shared projections if requested
        shared_projections = None
        if share_qkv:
            self.shared_q = nn.Linear(hidden_size, hidden_size)
            self.shared_k = nn.Linear(hidden_size, hidden_size)
            self.shared_v = nn.Linear(hidden_size, hidden_size)
            self.shared_out = nn.Linear(hidden_size, hidden_size)
            shared_projections = {
                'q_proj': self.shared_q,
                'k_proj': self.shared_k,
                'v_proj': self.shared_v,
                'out_proj': self.shared_out
            }
        
        # Create structure attention layers
        self.structure_layers = nn.ModuleList([
            InterleavedStructureAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                structure_bias_module=self.structure_biases[i],
                init_from_backbone=self.encoder_layers[i] if not share_qkv else None,
                shared_projections=shared_projections
            )
            for i in range(num_layers)
        ])
    
    def _find_encoder_layers(self, backbone: nn.Module) -> List[nn.Module]:
        """Find transformer encoder layers in backbone"""
        for attr_path in ['encoder.layer', 'layers', 'transformer.layer', 'transformer.layers']:
            obj = backbone
            for attr in attr_path.split('.'):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    break
            else:
                if hasattr(obj, '__len__') and len(obj) > 0:
                    return list(obj)
        
        # Fallback: search for ModuleList
        for name, module in backbone.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    first = module[0]
                    if hasattr(first, 'attention') or hasattr(first, 'self_attn'):
                    return list(module)
        
        raise ValueError("Could not find encoder layers in backbone")


def create_interleaved_structure_bias(
    encoding_type: str,
    num_heads: int,
    max_distance: int = 32,
    max_path_length: int = 8,
    init_scale: float = 0.1
) -> nn.Module:
    """
    Factory function to create structure bias module.
    
    Args:
        encoding_type: 'spd' or 'graphormer'
        num_heads: Number of attention heads
        max_distance: Maximum SPD distance
        max_path_length: Maximum path length for graphormer encoding
        init_scale: Initial scale for bias values
    """
    if encoding_type == 'spd':
        return SPDStructureBias(
            num_heads=num_heads,
            max_distance=max_distance,
            init_scale=init_scale
        )
    elif encoding_type == 'graphormer':
        return GraphormerStructureBias(
            num_heads=num_heads,
            max_distance=max_distance,
            max_path_length=max_path_length,
            init_scale=init_scale
        )
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
