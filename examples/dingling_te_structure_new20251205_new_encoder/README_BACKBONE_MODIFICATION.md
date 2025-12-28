# 直接修改 Backbone 的结构注入方法

## 概述

本实现直接修改 transformer backbone 的 attention 层，在 attention scores 计算后注入结构偏置，而不是添加额外的 attention 层。

### 核心优势

| 特性 | Interleaved 方法 | **Backbone 修改方法** |
|------|------------------|----------------------|
| 每层新增参数 | ~1.05M (Q/K/V/O) | **~800-1000** |
| 16层总参数 | ~17M | **~13K-17K** |
| 结构融合深度 | 层间融合 | **层内融合** |
| 模型层数 | 2N | N |
| 实现复杂度 | 中等 | 低 |

## 架构对比

### 之前：Interleaved 方法
```
Backbone L1 → [Extra Attn L1 + Structure] → Backbone L2 → [Extra Attn L2 + Structure] → ...
                  ↑ 1.05M参数                                  ↑ 1.05M参数
```

### 现在：直接 Backbone 修改
```
[Backbone L1 + Structure Bias] → [Backbone L2 + Structure Bias] → ...
        ↑ ~800参数                        ↑ ~800参数
```

## 文件结构

```
dingling_te_structure_new20251205_new_encoder/
├── structure_aware_backbone.py          # 核心模块
│   ├── SPDStructureBias                 # SPD 空间编码 (792 params/layer)
│   ├── GraphormerStructureBias          # Graphormer 编码 (1,056 params/layer)
│   ├── PairingStructureBias             # 简单配对编码 (48 params/layer)
│   ├── StructureAwareAttentionWrapper   # Attention 层包装器
│   ├── StructureAwareBackbone           # Backbone 包装器
│   └── patch_backbone_with_structure()  # 便捷函数
│
└── triclass_seq_tissue_structure_backbone.py  # 训练脚本
```

## 实现原理

### 1. 注入位置

在原始 attention 计算中：
```python
# 原始代码
attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
attention_scores = attention_scores + attention_mask
attention_probs = softmax(attention_scores)
```

修改后：
```python
# 修改后代码
attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

# 🔑 在这里注入结构偏置
if structure_info is not None:
    structure_bias = self.structure_bias(**structure_info)
    attention_scores = attention_scores + structure_bias

attention_scores = attention_scores + attention_mask
attention_probs = softmax(attention_scores)
```

### 2. 动态包装机制

使用 `StructureAwareAttentionWrapper` 包装原始 attention 模块：

```python
# 自动检测并替换 backbone 中的 attention 层
patched_backbone = patch_backbone_with_structure(
    backbone=model.model,
    structure_bias_type='spd',  # 或 'graphormer', 'pairing'
    num_heads=24,
    max_distance=32
)

# 训练时设置结构信息
patched_backbone.set_structure_info(spd_matrix=spd_matrix)
outputs = patched_backbone(**inputs)
patched_backbone.clear_structure_info()
```

## 参数量详解

### SPD 编码 (每层 792 参数)
```python
spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))
# Shape: (24, 33) = 792
```

### Graphormer 编码 (每层 1,056 参数)
```python
# 空间编码
spatial_bias = nn.Parameter(torch.zeros(num_heads, max_distance + 1))  # 792
# 边类型嵌入
edge_type_embedding = nn.Parameter(torch.zeros(num_edge_types, num_heads))  # 3×24 = 72
# 路径位置权重
position_weights = nn.Parameter(torch.ones(max_path_length, num_heads))  # 8×24 = 192  最多只看路径上的前 8 条边，位置感知的边特征融合，初始化为1
# 总计: 792 + 72 + 192 = 1,056
```

### 配对编码 (每层 48 参数)
```python
paired_bias = nn.Parameter(torch.full((num_heads,), init_paired))  # 24
unpaired_bias = nn.Parameter(torch.full((num_heads,), init_unpaired))  # 24
# 总计: 48
```

## 使用方法

### 命令行训练

```bash
# SPD 编码
python triclass_seq_tissue_structure_backbone.py --encoding_type spd

# Graphormer 编码 (SPD + Edge)
python triclass_seq_tissue_structure_backbone.py --encoding_type graphormer

# 简单配对编码
python triclass_seq_tissue_structure_backbone.py --encoding_type pairing

# 所有层共享偏置参数
python triclass_seq_tissue_structure_backbone.py --encoding_type spd --share_bias
```

### 代码集成

```python
from structure_aware_backbone import (
    patch_backbone_with_structure,
    SPDStructureBias,
    compute_shortest_path_distance,
    dot_bracket_to_edges
)

# 1. 加载预训练模型
model = OmniModelForSequenceClassification(
    "yangheng/OmniGenome-52M",
    tokenizer,
    num_labels=3
)

# 2. 修改 backbone
patched_backbone = patch_backbone_with_structure(
    backbone=model.model,
    structure_bias_type='spd',
    max_distance=32
)

# 3. 计算结构信息
structure = "..((...))"
pair_edges = dot_bracket_to_edges(structure)
spd_matrix = compute_shortest_path_distance(len(structure), pair_edges)

# 4. 设置结构信息并前向传播
patched_backbone.set_structure_info(spd_matrix=spd_matrix.unsqueeze(0))
outputs = model(**inputs)
patched_backbone.clear_structure_info()
```

## 向后兼容性

当不提供结构信息时，模型行为与原始模型完全相同：

```python
# 不设置结构信息，正常工作
outputs = model(**inputs)  # 等价于原始模型

# 设置结构信息，注入结构偏置
patched_backbone.set_structure_info(spd_matrix=spd_matrix)
outputs = model(**inputs)  # 包含结构信息
patched_backbone.clear_structure_info()
```

## 与 Interleaved 方法的选择

| 场景 | 推荐方法 |
|------|---------|
| 参数量敏感 | **Backbone 修改** |
| 需要最深层结构融合 | **Backbone 修改** |
| 不想修改 backbone 代码 | Interleaved |
| 需要与 FlashAttention 兼容 | Interleaved (目前) |
| 需要可视化结构 attention | 两者都可 |

## 注意事项

1. **FlashAttention 兼容性**：当前实现在检测到 FlashAttention 时会回退到原始 attention（无结构偏置）
2. **内存优化**：SPD 矩阵使用 `uint8` 类型存储以节省内存
3. **层选择**：可通过 `layers_to_patch` 参数只修改部分层

## 自测试

```bash
python structure_aware_backbone.py
```

输出示例：
```
============================================================
Structure-Aware Backbone Module Self-Test
============================================================
1. Testing SPD computation...
   SPD bias parameters: 792
2. Testing Graphormer computation...
   Graphormer bias parameters: 1,056
3. Testing Pairing computation...
   Pairing bias parameters: 48

✅ All self-tests passed!
```
