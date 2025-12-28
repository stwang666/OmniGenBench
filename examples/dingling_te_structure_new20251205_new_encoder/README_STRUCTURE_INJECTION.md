# RNA 二级结构注入 Attention 的实现说明

## 核心问题

如何将 RNA 二级结构信息（点括号表示法）注入到 Transformer 的 Self-Attention 中？

## 实现方案：多类型可学习偏置 (Graphormer-style)

### 1. 数学原理

修改 Self-Attention 的计算公式：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B_{\text{struct}}\right)V
$$

其中 $B_{\text{struct}}$ 是结构偏置矩阵：

$$
B_{\text{struct}}[i,j] = \begin{cases}
b_{\text{paired}} & \text{如果 } M_{ij} = 1 \text{ (i和j配对)} \\
b_{\text{unpaired}} & \text{如果 } M_{ij} = 0 \text{ 且 } i \neq j \\
b_{\text{self}} & \text{如果 } i = j
\end{cases}
$$

其中 $b_{\text{paired}}, b_{\text{unpaired}}, b_{\text{self}}$ 都是**可学习参数**。

### 2. 关键设计决策

| 问题 | 决策 | 原因 |
|------|------|------|
| 固定值 vs 可学习 | **可学习** | 让模型自动确定最优偏置值 |
| 单一偏置 vs 多类型 | **多类型** | 区分配对/非配对/自身 |
| 共享 vs 独立 | **每个 head 独立** | 不同 head 学习不同的结构偏好 |
| 非配对处理 | **可学习偏置（非零）** | 模型可以学习抑制或增强非配对位置 |

### 3. 代码结构

```
triclass_seq_tissue_structure.py
│
├── dot_bracket_to_pairing_matrix()     # 解析点括号为配对矩阵
│
├── StructureAttentionBias              # 可学习偏置模块
│   ├── paired_bias                     # 配对位置偏置 (num_heads,)
│   ├── unpaired_bias                   # 非配对位置偏置 (num_heads,)
│   └── self_bias                       # 对角线偏置 (num_heads,)
│
├── OmniDatasetWithTissueAndStructure   # 数据集类
│   └── prepare_input()                 # 返回 pairing_matrix
│
└── OmniModelWithTissueAndStructure     # 模型类
    ├── structure_bias                  # StructureAttentionBias 实例
    ├── tissue_embedding                # Tissue 嵌入层
    └── forward()                       # 注入结构偏置
```

### 4. 是否需要修改模型内部？

**不需要修改 Self-Attention 代码！**

有两种注入方式：

#### 方式 A：通过扩展 attention_mask（推荐）

大多数 HuggingFace 模型的 attention_mask 是**加性的**：

```python
# HuggingFace BERT 内部实现
attention_scores = attention_scores + attention_mask
```

所以我们可以构造一个 4D 的 attention_mask：

```python
# 原始 padding mask: (batch, seq_len)
# 扩展为: (batch, num_heads, seq_len, seq_len)
extended_mask = padding_mask + structure_bias
```

#### 方式 B：通过 register_forward_hook

```python
def attention_hook(module, args, kwargs):
    if 'attention_mask' in kwargs:
        kwargs['attention_mask'] = kwargs['attention_mask'] + structure_bias
    return args, kwargs

for module in model.modules():
    if 'attention' in module.__class__.__name__.lower():
        module.register_forward_pre_hook(attention_hook, with_kwargs=True)
```

### 5. 参数初始化

```python
# 推荐的初始化
paired_bias = 0.1      # 小正值，轻微鼓励关注配对位置
unpaired_bias = 0.0    # 中性，让模型学习
self_bias = 0.0        # 中性
```

### 6. 与 Graphormer 的对比

| 方面 | Graphormer | 我们的实现 |
|------|------------|-----------|
| 输入 | 图的最短路径距离 | RNA 配对矩阵 |
| 编码类型 | 距离离散值 → 嵌入 | 二元值 → 可学习偏置 |
| 每头独立 | ✓ | ✓ |
| 可学习 | ✓ | ✓ |

### 7. 训练后可以观察到

```python
# 训练后检查学习到的偏置
print(f"paired_bias: {model.structure_bias.paired_bias.data.mean():.4f}")
print(f"unpaired_bias: {model.structure_bias.unpaired_bias.data.mean():.4f}")
print(f"self_bias: {model.structure_bias.self_bias.data.mean():.4f}")
```

如果 `paired_bias > unpaired_bias`，说明模型学会了更多关注配对位置。

### 8. 运行示例

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder
python triclass_seq_tissue_structure.py
```

## 参考文献

1. Graphormer: Do Transformers Really Perform Bad for Graph Representation? (NeurIPS 2021)
2. ERNIE-RNA: An RNA Language Model with Structure-sequence Pre-training (Nature Communications 2024)
3. trRosettaRNA: Automated prediction of RNA 3D structure (Nature Communications 2023)
