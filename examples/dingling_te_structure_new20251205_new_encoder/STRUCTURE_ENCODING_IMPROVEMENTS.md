# 结构编码改进说明

## 问题1：SPD Bias 的影响方向

### 问题描述
在当前的SPD实现中，距离为1的位置相比距离为3的位置，在attention里面数字较大的反而告诉模型的影响更大一些吗？但实际上距离越近更可能形成motif。

### 当前实现分析

当前代码中的初始化策略（`_init_bias`）：
```python
- 距离 0 (自身): 0.5
- 距离 1 (直接相邻): 0.3
- 距离 2: 0.2
- 距离 3+: 逐渐衰减到 0
```

**关键点：**
1. ✅ **距离越近，bias越大**：这是正确的，符合生物学直觉
2. ⚠️ **Softmax的放大效应**：bias在softmax之前加入，softmax的指数性质会放大这种差异

### Softmax的影响

假设两个位置的attention score：
- 位置A（距离1）：`score_A = base_score + 0.3`
- 位置B（距离3）：`score_B = base_score + 0.09`

在softmax之后：
```
prob_A = exp(score_A) / (exp(score_A) + exp(score_B) + ...)
prob_B = exp(score_B) / (exp(score_A) + exp(score_B) + ...)
```

由于指数函数，即使bias差异只有0.21（0.3 - 0.09），softmax后的概率差异会被显著放大。

### 改进方案

在 `structure_encoding_improvements.py` 中的 `ImprovedSPDStructureBias` 类：

1. **更合理的初始化**：
   - 距离0：1.0 * scale（自身）
   - 距离1：0.5 * scale（直接相邻，容易形成motif）
   - 距离2：0.3 * scale
   - 距离3+：指数衰减

2. **温度缩放**：
   ```python
   spatial_bias = self.spatial_bias / self.temperature
   ```
   通过调整temperature可以控制softmax的锐度，避免过度放大。

3. **衰减函数选择**：
   - `exponential`: `exp(-alpha * (d - 2))`
   - `inverse`: `1 / (1 + beta * (d - 2))`

### 验证

当前实现是**正确的**：距离越近，bias越大，这符合生物学直觉。softmax会进一步放大这种差异，使得模型更关注距离近的位置（更容易形成motif）。

---

## 问题2：配对类型编码

### 问题描述
在传入结构信息时，有没有告诉模型对于配对位置的不同情况，比如GC含有三个氢键更稳定、折叠的能量也更高；AU两个氢键这种信息。稳定性不同对TE的影响也不同。

### 当前实现分析

当前代码中：
- `dot_bracket_to_pairing_matrix()` 只从点括号表示法提取配对/非配对信息
- 点括号表示法（如 `"(((...)))"`）**不包含碱基类型信息**
- 因此无法区分GC、AU、GU等不同类型的配对

### 生物学背景

RNA碱基对的稳定性（基于氢键数量和自由能）：

| 配对类型 | 氢键数 | 自由能 (ΔG, kcal/mol) | 稳定性 |
|---------|--------|----------------------|--------|
| GC / CG | 3      | ~-3.4                | 最稳定 |
| AU / UA | 2      | ~-2.1                | 中等稳定 |
| GU / UG | 2      | ~-1.3                | 较不稳定（wobble配对）|

**对翻译效率（TE）的影响：**
- 更稳定的配对（GC）→ 更稳定的二级结构 → 可能影响核糖体结合 → 影响TE
- 不稳定的配对（GU）→ 结构更灵活 → 可能有利于某些翻译过程

### 改进方案

在 `structure_encoding_improvements.py` 中提供了完整的配对类型编码：

#### 1. 配对类型提取

```python
def dot_bracket_to_pairing_matrix_with_types(
    structure: str,
    sequence: str,  # 需要序列信息！
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回：
    - pairing_matrix: (L, L) 0/1矩阵
    - pairing_type_matrix: (L, L) 配对类型ID (0=GC, 1=AU, 2=GU, 3=OTHER)
    - pairing_stability_matrix: (L, L) 稳定性权重
    """
```

#### 2. 配对类型Bias

```python
class BasePairTypeBias(nn.Module):
    """
    为不同配对类型分配可学习的bias：
    - GC (type 0): 最稳定，bias = 0.5 * scale
    - AU (type 1): 中等稳定，bias = 0.3 * scale
    - GU (type 2): 较不稳定，bias = 0.1 * scale
    - OTHER (type 3): 非配对，bias = 0.0
    """
```

#### 3. 组合编码

```python
class CombinedStructureBias(nn.Module):
    """
    组合SPD和配对类型编码：
    
    A_ij = (Q_i K_j^T) / √d + b_spatial(SPD_ij) + b_pair_type(pair_type_ij)
    
    优势：
    1. SPD捕获空间距离（距离越近，bias越大）
    2. 配对类型捕获稳定性（GC > AU > GU）
    """
```

### 使用示例

```python
# 1. 从点括号和序列提取配对类型
pairing_matrix, pairing_type_matrix, pairing_stability_matrix = \
    dot_bracket_to_pairing_matrix_with_types(
        structure="(((...)))",
        sequence="AUGCAUGC",  # 需要序列！
        max_length=512
    )

# 2. 在模型中应用
combined_bias = CombinedStructureBias(
    num_heads=12,
    max_distance=32,
    use_pair_type=True,  # 启用配对类型编码
)

# 3. 计算attention bias
total_bias = combined_bias(
    spd_matrix,
    pairing_matrix,
    pairing_type_matrix,
    pairing_stability_matrix,
)
```

### 集成到现有代码

要集成到现有代码中，需要：

1. **修改数据集类**：
   - 在 `prepare_input()` 中同时读取 `sequence` 和 `structure`
   - 使用 `dot_bracket_to_pairing_matrix_with_types()` 计算配对类型矩阵

2. **修改模型类**：
   - 将 `SPDStructureBias` 替换为 `CombinedStructureBias`
   - 在 `forward()` 中传入配对类型矩阵

3. **数据要求**：
   - CSV文件需要同时包含 `sequence` 和 `structure` 列
   - 确保序列和结构的长度一致

---

## 总结

### 问题1：SPD Bias ✅
- **当前实现是正确的**：距离越近，bias越大
- **改进建议**：使用 `ImprovedSPDStructureBias` 获得更好的初始化和温度控制

### 问题2：配对类型编码 ❌
- **当前实现不包含配对类型信息**
- **改进方案**：使用 `BasePairTypeBias` 和 `CombinedStructureBias` 来编码GC、AU、GU等配对类型的稳定性差异

### 下一步

1. 在数据集中添加序列信息提取
2. 使用 `structure_encoding_improvements.py` 中的改进类
3. 验证配对类型编码对模型性能的影响



