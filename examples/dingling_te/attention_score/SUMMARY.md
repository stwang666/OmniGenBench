# 🎯 Attention分析总结

## ✅ 您的两个问题的答案

### 问题1: 画的是 average self-attention weights in the last encoder layer吗？

**答案：是的！**

```python
layer_idx = -1  # 最后一个encoder layer (layer 15, 共16层)
avg_attention = attention_result['attentions'][layer_idx].mean(dim=0)
# 平均所有24个attention heads
```

- **层级**: 最后一个Encoder Layer (Layer 15)
- **聚合**: 平均所有24个attention heads
- **矩阵形状**: [序列长度, 序列长度]

### 问题2: self attention weight是这里的query对应的key吗？

**答案：完全正确！**

Self-Attention的计算过程：

```
1. Query (Q) = Input × W_Q
2. Key   (K) = Input × W_K
3. Value (V) = Input × W_V

4. Attention_scores = Q × K^T          # [seq_len, seq_len]
5. Attention_weights = softmax(Scores / √d_k)
6. Output = Attention_weights × V
```

**注意力矩阵 [i, j] 的含义：**

- **i (行, Query)**: 位置i作为Query，"我在提问"
- **j (列, Key)**: 位置j作为Key，"我被查询"
- **值**: Query_i 对 Key_j 的注意力权重
- **约束**: 每行的和 = 1.0 (softmax归一化)

---

## 🔬 关键发现

### 1. 权重加载问题（已解决✅）

**问题**: 之前使用的是随机初始化的权重，而非微调权重

**证据**:
```
对比统计:
- 对角线强度: 错误权重 0.001890 → 正确权重 0.042607 (相差22倍!)
- 最大注意力: 错误权重 0.002264 → 正确权重 0.131478 (相差58倍!)
- 相关系数: -0.13 (完全不同，甚至负相关)
```

**原因**: 保存的权重键名有 `model.` 前缀，直接加载时键名不匹配

**解决方案**: 使用 `attention_analysis_correct_weights.py`
- 手动移除 `model.` 前缀
- 正确加载微调权重到模型

### 2. "斜状式"注意力模式是正常的！✅

**即使使用正确的微调权重，注意力图仍然显示强对角线模式**

这是**正常的**，原因：

1. **任务特性**: TE表达分类主要依赖局部序列特征（motifs、短重复序列）
2. **模型策略**: 
   - 通过局部注意力识别motif
   - 通过分类头整合信息做预测
   - 不需要学习长距离依赖

3. **微调效果**:
   - ✅ 分类头优化显著（准确率99%）
   - ⚠️ 底层注意力模式改变较小（预训练已足够好）

### 3. Multi-Head Attention的作用

您的模型有24个attention heads，每个学习不同模式：

| Head | 可能的关注点 | 熵值 | 集中度 |
|------|------------|------|--------|
| Head 2 | 长距离依赖（对角线最弱） | 5.78 | 0.282 |
| Head 23 | 局部特征（熵最低） | 2.51 | 0.907 |
| Head 4 | 特定位置（集中度最高） | 5.34 | 0.990 |

平均注意力 = 所有heads的综合表现

---

## 📊 生成的可视化文件

### 核心文件（推荐查看）

1. **CORRECT_step1_attention_avg.png**
   - 最后一层，平均所有heads
   - ✅ 使用微调权重
   - 显示整体注意力模式

2. **COMPARISON_wrong_vs_correct_weights.png**
   - 对比：错误权重 vs 正确权重
   - 证明权重加载的重要性

3. **CORRECT_step3_weakest_diagonal_head02.png**
   - Head 2: 对角线最弱的head
   - 可能学到了长距离依赖

### 解释性文件

4. **EXPLAIN_attention_matrix.png**
   - 理论解释：attention矩阵的含义
   - 带标注的示例

5. **EXPLAIN_your_model_attention.png**
   - 使用您的模型的实际示例
   - 展示Query-Key关系

---

## 🚀 使用建议

### 1. 正确的注意力分析脚本

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te/attention_score
python attention_analysis_correct_weights.py
```

**关键代码**（正确加载权重）：
```python
# 加载保存的权重
saved_weights = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location='cpu')

# 移除 'model.' 前缀
corrected_state_dict = {}
for key, value in saved_weights.items():
    if key.startswith('model.'):
        corrected_state_dict[key[6:]] = value
    else:
        corrected_state_dict[key] = value

# 加载到模型
model.model.load_state_dict(corrected_state_dict, strict=False)
```

### 2. 进阶分析建议

**a) 查看不同层的注意力**
```python
# 修改layer_idx
for layer_idx in [0, 5, 10, 15]:  # 首层、中间层、最后层
    # ... 可视化
```

**b) 分析预测错误的样本**
```python
# 对比正确vs错误预测的attention差异
correct_samples = [...]
wrong_samples = [...]
# 可能发现：错误样本的attention熵更高或模式异常
```

**c) 关注特定的heads**
```python
# 基于统计选择代表性heads
head_lowest_entropy     # 最集中
head_weakest_diagonal   # 可能的长距离依赖
head_highest_concentration  # 最明确的关注点
```

---

## 📖 技术细节

### Attention矩阵的数学定义

给定输入序列 $X = [x_1, x_2, ..., x_n]$：

1. **线性变换**:
   - $Q = XW_Q$ (Query矩阵)
   - $K = XW_K$ (Key矩阵)
   - $V = XW_V$ (Value矩阵)

2. **计算注意力分数**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

3. **注意力权重矩阵**:
   $$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$
   - $A_{ij}$ = Query位置i 对 Key位置j 的注意力

4. **Multi-Head机制**:
   $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$
   
   其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 您的模型配置

```
模型: OmniGenome-52M (微调版本)
- Encoder Layers: 16
- Attention Heads: 24 (每层)
- Hidden Size: 480
- 序列长度: 512

任务: 3类TE表达分类 (Low/Medium/High) × 9组织
准确率: 99%
```

---

## 🎯 结论

1. ✅ **您的理解完全正确**:
   - 画的是最后一层的平均self-attention权重
   - attention weight就是Query对Key的权重

2. ✅ **权重加载问题已解决**:
   - 之前用的是错误的随机权重
   - 现在正确加载了微调权重

3. ✅ **"斜状式"模式是正常的**:
   - 反映了您的基因组任务特性
   - 模型学习局部特征，而非长距离依赖
   - 这不影响99%的高预测准确率

4. 💡 **微调主要优化了分类头**:
   - 底层注意力模式变化较小
   - 说明预训练的表示已经很好
   - 符合transfer learning的典型行为

---

## 📚 参考资料

- Attention机制: "Attention Is All You Need" (Vaswani et al., 2017)
- BERT架构: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- 基因组Transformer: "The Nucleotide Transformer" (InstaDeep AI, 2023)

---

生成时间: 2025-11-05
模型版本: OmniGenome-52M (Fine-tuned for TE Classification)

