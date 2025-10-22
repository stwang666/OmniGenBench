# 🎴 快速参考卡片

## 1. 迁移学习？✅ 是的！

```
预训练阶段                  微调阶段
(OmniGenome)    →    (你的TE表达预测)
│                          │
├─ 数据: TB级DNA序列        ├─ 数据: 你的标注数据
├─ 任务: 预测遮盖碱基       ├─ 任务: 9组织×3类别
└─ 学到: 通用DNA表示        └─ 学到: 特定任务知识
```

**为什么有效？**
- 📚 预训练模型已懂DNA"语法"
- 🚀 训练更快（数小时 vs 数天）
- 💯 性能更好（即使数据少）

---

## 2. 线性分类器：hidden_size → 27

### 公式
```
output = input @ W^T + b
[batch, 27] = [batch, 768] @ [768, 27]^T + [27]
```

### 原理
```
768维特征 → 27维输出
每一维 = 768个特征的加权和
```

### 为什么27维？
```
9个组织 × 3个类别 = 27
[27] → reshape → [9, 3]
对每个组织: softmax([Low, Med, High])
```

### Linear vs MLP

| 特性 | Linear | MLP |
|------|--------|-----|
| 层数 | 1 | 2+ |
| 非线性 | ❌ | ✅ ReLU |
| 参数量 | 20K | 300K+ |
| 决策边界 | 直线 | 曲线 |
| 表达能力 | 弱 | 强 |

---

## 3. 注意力机制：让模型"看重点"

### 核心思想
```
"The cat sat on the mat because it was tired"
                                 ↑       ↑
                                 关注  "cat"
```

### 数学（4步）
```
1. Q, K, V = X @ W_q, X @ W_k, X @ W_v
2. scores = Q @ K^T / sqrt(d_k)
3. weights = softmax(scores)  # 归一化
4. output = weights @ V       # 加权和
```

### 在DNA中的作用
```
序列: A T C G A T C G
      ↑       ↑
   位置1关注位置5
   
可能原因:
  - 功能相关（启动子-增强子）
  - 配对碱基（A-T）
  - Motif组成部分
```

### Multi-Head（多头）
```
8个头 → 8种模式
  头1: 局部motif
  头2: 远距离互作
  头3: GC含量
  头4: 重复序列
  ...
```

### 可视化
```python
outputs = model(..., output_attentions=True)
attention = outputs.attentions[-1].mean(1)  # 平均所有头
sns.heatmap(attention[0].cpu().numpy())
```

---

## 4. k-mer：DNA的"单词"

### 定义
```
k-mer = 长度为k的连续子序列

序列: ATCGATCG
1-mer: A, T, C, G, A, T, C, G
2-mer: AT, TC, CG, GA, AT, TC, CG
3-mer: ATC, TCG, CGA, GAT, ATC, TCG
```

### 为什么用k-mer？

**问题：单碱基tokenization**
```
"ATGCAT" → ['A','T','G','C','A','T']
❌ "ATG"起始密码子被拆分
❌ 序列太长
```

**解决：k-mer tokenization**
```
"ATGCAT" → ['ATG', 'TGC', 'GCA', 'CAT']
✅ 保留motif完整性
✅ 序列压缩3倍
✅ 训练更快
```

### k值选择

| k | 可能数 | 用途 |
|---|--------|------|
| 1 | 4 | 碱基组成 |
| 3 | 64 | 密码子分析 |
| 4-6 | 256-4K | Motif识别 |
| 21+ | 巨大 | 基因组组装 |

### OmniGenome的策略
```
不是固定k，而是BPE（类似GPT）

自动学习常见组合:
  ['A','T'] → 'AT'（频繁）
  ['AT','G'] → 'ATG'（起始密码子）
  ['TA','TA'] → 'TATA'（TATA box）
  
最终词表:
  单碱基 + 常见k-mer + 功能motif
```

---

## 🔗 概念关联图

```
DNA序列: ATGCATGC...
    ↓
[k-mer tokenization]
    ↓
Tokens: [ATG, TGC, CAT, ...]
    ↓
[Embedding + 位置编码]
    ↓
[Transformer层 × 12]
  ├─ Multi-Head Attention ← 学习序列依赖
  └─ Feed Forward
    ↓
[Pooling]
    ↓
Hidden: [batch, 768]
    ↓
[Linear/MLP分类器]
    ↓
Output: [batch, 27]
    ↓
[Reshape]
    ↓
Logits: [batch, 9, 3]
    ↓
[Softmax]
    ↓
预测: 9组织 × (Low/Med/High)
```

---

## 🎯 一句话总结

1. **迁移学习**：站在巨人肩膀上（用预训练模型）
2. **线性分类器**：把768维特征压缩成27维输出
3. **注意力机制**：让模型知道哪里重要
4. **k-mer**：把DNA序列变成"单词"

---

## 🚀 运行示例

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te

# 三个核心概念的详细示例
python linear_vs_mlp_explanation.py  # 线性 vs MLP
python attention_explanation.py       # 注意力机制
python kmer_explanation.py           # k-mer分析

# 查看完整文档
cat CONCEPTS_EXPLAINED.md
```

---

## 📊 实际数值

**你的模型**：
```
输入: DNA序列（可变长度）
  ↓ tokenization
Tokens: ~300-500个token
  ↓ OmniGenome-52M (768维)
Hidden: [batch, seq_len, 768]
  ↓ pooling
Pooled: [batch, 768]
  ↓ Linear(768→27)
Output: [batch, 27]
  ↓ reshape
Final: [batch, 9, 3]
```

**参数量**：
- 预训练模型：52,000,000
- 分类器：20,763 (Linear) 或 300,000+ (MLP)
- 总计：~52,020,763 或 ~52,300,000

**训练数据**：
- 预训练：TB级基因组数据
- 微调：你的数据（数百-数千样本）

---

## 💡 记忆技巧

```
迁移学习 = 借用知识
  (就像会英语后学法语更快)

k-mer = DNA的单词
  (就像句子由单词组成)

Attention = 划重点
  (就像阅读时关注重要信息)

Linear = 线性变换
  (就像y = ax + b)
```



