# 🏗️ 模型架构详解：线性分类器在哪里？

## 答案：线性分类器在微调模型里，不在预训练模型（52M）里！

---

## 📊 完整模型架构图

```
┌─────────────────────────────────────────────────────────┐
│                   完整模型结构                            │
└─────────────────────────────────────────────────────────┘

输入DNA序列: "ATCGATCG..."
      ↓
┌─────────────────────────────────────────────────────────┐
│  📦 预训练模型部分（OmniGenome-52M）                     │
│     - 这部分是从HuggingFace加载的                        │
│     - 已经在TB级数据上训练好了                           │
├─────────────────────────────────────────────────────────┤
│  1. Tokenizer (k-mer tokenization)                     │
│     输入: "ATCGATCG..."                                 │
│     输出: token_ids [101, 234, 567, ...]                │
│                                                          │
│  2. Embedding层                                         │
│     token_ids → embedding vectors                       │
│     [seq_len] → [seq_len, 768]                         │
│                                                          │
│  3. Transformer Encoder (12层)                          │
│     每层包含:                                            │
│     - Multi-Head Self-Attention (8个头)                │
│     - Feed Forward Network                             │
│     - Layer Normalization                              │
│     - Residual Connection                              │
│     输出: [batch, seq_len, 768]                         │
└─────────────────────────────────────────────────────────┘
                    ↓
        last_hidden_state [batch, seq_len, 768]
                    ↓
┌─────────────────────────────────────────────────────────┐
│  🆕 微调模型部分（你添加的新层）                          │
│     - 这部分是你的代码定义的                             │
│     - 需要在你的数据上训练                               │
├─────────────────────────────────────────────────────────┤
│  4. Pooling层 (OmniPooling)                            │
│     [batch, seq_len, 768] → [batch, 768]               │
│     (把整个序列压缩成单个向量)                           │
│                                                          │
│  5. 🔑 Linear分类器 ← 就是这里！                        │
│     self.classifier = nn.Linear(768, 27)               │
│     [batch, 768] → [batch, 27]                         │
│                                                          │
│  6. Reshape                                             │
│     [batch, 27] → [batch, 9, 3]                        │
│                                                          │
│  7. Softmax (推理时)                                     │
│     [batch, 9, 3] → 概率分布                            │
└─────────────────────────────────────────────────────────┘
                    ↓
         预测结果: 9个组织 × 3个类别
```

---

## 🔍 代码分析

### 代码中的关键部分

```python
class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, *args, **kwargs):
        # 第1步：初始化父类（加载预训练模型）
        super().__init__(config_or_model, tokenizer, ...)
        
        # 第2步：创建Pooling层（微调模型的一部分）
        self.pooler = OmniPooling(self.config)
        
        # 第3步：🔑 创建线性分类器（微调模型的一部分！）
        self.classifier = torch.nn.Linear(
            self.config.hidden_size,  # 768 (来自预训练模型)
            self.num_classes * self.num_labels  # 27 (你的任务特定)
        )
        # ☝️ 这一行创建的分类器是全新的，不是预训练的！
        
        # 第4步：定义损失函数
        self.loss_fn = torch.nn.CrossEntropyLoss(...)
```

### 前向传播过程

```python
def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
    # 步骤1: 通过预训练模型（OmniGenome-52M）
    outputs = self.model(  # ← self.model是预训练的52M模型
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs
    )
    # outputs.logits: [batch, seq_len, 768]
    
    # 步骤2: Pooling（微调层）
    pooled = self.pooler(input_ids, outputs.logits)
    # pooled: [batch, 768]
    
    # 步骤3: 🔑 通过线性分类器（微调层）
    logits = self.classifier(pooled)  # ← 这是你自己的分类器
    # logits: [batch, 27]
    
    # 步骤4: Reshape
    logits = logits.view(batch_size, self.num_labels, self.num_classes)
    # logits: [batch, 9, 3]
    
    return {"loss": loss, "logits": logits, ...}
```

---

## 📦 参数分布

### 预训练模型（OmniGenome-52M）

| 组件 | 参数量 | 是否预训练 |
|------|--------|-----------|
| Embedding层 | ~2M | ✅ 是 |
| 12个Transformer层 | ~48M | ✅ 是 |
| Layer Normalization | ~100K | ✅ 是 |
| **总计** | **~52M** | **✅ 全部预训练好** |

### 微调模型（你添加的）

| 组件 | 参数量 | 是否预训练 |
|------|--------|-----------|
| Pooling层 | 0 或很少 | ❌ 否 |
| **Linear分类器** | **768×27+27=20,763** | **❌ 随机初始化** |
| **总计** | **~20K** | **❌ 需要训练** |

### 完整模型

```
总参数量 = 52,000,000 (预训练) + 20,763 (新增) ≈ 52,020,763

其中：
  - 99.96% 的参数来自预训练模型
  - 0.04% 的参数是新增的分类器
```

---

## 🔄 训练过程中发生了什么？

### 方案1: 全量微调（Fine-tuning All）

```python
# 所有参数都可以更新
model = OmniModelForTriClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,
    num_classes=3,
)

# 训练时：
# ✅ 预训练模型的52M参数会被微调（稍微调整）
# ✅ 线性分类器的20K参数从零开始学习
```

**优点**：性能最好（如果数据足够）  
**缺点**：容易过拟合（如果数据少）

### 方案2: 冻结预训练层（Frozen Backbone）

```python
model = OmniModelForTriClassTESequenceClassification(...)

# 冻结预训练模型的参数
for param in model.model.parameters():  # model.model是预训练部分
    param.requires_grad = False

# 训练时：
# ❌ 预训练模型的52M参数保持不变
# ✅ 只训练线性分类器的20K参数
```

**优点**：防止过拟合，训练速度快  
**缺点**：性能可能不如方案1

### 方案3: 部分冻结（你的改进代码使用）

```python
class ImprovedModel:
    def __init__(self, freeze_layers=6):
        # 冻结底层的6层
        for i, layer in enumerate(self.model.encoder.layer):
            if i < freeze_layers:  # 只冻结前6层
                for param in layer.parameters():
                    param.requires_grad = False

# 训练时：
# ❌ 底层6层: 保持不变（捕捉通用特征）
# ✅ 顶层6层: 微调（适应你的任务）
# ✅ 分类器: 从零训练
```

**优点**：平衡性能和过拟合风险  
**缺点**：需要调节冻结层数

---

## 🎨 可视化参数分布

```
完整模型参数分布：

预训练模型 (52M)         微调层 (20K)
├─────────────────┐    ┌─────┐
│█████████████████│    │█    │
│█████████████████│    │█    │
│█████████████████│    │█    │  ← 线性分类器
│█████████████████│    └─────┘
│█████████████████│      
│█████████████████│      只占0.04%
│█████████████████│      但非常重要！
└─────────────────┘
    占99.96%
    已经训练好
```

---

## ❓ 常见问题

### Q1: 为什么不把分类器也包含在预训练模型里？

**A**: 因为预训练模型是通用的，不知道你的具体任务！

```
OmniGenome-52M预训练时的任务：
  - 掩码语言建模（预测被遮盖的token）
  - 输出：每个位置的token概率
  - 不涉及你的9组织×3类别分类

你的任务：
  - 9组织的表达水平分类
  - 输出：9×3=27维
  - 需要任务特定的分类器
```

### Q2: 能否复用预训练模型自带的分类器？

**A**: 不能！预训练模型的输出层是为掩码语言建模设计的：

```
预训练模型的输出层：
  输入: [batch, seq_len, 768]
  输出: [batch, seq_len, vocab_size]  # 每个位置预测token
  
  不适合你的任务！

你的任务需要：
  输入: [batch, 768]  (pooled)
  输出: [batch, 27]   (9组织×3类别)
  
  所以必须自己定义分类器！
```

### Q3: 线性分类器的初始权重是什么？

**A**: 随机初始化！

```python
self.classifier = nn.Linear(768, 27)

# PyTorch默认初始化：
# - 权重W: 从均匀分布U(-√k, √k)采样，k=1/768
# - 偏置b: 从均匀分布U(-√k, √k)采样

# 初始化后的权重是随机的，没有意义
# 需要通过训练学习到有意义的权重
```

### Q4: 为什么不用预训练的分类器再训练？

**A**: 因为维度不匹配！

```
假设预训练模型有分类器：
  classifier_pretrained: [768, vocab_size]
  vocab_size可能是32000（token词表大小）
  
你的任务需要：
  classifier_yours: [768, 27]
  
维度完全不同，无法复用！
```

---

## 💡 总结

### 关键要点

1. **线性分类器在微调模型里**
   - 不是预训练模型的一部分
   - 是你的代码定义的
   - 从随机初始化开始训练

2. **模型分为两部分**
   - **预训练部分**：OmniGenome-52M（52M参数）
   - **微调部分**：Pooling + 线性分类器（20K参数）

3. **训练时的参数更新**
   - 全量微调：所有参数都更新
   - 冻结预训练：只更新分类器
   - 部分冻结：底层冻结，顶层+分类器更新

4. **为什么这样设计？**
   - 预训练模型提供通用的DNA序列表示
   - 线性分类器适配你的特定任务
   - 迁移学习的标准做法

---

## 🔗 相关文件

- `triclass_te.py`: 查看完整的模型定义
- `triclass_te_improved.py`: 改进版（添加Dropout、冻结层等）
- `CONCEPTS_EXPLAINED.md`: 迁移学习详解



