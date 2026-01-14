# Attention信息文件导出说明

## 文件格式

`.atten` 文件是TSV（Tab-Separated Values）格式，包含以下列：

- **Sequence**: DNA/RNA序列
- **Prediction**: 模型预测的类别（0或1）
- **Probabilities**: 预测概率值（逗号分隔，格式：`prob_class0,prob_class1`）
- **Actual Label**: 实际标签
- **Attention**: Attention权重值（逗号分隔的数值列表，对应序列中每个位置的attention权重）
- **Other**: JSON格式的其他元数据（如基因ID等）

## 导出方法

### 方法1：使用OmniGenBench的Attention提取功能

基于代码库中的 `extract_attention_scores` 方法，可以按以下步骤导出：

```python
from omnigenbench import ModelHub, OmniTokenizer
import json
import csv

# 1. 加载模型和tokenizer
model_path = "your_model_path"
model = ModelHub.load(model_path)
tokenizer = OmniTokenizer.from_pretrained(model_path)

# 2. 准备数据
sequences = ["CTGTCAATCGCCGAATGGCACCCTGCCGCACGGCAGAGGATCCGCGTCCACAAAACCCAACCCCCACGCACCGCGCGCAGCCGTTTTACCCAGCATGGCAAGAGGCGCATCCCGAGCATCACTCGTCGACTGACGAGGACCTGAGGCGGAGGCCCTAGAGGGAGCAAAGCAAAAGGTGCTGGTGAGTGGTGACTATAGCTACCATGACAAGTTGAGAGAGAGAGGGAGAGACCCACACTAGGCTGTCGCACCAACGGGTCAGGAGGGAGGGAGAGAGAGAGAGAGACGGGGGTTTTAATTTGAAGGGCGAAGCCTGGTCACTCGTGTGTGTGCGTGTGTTTCCCCCCGTCGTCGCCACTCTCCCTCTTTCCCCTCTCTCCCTCGAGAGAGGAGAGGAGAGGGGGGCCAGCAAGCAGGCAAGAGCAGTGTTCCACCTCCACTTCCACGACCAGCATCGCCAGCGCACCACGAGCTAGCTAGGGCCGACGACAGGCGCCGCA"]
labels = [1]  # 实际标签
other_info = [{"gene": "TraesCS7A03G0567100.1"}]  # 其他元数据

# 3. 提取attention和进行预测
results = []
for seq, label, info in zip(sequences, labels, other_info):
    # 提取attention
    attention_result = model.extract_attention_scores(
        sequence=seq,
        max_length=512,
        return_on_cpu=True
    )
    
    # 进行预测
    outputs = model.inference(seq)
    prediction = outputs['predictions']
    probabilities = outputs['confidence']  # 或使用logits计算
    
    # 获取attention权重（平均所有层和头）
    attentions = attention_result['attentions']  # [layers, heads, seq_len, seq_len]
    # 平均所有层和头，然后对每个位置求平均（或使用其他聚合方式）
    mean_attention = attentions.mean(dim=(0, 1)).mean(dim=0).numpy().tolist()
    
    # 格式化概率
    if isinstance(probabilities, (list, tuple)):
        prob_str = ",".join(map(str, probabilities))
    else:
        prob_str = f"{1-probabilities},{probabilities}"  # 二分类情况
    
    # 格式化attention
    attention_str = ",".join(map(str, mean_attention))
    
    # 格式化other信息
    other_str = json.dumps(info)
    
    results.append({
        'Sequence': seq,
        'Prediction': prediction,
        'Probabilities': prob_str,
        'Actual Label': label,
        'Attention': attention_str,
        'Other': other_str
    })

# 4. 写入TSV文件
output_file = "output.atten"
with open(output_file, 'w', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Sequence', 'Prediction', 'Probabilities', 
                                          'Actual Label', 'Attention', 'Other'], 
                           delimiter='\t')
    writer.writeheader()
    writer.writerows(results)
```

### 方法2：使用模型的前向传播直接提取

```python
import torch
import json
import csv
from omnigenbench import ModelHub, OmniTokenizer

model_path = "your_model_path"
model = ModelHub.load(model_path)
tokenizer = OmniTokenizer.from_pretrained(model_path)
model.model.config.output_attentions = True  # 启用attention输出

sequences = [...]  # 你的序列列表
labels = [...]    # 实际标签列表

results = []
for seq, label in zip(sequences, labels):
    # Tokenize
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        outputs = model.model(**inputs, output_attentions=True)
    
    # 获取预测结果
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    prediction = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    
    # 获取attention权重
    attentions = outputs.attentions  # tuple of tensors
    # 堆叠所有层: [layers, batch, heads, seq_len, seq_len]
    attn_stack = torch.stack(attentions, dim=0)
    # 移除batch维度并平均所有层和头
    attn_mean = attn_stack[0].mean(dim=(0, 1)).mean(dim=0).cpu().numpy()
    
    # 格式化输出
    prob_str = ",".join(map(str, probs))
    attention_str = ",".join(map(str, attn_mean))
    
    results.append({
        'Sequence': seq,
        'Prediction': prediction,
        'Probabilities': prob_str,
        'Actual Label': label,
        'Attention': attention_str,
        'Other': json.dumps({})  # 添加其他元数据
    })

# 写入文件
with open("output.atten", 'w', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Sequence', 'Prediction', 'Probabilities', 
                                          'Actual Label', 'Attention', 'Other'], 
                           delimiter='\t')
    writer.writeheader()
    writer.writerows(results)
```

## 关键步骤说明

1. **启用Attention输出**: 确保模型配置中 `output_attentions=True`
2. **提取Attention权重**: 从模型输出中获取attention张量
3. **聚合Attention**: 通常需要：
   - 平均所有transformer层
   - 平均所有attention头
   - 对每个序列位置计算聚合值（如行平均或列平均）
4. **格式化输出**: 将数值列表转换为逗号分隔的字符串
5. **写入TSV**: 使用tab分隔符写入文件

## 注意事项

- Attention权重的维度通常是 `[layers, heads, seq_len, seq_len]`
- 需要根据具体需求选择聚合方式（平均、最大、求和等）
- 序列长度可能因padding而不同，需要处理attention mask
- 对于分类任务，probabilities通常是softmax后的概率分布

## 相关代码文件

- `omnigenbench/src/abc/embedding_mixin.py`: 包含 `extract_attention_scores` 方法
- `attention/attention_seq.py`: Attention可视化示例
- `attention/attention_pairings.py`: Attention配对分析示例
