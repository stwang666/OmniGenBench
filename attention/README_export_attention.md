# Attention导出工具使用说明

## 概述

`export_attention.py` 是一个用于导出模型attention信息到 `.atten` 格式文件的工具。它使用 `embedding_mixin.py` 中的批量处理函数，支持灵活配置层选择和头聚合方式。

## 功能特性

- ✅ 支持批量处理序列
- ✅ 可选择输出模型的特定层（单层或多层）
- ✅ 支持多种头聚合方式（mean, max, sum）
- ✅ 支持多种位置聚合方式（row_mean, col_mean, row_max, col_max, diag）
- ✅ 自动获取模型预测结果和概率
- ✅ 支持从数据集或序列列表加载数据
- ✅ 输出标准 `.atten` 格式文件

## 安装要求

```bash
# 确保已安装omnigenbench
pip install omnigenbench
```

## 使用方法

### 命令行使用

#### 基本用法（从数据集导出）

```bash
python export_attention.py \
    --model_path /path/to/your/model \
    --dataset_path /path/to/your/dataset \
    --output_file output.atten \
    --layer_indices -1 \
    --head_aggregation mean \
    --position_aggregation row_mean
```

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model_path` | str | ✅ | - | 模型路径 |
| `--dataset_path` | str | ❌ | None | 数据集路径（与`--sequences`二选一） |
| `--sequences` | str[] | ❌ | None | 序列列表（与`--dataset_path`二选一） |
| `--labels` | int[] | ❌ | None | 标签列表 |
| `--output_file` | str | ❌ | attention_export.atten | 输出文件路径 |
| `--layer_indices` | int[] | ❌ | None | 层索引，如 `-1`（最后一层）或 `0 5 10`（多层） |
| `--head_aggregation` | str | ❌ | mean | 头聚合方式：`mean`, `max`, `sum` |
| `--position_aggregation` | str | ❌ | row_mean | 位置聚合方式：`row_mean`, `col_mean`, `row_max`, `col_max`, `diag` |
| `--batch_size` | int | ❌ | 4 | 批处理大小 |
| `--max_length` | int | ❌ | 512 | 最大序列长度 |
| `--device` | str | ❌ | auto | 设备（cuda/cpu） |

#### 层选择说明

- **单层**: `--layer_indices -1` （最后一层）
- **多层**: `--layer_indices 0 5 10` （第0, 5, 10层，会自动平均）
- **所有层**: 不指定 `--layer_indices` （会平均所有层）

#### 头聚合方式说明

- **mean**: 平均所有头的attention权重（推荐，最常用）
- **max**: 取所有头的最大attention权重
- **sum**: 求和所有头的attention权重

#### 位置聚合方式说明

- **row_mean**: 对每行求平均（该位置关注其他位置的平均权重）⭐ 推荐
- **col_mean**: 对每列求平均（其他位置关注该位置的平均权重）
- **row_max**: 对每行求最大值
- **col_max**: 对每列求最大值
- **diag**: 对角线值（自注意力权重）

### Python代码使用

```python
from export_attention import export_attention_to_file

# 从数据集导出
export_attention_to_file(
    model_path="/path/to/model",
    dataset_path="/path/to/dataset",
    output_file="output.atten",
    layer_indices=-1,  # 最后一层
    head_aggregation="mean",  # 平均所有头
    position_aggregation="row_mean",  # 行平均
    batch_size=4,
    max_length=512,
)

# 从序列列表导出
sequences = ["ATCGATCG", "GGCCTTAACCGG"]
labels = [1, 0]
other_info = [
    {"gene": "Gene1"},
    {"gene": "Gene2"},
]

export_attention_to_file(
    model_path="/path/to/model",
    sequences=sequences,
    labels=labels,
    other_info=other_info,
    output_file="output.atten",
    layer_indices=-1,
    head_aggregation="mean",
    position_aggregation="row_mean",
)
```

## 使用示例

### 示例1: 导出最后一层的attention（推荐用于解释预测）

```bash
python export_attention.py \
    --model_path /path/to/model \
    --dataset_path /path/to/dataset \
    --output_file attention_last_layer.atten \
    --layer_indices -1 \
    --head_aggregation mean \
    --position_aggregation row_mean
```

**适用场景**: 解释模型预测结果，理解模型决策依据

### 示例2: 导出所有层平均的attention

```bash
python export_attention.py \
    --model_path /path/to/model \
    --dataset_path /path/to/dataset \
    --output_file attention_all_layers.atten \
    --head_aggregation mean \
    --position_aggregation row_mean
```

**适用场景**: 整体分析模型的attention模式

### 示例3: 导出多个指定层的attention

```bash
python export_attention.py \
    --model_path /path/to/model \
    --dataset_path /path/to/dataset \
    --output_file attention_multi_layers.atten \
    --layer_indices 0 5 10 \
    --head_aggregation mean \
    --position_aggregation row_mean
```

**适用场景**: 分析不同层的attention差异

### 示例4: 使用最大头聚合

```bash
python export_attention.py \
    --model_path /path/to/model \
    --dataset_path /path/to/dataset \
    --output_file attention_max_head.atten \
    --layer_indices -1 \
    --head_aggregation max \
    --position_aggregation col_mean
```

**适用场景**: 关注最强的attention信号

## 输出文件格式

输出的 `.atten` 文件是TSV格式，包含以下列：

- **Sequence**: DNA/RNA序列
- **Prediction**: 模型预测的类别
- **Probabilities**: 预测概率值（逗号分隔）
- **Actual Label**: 实际标签
- **Attention**: Attention权重值（逗号分隔，对应序列中每个位置）
- **Other**: JSON格式的其他元数据（如基因ID等）

示例：
```
Sequence	Prediction	Probabilities	Actual Label	Attention	Other
ATCGATCG	1	0.1,0.9	1	0.001,0.002,0.003,0.004	{"gene": "Gene1"}
```

## 常见问题

### Q1: 应该选择哪一层？

- **解释预测**: 使用最后一层 (`-1`)
- **整体分析**: 使用所有层平均（不指定`layer_indices`）
- **局部模式**: 使用第一层 (`0`)

### Q2: 头聚合方式如何选择？

- **mean** (推荐): 最常用，提供平均的attention模式
- **max**: 关注最强的attention信号
- **sum**: 较少使用，可能放大attention值

### Q3: 位置聚合方式如何选择？

- **row_mean** (推荐): 表示"该位置关注其他位置的平均权重"，最常用
- **col_mean**: 表示"其他位置关注该位置的平均权重"，用于分析哪些位置被重点关注
- **diag**: 自注意力权重，表示每个位置对自己的关注度

### Q4: 处理速度慢怎么办？

- 减小 `batch_size`
- 减小 `max_length`
- 使用GPU (`--device cuda`)
- 只提取特定层而不是所有层

### Q5: 内存不足怎么办？

- 减小 `batch_size`
- 减小 `max_length`
- 只提取单层而不是所有层

## 技术细节

### Attention提取流程

1. 使用 `batch_extract_attention_scores` 批量提取attention
2. 根据 `layer_indices` 选择特定层
3. 根据 `head_aggregation` 聚合所有头
4. 根据 `position_aggregation` 提取位置级别的attention
5. 进行模型预测获取probabilities
6. 写入TSV文件

### Attention张量维度

- 原始attention: `(layers, heads, seq_len, seq_len)`
- 聚合头后: `(layers, seq_len, seq_len)`
- 选择层后: `(seq_len, seq_len)`
- 位置聚合后: `(seq_len,)`

## 相关文件

- `export_attention.py`: 主程序
- `example_export_attention.py`: 使用示例
- `README_attention_export.md`: Attention导出原理说明
- `embedding_mixin.py`: 底层attention提取函数

## 许可证

与OmniGenBench项目保持一致。

