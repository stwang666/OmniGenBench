# 数据合并说明文档

## 📊 数据合并结果

已成功将9个tissue的CSV文件合并为一个统一的数据文件。

### 输入文件

```
preprocess_data/
├── root_TE.csv (59,636 个样本)
├── seedling_TE.csv (62,176 个样本)
├── leaf_TE.csv (44,917 个样本)
├── FMI_TE.csv (63,492 个样本)
├── FOD_TE.csv (64,243 个样本)
├── Prophase-I-pollen_TE.csv (53,095 个样本)
├── Tricellular-pollen_TE.csv (65,352 个样本)
├── flag_TE.csv (45,755 个样本)
└── grain_TE.csv (49,769 个样本)
```

### 输出文件

```
preprocess_data/
├── merged_tissue_data.csv (92,715 个样本, 47.2 MB)
└── merge_report.txt (详细统计报告)
```

---

## 📝 输出数据格式

### CSV文件结构

```csv
ID,sequence,root_TE_label,seedling_TE_label,leaf_TE_label,FMI_TE_label,FOD_TE_label,Prophase-I-pollen_TE_label,Tricellular-pollen_TE_label,flag_TE_label,grain_TE_label
```

### 列说明

| 列名 | 说明 | 示例 |
|------|------|------|
| `ID` | 样本唯一标识符 | `TraesCS1A03G0000100LC` |
| `sequence` | DNA序列 | `ATCGATCG...` |
| `root_TE_label` | root tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `seedling_TE_label` | seedling tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `leaf_TE_label` | leaf tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `FMI_TE_label` | FMI tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `FOD_TE_label` | FOD tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `Prophase-I-pollen_TE_label` | Prophase-I-pollen tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `Tricellular-pollen_TE_label` | Tricellular-pollen tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `flag_TE_label` | flag tissue的TE表达标签 | `0`, `1`, 或空（NA）|
| `grain_TE_label` | grain tissue的TE表达标签 | `0`, `1`, 或空（NA）|

### 标签含义

- `0` = 低表达（Low）
- `1` = 高表达（High）
- 空值（NA）= 缺失数据

---

## 📊 数据统计

### 总体统计

- **总样本数**: 92,715 个
- **文件大小**: 47.2 MB
- **成功合并的tissue**: 全部9个

### 各tissue标签分布

| Tissue | 有效标签 | 标签=0 | 标签=1 | 标签=NA |
|--------|---------|--------|--------|---------|
| root | 13,093 (14.1%) | 6,487 (7.0%) | 6,606 (7.1%) | 79,622 (85.9%) |
| seedling | 12,604 (13.6%) | 6,058 (6.5%) | 6,546 (7.1%) | 80,111 (86.4%) |
| leaf | 9,433 (10.2%) | 5,056 (5.5%) | 4,377 (4.7%) | 83,282 (89.8%) |
| FMI | 14,753 (15.9%) | 8,390 (9.0%) | 6,363 (6.9%) | 77,962 (84.1%) |
| FOD | 15,699 (16.9%) | 8,928 (9.6%) | 6,771 (7.3%) | 77,016 (83.1%) |
| Prophase-I-pollen | 9,785 (10.6%) | 5,079 (5.5%) | 4,706 (5.1%) | 82,930 (89.4%) |
| Tricellular-pollen | 13,731 (14.8%) | 7,333 (7.9%) | 6,398 (6.9%) | 78,984 (85.2%) |
| flag | 9,888 (10.7%) | 5,292 (5.7%) | 4,596 (5.0%) | 82,827 (89.3%) |
| grain | 12,954 (14.0%) | 6,061 (6.5%) | 6,893 (7.4%) | 79,761 (86.0%) |

### 样本的tissue标签数量分布

| Tissue数量 | 样本数 | 百分比 |
|-----------|--------|--------|
| 0个tissue | 32,396 | 34.9% |
| 1个tissue | 29,769 | 32.1% |
| 2个tissue | 16,644 | 18.0% |
| 3个tissue | 8,721 | 9.4% |
| 4个tissue | 3,633 | 3.9% |
| 5个tissue | 1,181 | 1.3% |
| 6个tissue | 315 | 0.3% |
| 7个tissue | 55 | 0.1% |
| 8个tissue | 1 | 0.0% |
| 9个tissue | 0 | 0.0% |

**注意**：
- 34.9%的样本在所有tissue中都没有标签（全部为NA）
- 只有约0.1%的样本有7个或更多tissue的标签
- 没有样本在所有9个tissue中都有标签

---

## 🔍 数据质量验证

### ✅ 已完成的检查

1. **序列一致性检查**: 所有ID的序列一致，没有发现ID相同但序列不同的情况
2. **列名正确性**: 所有label列按照指定顺序正确命名
3. **数据完整性**: 所有9个tissue的数据都成功合并

### 📌 注意事项

1. **缺失数据较多**: 每个tissue约有80-90%的样本标签为NA
2. **标签不均衡**: 不同tissue的标签分布有所差异
3. **多标签学习**: 由于大量NA，适合使用多标签学习方法，且需要在损失函数中忽略NA标签

---

## 🚀 如何使用这些数据

### 方法1: 直接使用合并后的数据训练

合并后的数据格式已经可以直接用于训练多标签分类模型。

```bash
# 如果你已有训练脚本（如train_binary_te.py）
# 只需确保脚本能读取这个格式的数据
python train_binary_te.py --data_file preprocess_data/merged_tissue_data.csv
```

### 方法2: 进一步划分train/valid/test

你可能需要将合并后的数据划分为训练集、验证集和测试集：

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('preprocess_data/merged_tissue_data.csv')

# 划分数据集
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)
```


### 方法3: 过滤数据

你可能想要过滤掉没有任何标签的样本：

```python
import pandas as pd

df = pd.read_csv('preprocess_data/merged_tissue_data.csv')

# 获取所有label列
label_cols = [col for col in df.columns if col.endswith('_TE_label')]

# 只保留至少有一个tissue有标签的样本
df_filtered = df[df[label_cols].notna().any(axis=1)]

print(f"原始样本数: {len(df)}")
print(f"过滤后样本数: {len(df_filtered)}")

df_filtered.to_csv('merged_tissue_data_filtered.csv', index=False)
```
原始样本数: 92715
过滤后样本数: 60319
---

## 📋 重新运行合并脚本

如果需要重新合并数据（例如输入文件有更新），可以运行：

```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te_newlabel

# 基本用法
python merge_tissue_data.py --data_dir preprocess_data

# 指定输出文件
python merge_tissue_data.py --data_dir preprocess_data --output my_merged_data.csv

# 不生成统计报告
python merge_tissue_data.py --data_dir preprocess_data --no_report
```

---

## 💡 数据特点和建议

### 数据特点

1. **稀疏标签**: 大量NA标签，这是多组织生物学数据的常见特征
2. **多标签学习**: 每个样本可能有多个tissue的标签
3. **标签关联性**: 不同tissue的标签可能存在生物学关联

### 训练建议

1. **使用多标签学习方法**: 一个模型同时预测9个tissue
2. **忽略NA标签**: 在损失函数中设置`ignore_index=-100`来忽略NA
3. **数据增强**: 考虑使用半监督学习方法利用没有标签的样本
4. **验证策略**: 使用stratified split确保train/valid/test的标签分布相似

### 模型设计

```python
# 推荐的模型设计
class MultiTissueTE:
    def __init__(self):
        self.num_tissues = 9  # 9个tissue
        self.num_classes = 2  # 2个类别（0=Low, 1=High）
        
    def forward(self, sequence):
        # 输出: [batch, 9, 2]
        # 9个tissue，每个有2个类别的logits
        pass
        
    def loss(self, logits, labels):
        # labels中的NA应该被映射为-100并被忽略
        # 使用CrossEntropyLoss(ignore_index=-100)
        pass
```

---

## 🔧 故障排除

### 问题1: 内存不足

如果数据太大无法一次加载到内存：

```python
# 分批读取
import pandas as pd

chunk_size = 10000
for chunk in pd.read_csv('merged_tissue_data.csv', chunksize=chunk_size):
    # 处理每个chunk
    process(chunk)
```

### 问题2: 训练时标签格式问题

确保NA被正确处理：

```python
import pandas as pd
import numpy as np

df = pd.read_csv('merged_tissue_data.csv')

# 将空值替换为特殊值（如-100）
label_cols = [col for col in df.columns if col.endswith('_TE_label')]
df[label_cols] = df[label_cols].fillna(-100)
```

---

## 📞 相关文件

- `merge_tissue_data.py` - 数据合并脚本
- `merged_tissue_data.csv` - 合并后的数据文件
- `merge_report.txt` - 详细统计报告
- `DATA_MERGE_README.md` - 本文档

---

**创建时间**: 2025-10-22  
**最后更新**: 2025-10-22  
**数据版本**: v1.0

