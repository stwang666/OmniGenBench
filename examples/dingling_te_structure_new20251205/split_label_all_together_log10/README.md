# TE数据处理和三类划分

## 📁 文件说明

1. **process_te_log10.py** - 数据预处理脚本
2. **split_triclass.py** - 三类划分脚本
3. **9tissue_structure_te_hc_deseq2_tp_split_log10.csv** - log10转换后的数据
4. **README.md** - 本说明文档

---

## 1️⃣ 数据预处理代码

### 功能
- 对TE列取log10转换
- 修改组织名称（Prophase.I.pollen → Prophase-I-pollen）

### 代码
```python
import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_csv('9tissue_structure_te_hc_deseq2_tp_split.csv')

# 1. 对TE取log10
df['TE'] = np.log10(df['TE'])

# 2. 修改组织名称
df['TISSUE'] = df['TISSUE'].str.replace('Prophase.I.pollen', 'Prophase-I-pollen', regex=False)
df['TISSUE'] = df['TISSUE'].str.replace('Tricellular.pollen', 'Tricellular-pollen', regex=False)

# 保存结果
df.to_csv('9tissue_structure_te_hc_deseq2_tp_split_log10.csv', index=False)
```

### 转换结果
- **原始TE范围**: 0.047 ~ 137.088
- **log10(TE)范围**: -1.326 ~ 2.137
- **样本数**: 73,355

---

## 2️⃣ 数据分布分析

### log10(TE) 统计量
```
样本数:      73,355
均值:        -0.028
标准差:       0.150
中位数:      -0.025
第25分位数:  -0.113
第75分位数:   0.062
```

### 数据特点
- ✅ 数据已经较为对称（均值≈中位数）
- ✅ 标准差较小，数据相对集中
- ✅ 适合用统计方法划分

---

## 3️⃣ 三类划分方案对比

### 方案1: mean ± std
**阈值**: [-0.178, 0.122]

| 类别 | 范围 | 样本数 | 占比 |
|------|------|--------|------|
| 0 | log10(TE) < -0.178 | 9,926 | 13.5% |
| 1 | -0.178 ≤ log10(TE) < 0.122 | 53,573 | **73.0%** ⚠️ |
| 2 | log10(TE) ≥ 0.122 | 9,856 | 13.4% |

**评价**:
- ❌ 类别1占比过大（73%），不均衡
- ✅ 符合正态分布的"68-95-99.7规则"
- 💡 适合：想要突出极端值（尾部样本）

---

### 方案2: mean ± 0.5×std (推荐⭐)
**阈值**: [-0.103, 0.047]

| 类别 | 范围 | 样本数 | 占比 |
|------|------|--------|------|
| 0 | log10(TE) < -0.103 | 19,962 | 27.2% |
| 1 | -0.103 ≤ log10(TE) < 0.047 | 32,420 | **44.2%** |
| 2 | log10(TE) ≥ 0.047 | 20,973 | 28.6% |

**评价**:
- ✅ 相对均衡（27%, 44%, 29%）
- ✅ 保留统计学意义
- ✅ 类别0和类别2数量接近
- 💡 **推荐使用**：兼顾统计特性和实用性

---

### 方案3: 等分位数 (33.3%/66.7%)
**阈值**: [-0.079, 0.029]

| 类别 | 范围 | 样本数 | 占比 |
|------|------|--------|------|
| 0 | log10(TE) < -0.079 | 24,452 | 33.3% |
| 1 | -0.079 ≤ log10(TE) < 0.029 | 24,451 | **33.3%** |
| 2 | log10(TE) ≥ 0.029 | 24,452 | 33.3% |

**评价**:
- ✅ 完全均等（各33.3%）
- ❌ 不考虑数据的统计分布
- 💡 适合：需要严格平衡的训练集

---

## 4️⃣ 推荐方案

### 🏆 最佳选择：方案2 (mean ± 0.5×std)

**理由**：
1. **相对均衡** - 三类占比为 27% : 44% : 29%
2. **统计学基础** - 基于数据的自然分布
3. **实用性强** - 中间类别略多符合实际情况
4. **生物学意义** - 保留了低、中、高翻译效率的区分度

### 使用代码
```python
import pandas as pd
import numpy as np

# 读取log10转换后的数据
df = pd.read_csv('9tissue_structure_te_hc_deseq2_tp_split_log10.csv')

# 计算阈值
mean = df['TE'].mean()
std = df['TE'].std()
threshold_low = mean - 0.5 * std
threshold_high = mean + 0.5 * std

# 分类
df['label'] = 0
df.loc[df['TE'] >= threshold_low, 'label'] = 1
df.loc[df['TE'] >= threshold_high, 'label'] = 2

# 保存
df.to_csv('9tissue_structure_te_hc_deseq2_tp_split_triclass.csv', index=False)

# 查看分布
print(df['label'].value_counts().sort_index())
```

---

## 5️⃣ 快速使用

### 方式1: 使用现成脚本
```bash
cd /home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205/split_label_all_together_log10
python split_triclass.py
```

### 方式2: 自定义方法
```python
from split_triclass import split_triclass

# 使用方案2（推荐）
df = split_triclass(
    input_file='9tissue_structure_te_hc_deseq2_tp_split_log10.csv',
    method='mean_half_std',
    save_plot=True
)

# 或使用等分位数
df = split_triclass(
    input_file='9tissue_structure_te_hc_deseq2_tp_split_log10.csv',
    method='quantile',
    save_plot=True
)
```

---

## 6️⃣ 输出文件

| 文件 | 说明 |
|------|------|
| `*_triclass_mean_half_std.csv` | 使用方案2的分类结果 |
| `*_triclass_quantile.csv` | 使用方案3的分类结果 |
| `triclass_distribution.png` | 数据分布可视化图 |

---

## 📊 类别含义

| 标签 | log10(TE)范围 | 原始TE范围 | 生物学意义 |
|------|---------------|------------|-----------|
| 0 | < -0.103 | < 0.789 | 低翻译效率 |
| 1 | -0.103 ~ 0.047 | 0.789 ~ 1.114 | 中等翻译效率 |
| 2 | ≥ 0.047 | ≥ 1.114 | 高翻译效率 |

---

## ❓ 常见问题

### Q1: 为什么要取log10？
**A**: 原始TE值范围很大(0.047~137)，取对数后：
- 减小数值范围，便于模型训练
- 使数据分布更接近正态分布
- 使不同数量级的值具有可比性

### Q2: 为什么推荐方案2而不是等分位数？
**A**: 
- 方案2保留了数据的统计特性
- 中间类别略多符合实际（大部分基因翻译效率接近平均水平）
- 如果需要完全均衡，可以选择方案3

### Q3: 能否直接用mean±std？
**A**: 
- 可以，但会导致中间类别占73%，不够均衡
- mean±0.5×std是更好的折中方案

---

## 📧 联系方式

如有问题，请参考代码注释或联系开发者。











