"""
Per-Tissue 3-label分类标注脚本
对每个tissue分别计算33%和66%分位数，确保每个tissue内label分布平衡
同时修正tissue名称格式（点号改为连字符）
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

# 输入输出路径
in_path = Path("/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205/9tissue_structure_te_hc_deseq2_tp_split.csv")
out_path = in_path.with_name("9tissue_structure_te_hc_deseq2_tp_split_labeled_per_tissue.csv")

# Tissue名称映射（修正格式）
TISSUE_NAME_MAP = {
    "Prophase.I.pollen": "Prophase-I-pollen",
    "Tricellular.pollen": "Tricellular-pollen",
}

def normalize_tissue_name(tissue_name: str) -> str:
    """标准化tissue名称，将点号替换为连字符"""
    return TISSUE_NAME_MAP.get(tissue_name, tissue_name)

def interp_quantile(vals_sorted: list, p: float) -> float:
    """计算分位数（线性插值）"""
    if not vals_sorted:
        return float("nan")
    n = len(vals_sorted)
    pos = p * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals_sorted[int(pos)]
    frac = pos - lo
    return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

def label_value(v: float, p33: float, p66: float) -> int:
    """根据分位数阈值标注label"""
    if v < p33:
        return 0  # Low
    elif v < p66:
        return 1  # Medium
    else:
        return 2  # High

# 读取数据
rows = []
tissue_data = defaultdict(list)  # {tissue: [(row, te_value), ...]}

with in_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 标准化tissue名称
        original_tissue = row["TISSUE"]
        normalized_tissue = normalize_tissue_name(original_tissue)
        row["TISSUE"] = normalized_tissue
        
        te = float(row["TE"])
        rows.append(row)
        tissue_data[normalized_tissue].append((row, te))

print(f"📊 读取数据: 总计 {len(rows)} 样本")
print(f"📊 发现 {len(tissue_data)} 个tissue")

# 对每个tissue分别计算分位数和标注
tissue_stats = {}
all_label_counts = [0, 0, 0]

for tissue, data in tissue_data.items():
    # 提取该tissue的TE值并排序
    te_values = sorted([te for _, te in data])
    n = len(te_values)
    
    # 计算该tissue的33%和66%分位数
    p33 = interp_quantile(te_values, 1/3)
    p66 = interp_quantile(te_values, 2/3)
    
    # 标注该tissue的所有样本
    label_counts = [0, 0, 0]
    for row, te in data:
        lab = label_value(te, p33, p66)
        row["label"] = str(lab)
        label_counts[lab] += 1
        all_label_counts[lab] += 1
    
    # 保存统计信息
    tissue_stats[tissue] = {
        "count": n,
        "min": min(te_values),
        "max": max(te_values),
        "mean": sum(te_values) / n,
        "median": interp_quantile(te_values, 0.5),
        "p33": p33,
        "p66": p66,
        "label_0": label_counts[0],
        "label_1": label_counts[1],
        "label_2": label_counts[2],
    }

# 确保fieldnames包含label列
fieldnames = list(rows[0].keys())
if "label" not in fieldnames:
    fieldnames.append("label")

# 写入标注后的数据
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# 打印统计信息
print(f"\n{'='*80}")
print(f"✅ 标注完成！输出文件: {out_path}")
print(f"{'='*80}")

print(f"\n📊 全局统计:")
print(f"  总样本数: {len(rows)}")
print(f"  Label 0 (Low):    {all_label_counts[0]:6d} ({all_label_counts[0]/len(rows)*100:5.1f}%)")
print(f"  Label 1 (Medium): {all_label_counts[1]:6d} ({all_label_counts[1]/len(rows)*100:5.1f}%)")
print(f"  Label 2 (High):   {all_label_counts[2]:6d} ({all_label_counts[2]/len(rows)*100:5.1f}%)")

print(f"\n📊 各Tissue统计:")
print(f"{'Tissue':<25s} {'Count':>6s} {'TE Range':>20s} {'P33':>8s} {'P66':>8s} {'L0':>6s} {'L1':>6s} {'L2':>6s} {'平衡度':>8s}")
print(f"{'-'*25} {'-'*6} {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

for tissue in sorted(tissue_stats.keys()):
    stats = tissue_stats[tissue]
    te_range = f"{stats['min']:.2f}-{stats['max']:.2f}"
    
    # 计算平衡度（标准差/均值）
    l0_pct = stats['label_0'] / stats['count'] * 100
    l1_pct = stats['label_1'] / stats['count'] * 100
    l2_pct = stats['label_2'] / stats['count'] * 100
    balance = max(l0_pct, l1_pct, l2_pct) - min(l0_pct, l1_pct, l2_pct)
    
    print(f"{tissue:<25s} {stats['count']:6d} {te_range:>20s} {stats['p33']:8.2f} {stats['p66']:8.2f} "
          f"{stats['label_0']:6d} {stats['label_1']:6d} {stats['label_2']:6d} {balance:7.1f}%")

print(f"\n💡 说明:")
print(f"  - 使用Per-Tissue分位数标注，每个tissue独立计算33%和66%分位数")
print(f"  - 这样可以确保每个tissue内的label分布平衡（约33%/33%/33%）")
print(f"  - Tissue名称已标准化: Prophase.I.pollen → Prophase-I-pollen")
print(f"  - Tissue名称已标准化: Tricellular.pollen → Tricellular-pollen")
