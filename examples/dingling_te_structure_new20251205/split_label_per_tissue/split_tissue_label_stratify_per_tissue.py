"""
按 tissue 与 label 联合分层将数据拆分为 train/valid/test (8:1:1)。
使用 per-tissue 标注的数据

输入：同目录下的 9tissue_structure_te_hc_deseq2_tp_split_labeled_per_tissue.csv
输出：train_per_tissue.csv、valid_per_tissue.csv、test_per_tissue.csv（保存在同目录）
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "9tissue_structure_te_hc_deseq2_tp_split_labeled_per_tissue.csv"
TRAIN = BASE_DIR / "train_per_tissue.csv"
VALID = BASE_DIR / "valid_per_tissue.csv"
TEST = BASE_DIR / "test_per_tissue.csv"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 42


def read_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def stratified_split(
    rows: Iterable[dict], seed: int = SEED
) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    groups = defaultdict(list)

    for row in rows:
        key = (row["tissue"], row["label"])
        groups[key].append(row)

    train: List[dict] = []
    valid: List[dict] = []
    test: List[dict] = []

    print(f"\n📊 各Tissue×Label组合的样本分布:")
    print(f"{'Tissue':<25s} {'Label':<6s} {'Total':>6s} {'Train':>6s} {'Valid':>6s} {'Test':>6s}")
    print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for key in sorted(groups.keys()):
        tissue, label = key
        items = groups[key]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        n_test = n - n_train - n_valid

        # 防止极小分组出现负数
        if n_test < 0:
            n_test = 0
            n_valid = n - n_train

        train.extend(items[:n_train])
        valid.extend(items[n_train : n_train + n_valid])
        test.extend(items[n_train + n_valid :])
        
        print(f"{tissue:<25s} {label:<6s} {n:6d} {n_train:6d} {n_valid:6d} {n_test:6d}")

    return train, valid, test


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_distribution(rows: List[dict], dataset_name: str) -> None:
    """分析数据集的label和tissue分布"""
    label_counts = defaultdict(int)
    tissue_counts = defaultdict(int)
    tissue_label_counts = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        label = row['label']
        tissue = row['tissue']
        label_counts[label] += 1
        tissue_counts[tissue] += 1
        tissue_label_counts[tissue][label] += 1
    
    total = len(rows)
    print(f"\n📊 {dataset_name} 数据集分析:")
    print(f"  总样本数: {total}")
    print(f"\n  Label分布:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / total * 100
        print(f"    Label {label}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n  Tissue分布:")
    for tissue in sorted(tissue_counts.keys()):
        count = tissue_counts[tissue]
        pct = count / total * 100
        l0 = tissue_label_counts[tissue]['0']
        l1 = tissue_label_counts[tissue]['1']
        l2 = tissue_label_counts[tissue]['2']
        print(f"    {tissue:<25s}: {count:6d} ({pct:5.1f}%) | L0:{l0:5d} L1:{l1:5d} L2:{l2:5d}")


def main() -> None:
    fieldnames, rows = read_rows(SOURCE)
    train, valid, test = stratified_split(rows, seed=SEED)

    write_rows(TRAIN, fieldnames, train)
    write_rows(VALID, fieldnames, valid)
    write_rows(TEST, fieldnames, test)

    print(f"\n{'='*80}")
    print(f"✅ 拆分完成!")
    print(f"  - 训练集: {TRAIN} ({len(train)} 样本)")
    print(f"  - 验证集: {VALID} ({len(valid)} 样本)")
    print(f"  - 测试集: {TEST} ({len(test)} 样本)")
    print(f"{'='*80}")

    # 分析各数据集的分布
    analyze_distribution(train, "训练集")
    analyze_distribution(valid, "验证集")
    analyze_distribution(test, "测试集")


if __name__ == "__main__":
    main()
