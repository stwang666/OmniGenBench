"""
按 tissue 与 label 联合分层将数据拆分为 train/valid/test (8:1:1)。

输入：同目录下的 9tissue_structure_te_hc_deseq2_tp_split_labeled.csv
输出：train.csv、valid.csv、test.csv（保存在同目录）
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "9tissue_structure_te_hc_deseq2_tp_split_labeled.csv"
TRAIN = BASE_DIR / "train.csv"
VALID = BASE_DIR / "valid.csv"
TEST = BASE_DIR / "test.csv"

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

    for key, items in groups.items():
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

    return train, valid, test


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    fieldnames, rows = read_rows(SOURCE)
    train, valid, test = stratified_split(rows, seed=SEED)

    write_rows(TRAIN, fieldnames, train)
    write_rows(VALID, fieldnames, valid)
    write_rows(TEST, fieldnames, test)

    print(
        f"done: total={len(rows)}, train={len(train)}, "
        f"valid={len(valid)}, test={len(test)}"
    )


if __name__ == "__main__":
    main()
