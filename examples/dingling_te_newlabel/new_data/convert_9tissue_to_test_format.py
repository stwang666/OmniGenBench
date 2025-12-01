#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 `9tissue_dot.txt` 中的逐组织记录整理成 test.csv 所需的宽表格式。

功能:
1. 读取原始 9 组织数据，清洗列名与标签
2. 以 (ID, sequence) 为唯一键，将 9 个组织的 label 透视到单行
3. 输出与 examples/dingling_te_newlabel/preprocess_data_Tmean_02/test.csv 相同的列结构

使用示例:
    python convert_9tissue_to_test_format.py \
        --input /path/to/9tissue_dot.txt \
        --output /path/to/9tissue_dot_test_format.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

TISSUE_ORDER: list[str] = [
    "root",
    "seedling",
    "leaf",
    "FMI",
    "FOD",
    "Prophase-I-pollen",
    "Tricellular-pollen",
    "flag",
    "grain",
]
LABEL_COLUMNS: list[str] = [f"{tissue}_TE_label" for tissue in TISSUE_ORDER]
REQUIRED_COLUMNS: set[str] = {"ID", "SEQ", "structure", "TE", "tissue", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 9tissue_dot.txt 整理成 test.csv 所需格式"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("9tissue_dot.txt"),
        help="原始 9tissue_dot.txt 路径 (默认: 脚本同目录)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径 (默认: 与输入同目录, 文件名加 _test_format)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="打印每个组织的标签统计",
    )
    parser.add_argument(
        "--drop-empty-labels",
        action="store_true",
        help="丢弃九个组织标签全部为空的样本",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="启用 8:1:1 的 train/valid/test 随机划分",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=None,
        help="存放划分结果的目录 (默认: 输出文件所在目录)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分随机种子 (默认: 42)",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {', '.join(sorted(missing))}")


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    validate_columns(df)

    # 统一列名与取值
    df = df.rename(columns={"SEQ": "sequence"}).copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["tissue"] = df["tissue"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    return df


def detect_inconsistent_sequences(df: pd.DataFrame) -> list[str]:
    """返回存在多个不同 sequence 的 ID 列表"""
    seq_counts = df.groupby("ID")["sequence"].nunique()
    inconsistent_ids = seq_counts[seq_counts > 1]
    return inconsistent_ids.index.tolist()


def deduplicate_by_tissue(df: pd.DataFrame) -> pd.DataFrame:
    """
    针对相同 (ID, sequence, tissue) 的多条记录，优先保留存在标签的记录。
    """
    # 让有标签的行优先，NaN 排在最后
    sorted_df = df.sort_values(by="label", na_position="last")
    dedup = sorted_df.drop_duplicates(subset=["ID", "sequence", "tissue"], keep="first")
    return dedup


def pivot_labels(df: pd.DataFrame) -> pd.DataFrame:
    pivot_df = (
        df.pivot(index=["ID", "sequence"], columns="tissue", values="label")
        .reindex(columns=TISSUE_ORDER)
        .rename(columns={t: f"{t}_TE_label" for t in TISSUE_ORDER})
    )
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df[["ID", "sequence", *LABEL_COLUMNS]]
    return pivot_df


def report_statistics(df: pd.DataFrame) -> None:
    print("\n📊 标签统计 (非空/0/1/NA)")
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            continue
        total = len(df)
        non_na = df[col].notna().sum()
        zero_count = (df[col] == 0).sum()
        one_count = (df[col] == 1).sum()
        na_count = df[col].isna().sum()
        print(
            f"{col:30s} -> 有效: {non_na:5d} "
            f"(0:{zero_count:5d}, 1:{one_count:5d}, NA:{na_count:5d}, {non_na/total:5.1%})"
        )


def ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def drop_rows_with_all_nan_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df[LABEL_COLUMNS].notna().any(axis=1)
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    return filtered, removed


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    valid_df = shuffled.iloc[train_end:valid_end].reset_index(drop=True)
    test_df = shuffled.iloc[valid_end:].reset_index(drop=True)
    return train_df, valid_df, test_df


def convert_dataset(
    input_path: Path,
    output_path: Path,
    show_report: bool = False,
    drop_empty: bool = False,
    split: bool = False,
    split_output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    raw_df = load_raw_dataframe(input_path)

    inconsistent_ids = detect_inconsistent_sequences(raw_df)
    if inconsistent_ids:
        print(
            f"⚠️  警告: 发现 {len(inconsistent_ids)} 个 ID 对应多个不同的 sequence，"
            "已按 (ID, sequence) 组合拆分。"
        )

    dedup_df = deduplicate_by_tissue(raw_df)
    wide_df = pivot_labels(dedup_df)

    if drop_empty:
        wide_df, removed_rows = drop_rows_with_all_nan_labels(wide_df)
        print(f"🧹 已移除 {removed_rows} 条九个标签均为空的样本，剩余 {len(wide_df)} 条")

    ensure_output_dir(output_path)
    wide_df.to_csv(output_path, index=False)
    print(f"✅ 已输出 {len(wide_df)} 条样本到 {output_path}")

    if split:
        target_dir = split_output_dir or output_path.parent
        target_dir = target_dir.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        train_df, valid_df, test_df = split_dataset(wide_df, seed=seed)
        train_path = target_dir / "train.csv"
        valid_path = target_dir / "valid.csv"
        test_path = target_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(
            "📂 已根据 8:1:1 划分并保存:\n"
            f"   train: {len(train_df)} -> {train_path}\n"
            f"   valid: {len(valid_df)} -> {valid_path}\n"
            f"   test : {len(test_df)} -> {test_path}"
        )

    if show_report:
        report_statistics(wide_df)

    return output_path


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.expanduser().resolve()
    output_path: Path
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_test_format.csv")
    else:
        output_path = args.output.expanduser().resolve()

    convert_dataset(
        input_path=input_path,
        output_path=output_path,
        show_report=args.report,
        drop_empty=args.drop_empty_labels,
        split=args.split,
        split_output_dir=args.split_output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


