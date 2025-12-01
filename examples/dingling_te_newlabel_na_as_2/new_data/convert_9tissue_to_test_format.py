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


def convert_dataset(input_path: Path, output_path: Path, show_report: bool = False) -> Path:
    raw_df = load_raw_dataframe(input_path)

    inconsistent_ids = detect_inconsistent_sequences(raw_df)
    if inconsistent_ids:
        print(
            f"⚠️  警告: 发现 {len(inconsistent_ids)} 个 ID 对应多个不同的 sequence，"
            "已按 (ID, sequence) 组合拆分。"
        )

    dedup_df = deduplicate_by_tissue(raw_df)
    wide_df = pivot_labels(dedup_df)

    ensure_output_dir(output_path)
    wide_df.to_csv(output_path, index=False)
    print(f"✅ 已输出 {len(wide_df)} 条样本到 {output_path}")

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

    convert_dataset(input_path, output_path, show_report=args.report)


if __name__ == "__main__":
    main()


