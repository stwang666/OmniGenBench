#!/usr/bin/env python3
"""修改 CSV 文件中的 split 列为对应的文件名。

将 train.csv、test.csv 和 valid.csv 中的 split 列值更新为：
- train.csv -> "train"
- test.csv -> "test"  
- valid.csv -> "val"
"""

import csv
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import shutil


def fix_split_column(csv_path: Path, target_split: str) -> None:
    """修改 CSV 文件中的 split 列值。
    
    Args:
        csv_path: CSV 文件路径
        target_split: 要设置的目标 split 值
    """
    # 读取 CSV 文件
    rows = []
    headers = None
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            print(f"警告: {csv_path} 文件为空或格式不正确", file=sys.stderr)
            return
        
        if "split" not in headers:
            print(f"警告: {csv_path} 中没有 'split' 列", file=sys.stderr)
            return
        
        for row in reader:
            row["split"] = target_split
            rows.append(row)
    
    # 写入修改后的内容
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"已更新 {csv_path.name}: {len(rows)} 行，split 列设置为 '{target_split}'")
    
    return len(rows)


def main():
    """主函数。"""
    base_dir = Path(__file__).parent
    
    # 定义文件名到 split 值的映射
    file_mappings = {
        "train.csv": "train",
        "test.csv": "test",
        "valid.csv": "val",
    }
    
    # 处理每个文件
    for filename, target_split in file_mappings.items():
        csv_path = base_dir / filename
        if csv_path.exists():
            fix_split_column(csv_path, target_split)
        else:
            print(f"警告: 文件 {filename} 不存在", file=sys.stderr)
    
    print("\n所有文件处理完成！")


if __name__ == "__main__":
    main()

