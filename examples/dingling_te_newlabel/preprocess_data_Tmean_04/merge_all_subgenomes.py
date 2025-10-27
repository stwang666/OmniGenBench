#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: merge_all_subgenomes.py
# time: 10:00 22/10/2025
# author: 合并所有亚基因组(A, B, D)的脚本
# 分别合并A、B和D三个亚基因组中的数据

import sys
import os
from pathlib import Path

# 添加A_TE目录到Python路径，以便导入merge_tissue_data模块
current_dir = Path(__file__).parent
a_te_dir = current_dir / "A_TE"
sys.path.insert(0, str(a_te_dir))

from merge_tissue_data import merge_all_subgenomes, merge_tissue_files

def main():
    """
    主函数：处理所有亚基因组的数据合并
    """
    base_dir = Path(__file__).parent
    
    print("🧬 开始处理所有亚基因组数据合并")
    print(f"📁 基础目录: {base_dir}")
    
    # 检查目录结构
    subgenome_dirs = ['A_TE', 'B_TE', 'D_TE']
    missing_dirs = []
    
    for subgenome_dir in subgenome_dirs:
        if not (base_dir / subgenome_dir).exists():
            missing_dirs.append(subgenome_dir)
    
    if missing_dirs:
        print(f"⚠️  警告: 缺少以下目录: {', '.join(missing_dirs)}")
        print("请确保目录结构如下:")
        print("preprocess_data_Tmean_04/")
        print("├── A_TE/")
        print("├── B_TE/")
        print("└── D_TE/")
        return
    
    # 执行所有亚基因组的合并
    try:
        results = merge_all_subgenomes(
            base_dir=base_dir,
            generate_report=True
        )
        
        if results:
            print(f"\n🎉 所有亚基因组处理完成！")
            print(f"成功处理的亚基因组: {', '.join(results.keys())}")
            
            # 显示每个亚基因组的样本数
            for subgenome, df in results.items():
                print(f"  {subgenome}亚基因组: {len(df)} 个样本")
        else:
            print("❌ 没有成功处理任何亚基因组")
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        return
    
    print("\n📊 处理完成！生成的文件:")
    print("  - merged_A_tissue_data.csv (A亚基因组合并数据)")
    print("  - merged_B_tissue_data.csv (B亚基因组合并数据)")
    print("  - merged_D_tissue_data.csv (D亚基因组合并数据)")
    print("  - merge_A_report.txt (A亚基因组统计报告)")
    print("  - merge_B_report.txt (B亚基因组统计报告)")
    print("  - merge_D_report.txt (D亚基因组统计报告)")
    print("  - overall_merge_report.txt (总体统计报告)")

if __name__ == '__main__':
    main()
