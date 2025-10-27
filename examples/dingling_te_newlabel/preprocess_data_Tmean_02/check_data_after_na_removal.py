#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查去除NA数据后，训练、验证和测试集分别还剩多少数据
"""

import pandas as pd
import os

def check_data_after_na_removal():
    """检查去除NA数据后的数据量"""
    
    print("=" * 60)
    print("检查去除NA数据后的数据量")
    print("=" * 60)
    
    data_dir = "/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE"
    
    splits = ['train', 'valid', 'test']
    results = []
    
    for split in splits:
        file_path = os.path.join(data_dir, f"{split}.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        print(f"\n📊 处理 {split.upper()} 数据...")
        
        # 读取数据
        df = pd.read_csv(file_path)
        original_count = len(df)
        
        print(f"   原始样本数: {original_count:,}")
        
        # 检查NA情况
        na_in_label = df['label'].isna().sum()
        na_in_seq = df['Seq'].isna().sum()
        total_na = na_in_label + na_in_seq
        
        print(f"   label列NA数: {na_in_label:,}")
        print(f"   Seq列NA数: {na_in_seq:,}")
        print(f"   总NA数: {total_na:,}")
        
        # 去除NA数据
        df_clean = df.dropna(subset=['label', 'Seq'])
        clean_count = len(df_clean)
        removed_count = original_count - clean_count
        
        print(f"   去除NA后样本数: {clean_count:,}")
        print(f"   去除的样本数: {removed_count:,}")
        print(f"   保留比例: {clean_count/original_count*100:.2f}%")
        
        # 检查标签分布
        if clean_count > 0:
            label_counts = df_clean['label'].value_counts()
            print(f"   标签分布:")
            for label, count in label_counts.items():
                print(f"     {label}: {count:,} ({count/clean_count*100:.2f}%)")
        
        results.append({
            'split': split,
            'original_count': original_count,
            'clean_count': clean_count,
            'removed_count': removed_count,
            'retention_rate': clean_count/original_count*100
        })
    
    # 生成汇总报告
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    
    if results:
        total_original = sum(r['original_count'] for r in results)
        total_clean = sum(r['clean_count'] for r in results)
        total_removed = sum(r['removed_count'] for r in results)
        
        print(f"\n📈 总体统计:")
        print(f"   原始总样本数: {total_original:,}")
        print(f"   去除NA后总样本数: {total_clean:,}")
        print(f"   总去除样本数: {total_removed:,}")
        print(f"   总体保留比例: {total_clean/total_original*100:.2f}%")
        
        print(f"\n📋 各数据集详情:")
        print(f"{'数据集':<10} {'原始样本':<12} {'去除NA后':<12} {'去除数量':<12} {'保留比例':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['split']:<10} {r['original_count']:<12,} {r['clean_count']:<12,} {r['removed_count']:<12,} {r['retention_rate']:<10.2f}%")
        
        # 保存详细报告
        report_file = os.path.join(data_dir, 'na_removal_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("NA数据去除报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总体统计:\n")
            f.write(f"  原始总样本数: {total_original:,}\n")
            f.write(f"  去除NA后总样本数: {total_clean:,}\n")
            f.write(f"  总去除样本数: {total_removed:,}\n")
            f.write(f"  总体保留比例: {total_clean/total_original*100:.2f}%\n\n")
            
            f.write(f"各数据集详情:\n")
            for r in results:
                f.write(f"  {r['split']}: {r['original_count']:,} -> {r['clean_count']:,} (去除{r['removed_count']:,}, 保留{r['retention_rate']:.2f}%)\n")
        
        print(f"\n💾 详细报告已保存到: {report_file}")
        
        # 检查是否有问题
        print(f"\n🔍 数据质量检查:")
        low_retention = [r for r in results if r['retention_rate'] < 50]
        if low_retention:
            print(f"   ⚠️  保留比例低于50%的数据集:")
            for r in low_retention:
                print(f"      {r['split']}: {r['retention_rate']:.2f}%")
        else:
            print(f"   ✅ 所有数据集的保留比例都在50%以上")
            
        empty_datasets = [r for r in results if r['clean_count'] == 0]
        if empty_datasets:
            print(f"   ⚠️  去除NA后为空的数据集:")
            for r in empty_datasets:
                print(f"      {r['split']}")
        else:
            print(f"   ✅ 所有数据集去除NA后都有数据")

if __name__ == "__main__":
    check_data_after_na_removal()






