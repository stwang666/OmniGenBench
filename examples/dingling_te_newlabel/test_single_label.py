#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单标签分类代码
"""

import sys
import os
sys.path.append('/home/sw1136/OmniGenBench')

def test_imports():
    """测试导入"""
    try:
        from biclass_te_single_tissue_modified import BiClassTEDataset, OmniModelForBiClassTESequenceClassification
        print("✅ 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_dataset_creation():
    """测试数据集创建"""
    try:
        from omnigenbench import OmniTokenizer
        from biclass_te_single_tissue_modified import BiClassTEDataset
        
        # 加载tokenizer
        tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-52M", trust_remote_code=True)
        
        # 测试数据集创建
        dataset = BiClassTEDataset(
            dataset_name_or_path="/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data_Tmean_02/FOD_TE",
            tokenizer=tokenizer,
            max_length=512
        )
        
        print("✅ 数据集创建成功")
        return True
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 开始测试单标签分类代码...")
    
    print("\n1. 测试导入...")
    if not test_imports():
        sys.exit(1)
    
    print("\n2. 测试数据集创建...")
    if not test_dataset_creation():
        sys.exit(1)
    
    print("\n🎉 所有测试通过！代码可以正常运行。")
