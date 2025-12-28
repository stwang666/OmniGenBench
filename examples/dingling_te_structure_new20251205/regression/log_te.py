import pandas as pd
import numpy as np
from pathlib import Path

# 数据目录
data_dir = Path("/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205/regression")

# 处理三个文件
for filename in ["train.csv", "valid.csv", "test.csv"]:
    filepath = data_dir / filename
    
    # 读取数据
    df = pd.read_csv(filepath)
    
    # 对TE列取log2
    # 注意：log2(0)是负无穷，需要先过滤或添加小的epsilon
    df['TE'] = np.log2(df['TE'] + 1e-10)  # 加一个很小的值避免log(0)
    # 或者更好的方式：
    # df['TE'] = np.log2(df['TE'].clip(lower=0.01))  # 截断最小值为0.01
    
    # 保存（覆盖原文件或保存为新文件）
    df.to_csv(filepath, index=False)
    print(f"✅ {filename}: TE值已转换为log2")
    print(f"   新的TE范围: [{df['TE'].min():.3f}, {df['TE'].max():.3f}]")