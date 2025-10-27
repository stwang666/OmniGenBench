# import pandas as pd

# df = pd.read_csv('/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data/merged_tissue_data.csv')

# # 获取所有label列
# label_cols = [col for col in df.columns if col.endswith('_TE_label')]

# # 只保留至少有一个tissue有标签的样本
# df_filtered = df[df[label_cols].notna().any(axis=1)]

# print(f"原始样本数: {len(df)}")
# print(f"过滤后样本数: {len(df_filtered)}")

# df_filtered.to_csv('merged_tissue_data_filtered.csv', index=False)

#数据划分
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('/home/sw1136/OmniGenBench/examples/dingling_te_newlabel/preprocess_data/merged_tissue_data_filtered.csv')

# 随机划分（不使用stratify）
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(valid_df)}")
print(f"测试集样本数: {len(test_df)}")

# 训练集样本数: 42223
# 验证集样本数: 9048
# 测试集样本数: 9048