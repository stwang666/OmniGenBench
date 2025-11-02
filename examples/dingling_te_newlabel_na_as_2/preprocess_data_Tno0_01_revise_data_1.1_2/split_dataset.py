#数据划分
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('/home/sw1136/OmniGenBench/examples/dingling_te_newlabel_na_as_2/preprocess_data_Tno0_01_revise_data_1.1_2/merged_tissue_labels_filtered.csv')

# 随机划分（不使用stratify）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)

print(f"训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(valid_df)}")
print(f"测试集样本数: {len(test_df)}")
# 训练集样本数: 47015
# 验证集样本数: 5877
# 测试集样本数: 5877