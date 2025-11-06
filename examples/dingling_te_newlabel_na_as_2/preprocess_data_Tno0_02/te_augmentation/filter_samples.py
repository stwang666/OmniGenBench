import pandas as pd

# 读取CSV文件
input_file = 'train.csv'
df = pd.read_csv(input_file)

# 标签列名
label_columns = [
    'root_TE_label',
    'seedling_TE_label', 
    'leaf_TE_label',
    'FMI_TE_label',
    'FOD_TE_label',
    'Prophase-I-pollen_TE_label',
    'Tricellular-pollen_TE_label',
    'flag_TE_label',
    'grain_TE_label'
]

# 筛选条件：标签0和标签1的总和大于等于5个
def count_labels(row):
    """统计每行中标签0和标签1的数量"""
    labels = [row[col] for col in label_columns if pd.notna(row[col])]
    count_0 = sum(1 for label in labels if label == 0.0 or label == 0)
    count_1 = sum(1 for label in labels if label == 1.0 or label == 1)
    return count_0, count_1

# 应用筛选条件：标签0和1的总和>=5
def filter_samples(row):
    count_0, count_1 = count_labels(row)
    return (count_0 + count_1) >= 3

# 筛选数据
filtered_df = df[df.apply(filter_samples, axis=1)].copy()

# 保存结果
output_file = 'train_filtered.csv'
filtered_df.to_csv(output_file, index=False)

print(f"原始样本数: {len(df)}")
print(f"筛选后样本数（标签0和1的总和>=5）: {len(filtered_df)}")

if len(filtered_df) == 0:
    print("\n注意：没有找到符合条件的样本。")
    print("\n空的CSV文件已保存，包含原始表头。")
else:
    print(f"\n结果已保存到: {output_file}")
    # 显示一些统计信息
    print("\n筛选后的样本标签统计示例（前10个样本）:")
    for idx, row in filtered_df.head(10).iterrows():
        count_0, count_1 = count_labels(row)
        total = count_0 + count_1
        print(f"样本 {row['ID']}: 标签0数量={count_0}, 标签1数量={count_1}, 总和={total}")

