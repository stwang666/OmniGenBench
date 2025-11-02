import pandas as pd

# 读取三个文件
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# 统计每个文件
print("=== 原始统计 ===")
print(f"train.csv 总行数: {len(train_df)}")
print(f"valid.csv 总行数: {len(valid_df)}")
print(f"test.csv 总行数: {len(test_df)}")
print(f"总计: {len(train_df) + len(valid_df) + len(test_df)}")

# 检查每个文件内的重复ID
print("\n=== 文件内重复检查 ===")
train_duplicates = train_df[train_df.duplicated(subset=['ID'], keep=False)]
valid_duplicates = valid_df[valid_df.duplicated(subset=['ID'], keep=False)]
test_duplicates = test_df[test_df.duplicated(subset=['ID'], keep=False)]

print(f"train.csv 内部重复ID数: {len(train_duplicates)}")
print(f"valid.csv 内部重复ID数: {len(valid_duplicates)}")
print(f"test.csv 内部重复ID数: {len(test_duplicates)}")

if len(train_duplicates) > 0:
    print(f"\ntrain.csv 重复的ID (前10个):")
    print(train_duplicates['ID'].unique()[:10])

if len(valid_duplicates) > 0:
    print(f"\nvalid.csv 重复的ID (前10个):")
    print(valid_duplicates['ID'].unique()[:10])

if len(test_duplicates) > 0:
    print(f"\ntest.csv 重复的ID (前10个):")
    print(test_duplicates['ID'].unique()[:10])

# 检查跨文件的重复
print("\n=== 跨文件重复检查 ===")
train_ids = set(train_df['ID'])
valid_ids = set(valid_df['ID'])
test_ids = set(test_df['ID'])

train_valid_overlap = train_ids & valid_ids
train_test_overlap = train_ids & test_ids
valid_test_overlap = valid_ids & test_ids

print(f"train 和 valid 之间的重复ID数: {len(train_valid_overlap)}")
print(f"train 和 test 之间的重复ID数: {len(train_test_overlap)}")
print(f"valid 和 test 之间的重复ID数: {len(valid_test_overlap)}")

if train_valid_overlap:
    print(f"\ntrain 和 valid 重复的ID (前10个): {list(train_valid_overlap)[:10]}")
if train_test_overlap:
    print(f"\ntrain 和 test 重复的ID (前10个): {list(train_test_overlap)[:10]}")
if valid_test_overlap:
    print(f"\nvalid 和 test 重复的ID (前10个): {list(valid_test_overlap)[:10]}")

# 统计唯一ID
print("\n=== 唯一ID统计 ===")
print(f"train.csv 唯一ID数: {train_df['ID'].nunique()}")
print(f"valid.csv 唯一ID数: {valid_df['ID'].nunique()}")
print(f"test.csv 唯一ID数: {test_df['ID'].nunique()}")

all_ids = pd.concat([train_df['ID'], valid_df['ID'], test_df['ID']])
print(f"三个文件合并后的唯一ID总数: {all_ids.nunique()}")
print(f"预期输出 (如果无重复): 65711")
print(f"实际获得: 64907")
print(f"丢失的序列数: {65711 - 64907}")

# 找出哪些行被过滤掉了
total_unique = all_ids.nunique()
expected_unique = len(train_df) + len(valid_df) + len(test_df)
if total_unique < expected_unique:
    print(f"\n=== 被过滤的序列分析 ===")
    print(f"总重复次数: {expected_unique - total_unique}")
    
    # 找出所有重复的ID
    duplicate_ids = all_ids[all_ids.duplicated(keep=False)].unique()
    print(f"涉及的重复ID数量: {len(duplicate_ids)}")
    print(f"\n重复的ID示例 (前20个):")
    for dup_id in sorted(duplicate_ids)[:20]:
        in_train = dup_id in train_ids
        in_valid = dup_id in valid_ids
        in_test = dup_id in test_ids
        print(f"  {dup_id}: train={in_train}, valid={in_valid}, test={in_test}")

# 检查train和valid之间的序列重复
print("\n=== train 和 valid 序列重复检查 ===")
# 过滤掉空序列
train_with_seq = train_df[train_df['seq'].notna() & (train_df['seq'] != '')].copy()
valid_with_seq = valid_df[valid_df['seq'].notna() & (valid_df['seq'] != '')].copy()

print(f"train 中有序列的行数: {len(train_with_seq)}")
print(f"valid 中有序列的行数: {len(valid_with_seq)}")

# 找出序列重复
train_seqs = set(train_with_seq['seq'])
valid_seqs = set(valid_with_seq['seq'])

seq_overlap = train_seqs & valid_seqs
print(f"\ntrain 和 valid 之间重复的序列数: {len(seq_overlap)}")

if seq_overlap:
    print(f"\n重复序列的详细信息 (前5个):")
    for i, seq in enumerate(list(seq_overlap)[:5]):
        train_rows = train_with_seq[train_with_seq['seq'] == seq]
        valid_rows = valid_with_seq[valid_with_seq['seq'] == seq]
        print(f"\n序列 {i+1} (长度={len(seq)}):")
        print(f"  在 train 中出现 {len(train_rows)} 次")
        print(f"  在 valid 中出现 {len(valid_rows)} 次")
        print(f"  train ID示例: {list(train_rows['ID'])[:3]}")
        print(f"  valid ID示例: {list(valid_rows['ID'])[:3]}")
        print(f"  序列片段: {seq[:50]}...")

# 检查train和valid之间整行内容是否完全一样
print("\n=== train 和 valid 整行内容重复检查 ===")

# 获取所有列名
all_columns = train_df.columns.tolist()
print(f"数据列: {all_columns}")

# 创建一个用于比较的字符串表示（排除ID列和split列，因为ID可能不同但内容相同，split列本身就是标识不同数据集的）
compare_columns = [col for col in all_columns if col not in ['ID', 'split']]
print(f"用于比较的列: {compare_columns}")

# 为每行创建内容指纹（将所有非ID列的值连接成字符串）
def create_content_fingerprint(row, columns):
    """创建行内容的指纹，用于比较"""
    values = []
    for col in columns:
        val = row[col]
        # 处理NaN值
        if pd.isna(val):
            values.append('__NA__')
        else:
            values.append(str(val))
    return '|||'.join(values)

train_df['__fingerprint__'] = train_df.apply(lambda row: create_content_fingerprint(row, compare_columns), axis=1)
valid_df['__fingerprint__'] = valid_df.apply(lambda row: create_content_fingerprint(row, compare_columns), axis=1)

# 找出内容重复的行
train_fingerprints = set(train_df['__fingerprint__'])
valid_fingerprints = set(valid_df['__fingerprint__'])

content_overlap = train_fingerprints & valid_fingerprints
print(f"\ntrain 和 valid 之间内容完全相同的行数: {len(content_overlap)}")

if content_overlap:
    print(f"\n内容重复的详细信息 (前5个):")
    for i, fingerprint in enumerate(list(content_overlap)[:5]):
        train_rows = train_df[train_df['__fingerprint__'] == fingerprint]
        valid_rows = valid_df[valid_df['__fingerprint__'] == fingerprint]
        print(f"\n重复内容 {i+1}:")
        print(f"  在 train 中出现 {len(train_rows)} 次")
        print(f"  在 valid 中出现 {len(valid_rows)} 次")
        print(f"  train ID示例: {list(train_rows['ID'])[:3]}")
        print(f"  valid ID示例: {list(valid_rows['ID'])[:3]}")
        # 显示第一行的实际内容
        sample_row = train_rows.iloc[0]
        print(f"  内容示例:")
        for col in compare_columns[:5]:  # 只显示前5列
            val = sample_row[col]
            if isinstance(val, str) and len(val) > 50:
                print(f"    {col}: {val[:50]}...")
            else:
                print(f"    {col}: {val}")

# 清理临时列
train_df.drop('__fingerprint__', axis=1, inplace=True)
valid_df.drop('__fingerprint__', axis=1, inplace=True)

# 统计总体重复情况
print(f"\n=== 总体重复统计 ===")
print(f"仅序列重复（ID可能不同）: {len(seq_overlap)}")
print(f"整行内容重复（ID可能不同）: {len(content_overlap)}")
print(f"ID重复: {len(train_valid_overlap)}")

# 检查train和valid之间的序列重复
print("\n=== train 和 valid 序列重复检查 ===")
# 过滤掉空序列
train_with_seq = train_df[train_df['seq'].notna() & (train_df['seq'] != '')].copy()
valid_with_seq = valid_df[valid_df['seq'].notna() & (valid_df['seq'] != '')].copy()

print(f"train 中有序列的行数: {len(train_with_seq)}")
print(f"valid 中有序列的行数: {len(valid_with_seq)}")

# 找出序列重复
train_seqs = set(train_with_seq['seq'])
valid_seqs = set(valid_with_seq['seq'])

seq_overlap = train_seqs & valid_seqs
print(f"\ntrain 和 valid 之间重复的序列数: {len(seq_overlap)}")

if seq_overlap:
    print(f"\n重复序列的详细信息 (前10个):")
    for i, seq in enumerate(list(seq_overlap)[:10]):
        train_rows = train_with_seq[train_with_seq['seq'] == seq]
        valid_rows = valid_with_seq[valid_with_seq['seq'] == seq]
        print(f"\n序列 {i+1} (长度={len(seq)}):")
        print(f"  在 train 中的ID: {list(train_rows['ID'])}")
        print(f"  在 valid 中的ID: {list(valid_rows['ID'])}")
        print(f"  序列片段: {seq[:50]}...")

# 检查train和valid之间整行内容是否完全一样
print("\n=== train 和 valid 整行内容重复检查 ===")

# 获取所有列名
all_columns = train_df.columns.tolist()
print(f"数据列: {all_columns}")

# 创建一个用于比较的字符串表示（排除ID列和split列，因为ID可能不同但内容相同，split列本身就是标识不同数据集的）
compare_columns = [col for col in all_columns if col not in ['ID', 'split']]
print(f"用于比较的列: {compare_columns}")

# 为每行创建内容指纹（将所有非ID列的值连接成字符串）
def create_content_fingerprint(row, columns):
    """创建行内容的指纹，用于比较"""
    values = []
    for col in columns:
        val = row[col]
        # 处理NaN值
        if pd.isna(val):
            values.append('__NA__')
        else:
            values.append(str(val))
    return '|||'.join(values)

train_df['__fingerprint__'] = train_df.apply(lambda row: create_content_fingerprint(row, compare_columns), axis=1)
valid_df['__fingerprint__'] = valid_df.apply(lambda row: create_content_fingerprint(row, compare_columns), axis=1)

# 找出内容重复的行
train_fingerprints = set(train_df['__fingerprint__'])
valid_fingerprints = set(valid_df['__fingerprint__'])

content_overlap = train_fingerprints & valid_fingerprints
print(f"\ntrain 和 valid 之间内容完全相同的行数: {len(content_overlap)}")

if content_overlap:
    print(f"\n内容重复的详细信息 (前10个):")
    for i, fingerprint in enumerate(list(content_overlap)[:10]):
        train_rows = train_df[train_df['__fingerprint__'] == fingerprint]
        valid_rows = valid_df[valid_df['__fingerprint__'] == fingerprint]
        print(f"\n重复内容 {i+1}:")
        print(f"  在 train 中出现 {len(train_rows)} 次，ID: {list(train_rows['ID'])}")
        print(f"  在 valid 中出现 {len(valid_rows)} 次，ID: {list(valid_rows['ID'])}")
        # 显示第一行的实际内容
        sample_row = train_rows.iloc[0]
        print(f"  内容示例:")
        for col in compare_columns[:5]:  # 只显示前5列
            val = sample_row[col]
            if isinstance(val, str) and len(val) > 50:
                print(f"    {col}: {val[:50]}...")
            else:
                print(f"    {col}: {val}")

# 清理临时列
train_df.drop('__fingerprint__', axis=1, inplace=True)
valid_df.drop('__fingerprint__', axis=1, inplace=True)

# 统计总体重复情况
print(f"\n=== 总体重复统计 ===")
print(f"仅序列重复（ID可能不同）: {len(seq_overlap)}")
print(f"整行内容重复（ID可能不同）: {len(content_overlap)}")
print(f"ID重复: {len(train_valid_overlap)}")

# 检查train和valid之间的序列重复
print("\n=== train 和 valid 序列重复检查 ===")
# 过滤掉空序列
train_with_seq = train_df[train_df['seq'].notna() & (train_df['seq'] != '')].copy()
valid_with_seq = valid_df[valid_df['seq'].notna() & (valid_df['seq'] != '')].copy()

print(f"train 中有序列的行数: {len(train_with_seq)}")
print(f"valid 中有序列的行数: {len(valid_with_seq)}")

# 找出序列重复
train_seqs = set(train_with_seq['seq'])
valid_seqs = set(valid_with_seq['seq'])

seq_overlap = train_seqs & valid_seqs
print(f"\ntrain 和 valid 之间重复的序列数: {len(seq_overlap)}")

if seq_overlap:
    print(f"\n重复序列的详细信息 (前10个):")
    for i, seq in enumerate(list(seq_overlap)[:10]):
        train_rows = train_with_seq[train_with_seq['seq'] == seq]
        valid_rows = valid_with_seq[valid_with_seq['seq'] == seq]
        print(f"\n序列 {i+1} (长度={len(seq)}):")
        print(f"  在 train 中的ID: {list(train_rows['ID'])}")
        print(f"  在 valid 中的ID: {list(valid_rows['ID'])}")
        print(f"  序列片段: {seq[:50]}...")