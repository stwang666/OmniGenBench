import pandas as pd
from pathlib import Path
from collections import Counter

base_dir = Path("/home/sw1136/OmniGenBench/examples/dingling_te_structure/triclass")
input_path = base_dir / "9tissue_dot.txt"
output_files = {split: base_dir / f"{split}.csv" for split in ("train", "valid", "test")}

df = pd.read_csv(input_path, sep="\t")
df["label"] = df["label"].fillna(2)
df.loc[df["label"].astype(str).str.strip() == "", "label"] = 2
df["label"] = df["label"].astype(int)

tr_parts, va_parts, te_parts = [], [], []
seed = 42
for label, group in df.groupby("label"):
    group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(group)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    if n_train == 0 and n > 0:
        n_train = 1
    if n_valid == 0 and n - n_train > 1:
        n_valid = 1
    n_test = n - n_train - n_valid
    if n_test < 0:
        n_test = 0
        if n_valid > 0:
            n_valid -= 1
        else:
            n_train -= 1
    tr_parts.append(group.iloc[:n_train])
    va_parts.append(group.iloc[n_train:n_train + n_valid])
    te_parts.append(group.iloc[n_train + n_valid:])

train_df = pd.concat(tr_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
valid_df = pd.concat(va_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
test_df = pd.concat(te_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

for split_df, split in ((train_df, "train"), (valid_df, "valid"), (test_df, "test")):
    split_df.to_csv(output_files[split], index=False)
    print(split, len(split_df), Counter(split_df["label"]))