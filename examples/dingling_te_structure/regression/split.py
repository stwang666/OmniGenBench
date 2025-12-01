import pandas as pd
from pathlib import Path
base = Path('/home/sw1136/OmniGenBench/examples/dingling_te_structure/regression')
source = base / '9tissue_dot.txt'
df = pd.read_csv(source, sep='\t')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

total = len(df)
train_end = int(total * 0.8)
valid_end = train_end + int(total * 0.1)

splits = {
    'train.csv': df.iloc[:train_end],
    'valid.csv': df.iloc[train_end:valid_end],
    'test.csv': df.iloc[valid_end:]
}

for name, split_df in splits.items():
    split_df.to_csv(base / name, index=False)

print(f"Total samples: {total}")
print(f"Train: {len(splits['train.csv'])}, Valid: {len(splits['valid.csv'])}, Test: {len(splits['test.csv'])}")