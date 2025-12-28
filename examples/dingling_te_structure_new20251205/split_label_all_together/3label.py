python - <<'PY'
import csv, math
from pathlib import Path

in_path=Path("/home/sw1136/OmniGenBench/examples/dingling_te_structure_new20251205/9tissue_structure_te_hc_deseq2_tp_split.csv")
out_path=in_path.with_name("9tissue_structure_te_hc_deseq2_tp_split_labeled.csv")
vals=[]
tissues=set()
rows=[]
with in_path.open() as f:
    reader=csv.DictReader(f)
    for row in reader:
        te=float(row["TE"])
        vals.append(te)
        tissues.add(row["TISSUE"])
        rows.append(row)

n=len(vals)
vals_sorted=sorted(vals)

def interp_quantile(p: float):
    if not vals_sorted:
        return float("nan")
    pos=p*(n-1)
    lo=math.floor(pos)
    hi=math.ceil(pos)
    if lo==hi:
        return vals_sorted[int(pos)]
    frac=pos-lo
    return vals_sorted[lo]*(1-frac)+vals_sorted[hi]*frac

p33=interp_quantile(1/3)
p66=interp_quantile(2/3)


def label_value(v: float):
    if v < p33:
        return 0
    elif v < p66:
        return 1
    else:
        return 2

counts=[0,0,0]
for row, te in zip(rows, vals):
    lab=label_value(te)
    counts[lab]+=1

for row in rows:
    row["label"]=str(label_value(float(row["TE"])))

fieldnames=list(rows[0].keys())
if "label" not in fieldnames:
    fieldnames.append("label")

with out_path.open("w", newline="") as f:
    writer=csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

stats={
    "count": n,
    "min": min(vals),
    "max": max(vals),
    "mean": sum(vals)/n,
    "median": interp_quantile(0.5),
    "p33": p33,
    "p66": p66,
}

print("STAT", stats)
print("COUNTS", counts)
print("TISSUE", sorted(tissues))
print("OUT", out_path)
PY