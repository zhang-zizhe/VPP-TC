import pandas as pd
import re
from collections import Counter

CSV = r"C:\VPP-TC\openarm_test\output\openarm_dual_inter_arm_3M.csv"

# peek columns
cols = pd.read_csv(CSV, nrows=1).columns.tolist()
off_cols = [c for c in cols if "offend" in c.lower()]
print("offender columns:", off_cols)
if not off_cols:
    print("NO offender columns -- columns are:")
    print(cols)
    raise SystemExit

a_col, b_col = off_cols[0], off_cols[1]

def simplify(name):
    if not isinstance(name, str):
        return "?"
    # arm side
    if name.startswith("openarm_left_"):
        side = "L"
        rest = name[len("openarm_left_"):]
    elif name.startswith("openarm_right_"):
        side = "R"
        rest = name[len("openarm_right_"):]
    else:
        return name  # torso/base etc
    if "finger" in rest:
        part = "finger"
    elif rest.startswith("link"):
        part = rest  # keep link number
    else:
        part = rest
    return f"{side}:{part}"

usecols = [a_col, b_col, "min_dist"] if "min_dist" in cols else [a_col, b_col]
pair_ctr = Counter()
collide_ctr = Counter()
n = 0
for chunk in pd.read_csv(CSV, usecols=usecols, chunksize=200000):
    for _, row in chunk.iterrows():
        sa, sb = simplify(row[a_col]), simplify(row[b_col])
        key = tuple(sorted([sa, sb]))
        pair_ctr[key] += 1
        if "min_dist" in usecols and row["min_dist"] < 0:
            collide_ctr[key] += 1
    n += len(chunk)
    print(f"  processed {n:,}", flush=True)

print("\n=== TOP 30 offender pairs (all rows) ===")
tot = sum(pair_ctr.values())
for k, v in pair_ctr.most_common(30):
    print(f"  {v/tot*100:6.2f}%  {v:>9,}   {k[0]:<14} <-> {k[1]:<14}")

print(f"\nTotal rows with offender info: {tot:,}")

# Specifically: midlink-vs-opposite-finger
def is_midlink_finger(k):
    a, b = k
    def midlink(x): return x in ("L:link4","R:link4","L:link5","R:link5","L:link3","R:link3")
    def finger(x): return x.endswith(":finger")
    return (midlink(a) and finger(b)) or (midlink(b) and finger(a))

mlf = sum(v for k,v in pair_ctr.items() if is_midlink_finger(k))
print(f"\nmid-link(3/4/5) vs finger pairs: {mlf:,} = {mlf/tot*100:.3f}%")

def has_finger(k): return k[0].endswith(":finger") or k[1].endswith(":finger")
fingers = sum(v for k,v in pair_ctr.items() if has_finger(k))
print(f"any pair involving a finger: {fingers:,} = {fingers/tot*100:.3f}%")
