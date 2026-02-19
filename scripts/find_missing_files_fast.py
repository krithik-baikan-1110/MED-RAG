# scripts/find_missing_files_fast.py
import argparse
from pathlib import Path
import pandas as pd
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--image-folder", required=True)
parser.add_argument("--filename-col", default=None)
parser.add_argument("--n-sample", type=int, default=20)
args = parser.parse_args()

csv_path = Path(args.csv)
img_root = Path(args.image_folder)

print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path)
print("Rows:", len(df), "Columns:", list(df.columns)[:30])

# detect filename column if not provided
fname_col = args.filename_col
cands = ["filename","file","image","image_path","filepath","Left-Fundus","Right-Fundus"]
if fname_col is None:
    for c in cands:
        if c in df.columns:
            fname_col = c
            break
if fname_col is None:
    raise SystemExit("Could not detect filename column. Provide --filename-col")

print("Using filename column:", fname_col)

# Build index of basenames in image folder (single pass)
print("Indexing files under:", img_root, "(this may take a few seconds)...")
index = {}
for p in img_root.rglob("*"):
    if p.is_file():
        index.setdefault(p.name.lower(), []).append(str(p))

print("Indexed files:", sum(len(v) for v in index.values()))

missing = []
found = 0
empty = 0
for idx, row in df.iterrows():
    raw = row.get(fname_col, "")
    if pd.isna(raw) or str(raw).strip() == "":
        empty += 1
        missing.append((idx, raw, "empty"))
        continue
    basename = Path(str(raw)).name.strip()
    matches = index.get(basename.lower())
    if matches:
        found += 1
    else:
        missing.append((idx, raw, "not_found"))

print(f"\nSummary: total={len(df)} found={found} empty={empty} missing={len(missing)}")

if missing:
    print("\nSample missing rows (index, raw_value, reason):")
    for t in missing[: args.n_sample]:
        print(t)

    print("\nTop 20 missing basenames:")
    cnt = Counter([Path(str(x[1])).name for x in missing if str(x[1]).strip() != ""])
    for b,c in cnt.most_common(20):
        print(f"{b} -> {c}")

print("\nSuggestions:")
print(" - If many 'empty', the CSV image column is empty or column name is wrong.")
print(" - If names not found, check extension mismatch (jpg vs jpeg vs png) or subfolder usage.")
print(" - To create a fixed CSV with 'filename' column, run the small helper scripts I provided earlier.")
