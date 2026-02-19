# scripts/map_pathology_ids_to_files.py
import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Input pathology CSV (with 'image' column)")
parser.add_argument("--image-folder", required=True, help="Folder containing image files")
parser.add_argument("--out-csv", default=None, help="Output fixed CSV path")
args = parser.parse_args()

csv_path = Path(args.csv)
img_root = Path(args.image_folder)
out_csv = Path(args.out_csv) if args.out_csv else csv_path.parent / (csv_path.stem + "_fixed.csv")

print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path)
if "image" not in df.columns:
    raise SystemExit("CSV must have an 'image' column. Found: " + ", ".join(df.columns))

# Index files by stem -> list(paths)
print("Indexing files under:", img_root)
stem_index = defaultdict(list)
for p in img_root.rglob("*"):
    if p.is_file():
        stem = p.stem  # filename without extension
        stem_index[stem.lower()].append(p)

print("Indexed stems:", len(stem_index))

matched = []
unmatched = []
duplicates = {}

for idx, row in df.iterrows():
    raw = str(row["image"]).strip()
    if raw == "" or raw.lower() == "nan":
        unmatched.append((idx, raw, "empty"))
        continue
    # raw may already be a full filename; try direct match first
    candidate = Path(raw).name
    cand_path = img_root / candidate
    if cand_path.exists():
        matched.append((idx, raw, cand_path.name))
        continue

    # try matching by stem (raw may be 'image159' -> look for stem 'image159')
    stem_key = candidate.split(".")[0].lower()
    hits = stem_index.get(stem_key)
    if hits:
        # if more than one hit, store duplicates info
        if len(hits) > 1:
            duplicates[stem_key] = [str(p.name) for p in hits]
        chosen = hits[0]  # pick first
        matched.append((idx, raw, chosen.name))
    else:
        unmatched.append((idx, raw, "not_found"))

print(f"Total rows: {len(df)}  matched: {len(matched)}  unmatched: {len(unmatched)}  duplicates: {len(duplicates)}")

# Build fixed dataframe: add 'filename' column (basename with extension) for matched, empty otherwise
df_fixed = df.copy()
df_fixed["filename"] = ""
for idx, raw, fname in matched:
    df_fixed.at[idx, "filename"] = fname

# save fixed CSV
df_fixed.to_csv(out_csv, index=False)
print("Saved fixed CSV:", out_csv)

# write unmatched list for inspection
if unmatched:
    unmatched_path = csv_path.parent / (csv_path.stem + "_unmatched.txt")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for t in unmatched:
            f.write(f"{t[0]}\t{t[1]}\t{t[2]}\n")
    print("Wrote unmatched rows to:", unmatched_path)

# write duplicate mapping for inspection
if duplicates:
    dup_path = csv_path.parent / (csv_path.stem + "_duplicates.json")
    import json
    with open(dup_path, "w", encoding="utf-8") as f:
        json.dump(duplicates, f, indent=2)
    print("Wrote duplicate stems to:", dup_path)

print("Done.")
