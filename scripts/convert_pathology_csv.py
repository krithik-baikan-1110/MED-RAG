# scripts/convert_pathology_csv.py
import argparse
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="Path to trainrenamed.csv or testrenamed.csv")
parser.add_argument("--image-folder", required=True, help="Folder containing pathology images (e.g. data/pathology/train)")
parser.add_argument("--out-csv", default=None, help="Output fixed CSV path (defaults to <csv>_fixed.csv)")
parser.add_argument("--sample", type=int, default=0, help="If >0, only process first N rows (for quick test)")
args = parser.parse_args()

csv_path = Path(args.csv)
img_root = Path(args.image_folder)
out_csv = Path(args.out_csv) if args.out_csv else csv_path.parent / (csv_path.stem + "_fixed.csv")

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")
if not img_root.exists():
    raise SystemExit(f"Image folder not found: {img_root}")

print("Loading CSV:", csv_path)
df = pd.read_csv(csv_path)
if args.sample and args.sample > 0:
    df = df.head(args.sample)
print("Rows to process:", len(df))
print("Columns:", list(df.columns)[:40])

# Index image folder by various useful keys
print("Indexing image folder (one pass)...")
index_by_name = defaultdict(list)   # exact filename -> path(s)
index_by_stem = defaultdict(list)   # stem (no ext) -> path(s)
for p in img_root.rglob("*"):
    if p.is_file():
        index_by_name[p.name.lower()].append(p)
        index_by_stem[p.stem.lower()].append(p)
num_files = sum(len(v) for v in index_by_name.values())
print(f"Indexed {num_files} files, {len(index_by_stem)} unique stems")

# helper to build report_text from row
def build_report_text(row):
    # priority: answer > (question + answer) > any text-like column > "No report available"
    # join with space and strip
    if "answer" in row and pd.notna(row["answer"]) and str(row["answer"]).strip() != "":
        return str(row["answer"]).strip()
    if "question" in row and pd.notna(row["question"]) and str(row["question"]).strip() != "":
        q = str(row["question"]).strip()
        a = str(row.get("answer","")).strip() if pd.notna(row.get("answer","")) else ""
        if a:
            return q + " " + a
        return q
    # try common label columns
    label_cols = [c for c in row.index if c.lower() in ("label","diagnosis","diagnoses","class","labels","target")]
    if label_cols:
        parts = []
        for c in label_cols:
            v = row.get(c)
            if pd.notna(v) and str(v).strip()!="":
                parts.append(str(v).strip())
        if parts:
            return "; ".join(parts)
    # fallback: combine all text-like columns up to 200 chars
    texts = []
    for c in row.index:
        if str(row[c]).strip() == "" or pd.isna(row[c]):
            continue
        if isinstance(row[c], (int,float)):
            continue
        texts.append(str(row[c]).strip())
    if texts:
        joined = " | ".join(texts)
        return joined[:1000]
    return "No report available"

# resolution heuristics for image id values
EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def resolve_image(raw_value):
    if pd.isna(raw_value):
        return None
    s = str(raw_value).strip()
    if s == "":
        return None
    # if value already looks like a filename with extension
    lower = s.lower()
    if lower in index_by_name:
        return index_by_name[lower][0].name
    # try with common extensions
    for ext in EXTS:
        cand = (s + ext).lower()
        if cand in index_by_name:
            return index_by_name[cand][0].name
    # try as stem match (raw may be 'image159' which is the stem)
    stem = Path(s).stem.lower()
    if stem in index_by_stem:
        # pick first match
        return index_by_stem[stem][0].name
    # if the raw value is purely numeric, try prefixing "image"
    if s.isdigit():
        for ext in EXTS:
            cand = f"image{s}{ext}"
            if cand in index_by_name:
                return index_by_name[cand][0].name
        if f"image{s}" in index_by_stem:
            return index_by_stem[f"image{s}"][0].name
    # sometimes values are like 'img_159' or 'img159', try common patterns
    variants = [f"image{stem}", f"img{stem}", f"img_{stem}", f"image_{stem}"]
    for v in variants:
        if v in index_by_stem:
            return index_by_stem[v][0].name
        for ext in EXTS:
            if (v + ext) in index_by_name:
                return index_by_name[(v+ext)][0].name
    # last attempt: try partial numeric inside name: find any file whose stem contains the digits of s
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        for st, paths in index_by_stem.items():
            if digits in st:
                return paths[0].name
    return None

# process rows
fixed_rows = []
unmatched = []
for idx, row in df.iterrows():
    # try common filename columns
    raw = None
    for c in ("filename","file","image","img","image_id","id"):
        if c in row.index:
            raw = row[c]
            break
    # if still None, pick the first column that contains 'image' in name
    if raw is None:
        for c in row.index:
            if "image" in c.lower() or "img" in c.lower():
                raw = row[c]
                break
    filename = resolve_image(raw)
    report_text = build_report_text(row)
    if filename:
        fixed_rows.append({"filename": filename, "report_text": report_text})
    else:
        unmatched.append((idx, raw, report_text))

print(f"Resolved: {len(fixed_rows)} rows; Unmatched: {len(unmatched)} rows")

# save fixed CSV
df_fixed = pd.DataFrame(fixed_rows)
df_fixed.to_csv(out_csv, index=False)
print("Wrote fixed CSV:", out_csv)

# write unmatched for manual review
if unmatched:
    unmatched_path = csv_path.parent / (csv_path.stem + "_unmatched.txt")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for t in unmatched:
            f.write(f"{t[0]}\t{t[1]}\t{t[2][:200].replace(chr(10),' ')}\n")
    print("Wrote unmatched rows to:", unmatched_path)

# also save a small sample for quick manual review
sample_path = csv_path.parent / (csv_path.stem + "_preview_sample.json")
with open(sample_path, "w", encoding="utf-8") as f:
    json.dump({"resolved_count":len(fixed_rows), "unmatched_count":len(unmatched),
               "sample_resolved": fixed_rows[:20], "sample_unmatched": unmatched[:20]}, f, indent=2)
print("Wrote preview json:", sample_path)
print("Done.")
