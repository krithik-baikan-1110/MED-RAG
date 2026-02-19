# scripts/split_odir.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import json

DATA_DIR = "data/ODIR-5K"
CSV_PATH = os.path.join(DATA_DIR, "full_df.csv")   # adjust if your CSV name differs
IMAGE_FOLDER = os.path.join(DATA_DIR, "images")
RANDOM_STATE = 42

def infer_patient_id_col(df):
    """Try to find an existing patient id column, else return None."""
    candidates = ["patient_id", "patient", "id", "case_id", "image_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def derive_patient_from_filename(fname):
    """Derive a stable patient id from filename like '26_left.jpg' or '026_left.jpeg'."""
    # strip extension:
    base = os.path.splitext(os.path.basename(fname))[0]
    # common separators: '_', '-', ' '
    for sep in ['_', '-', ' ']:
        if sep in base:
            return base.split(sep)[0]
    # fallback: entire base
    return base

def infer_label_column(df):
    """Return a (label_type, column) tuple.
       label_type: 'single' (single col with text labels),
                   'onehot' (multiple binary columns),
                   'unknown' (no obvious labels)
       column: name or list of names depending on label_type
    """
    # common single-label columns
    single_cands = ["diagnosis", "label", "labels", "disease", "finding", "diagnose", "diagnoses"]
    for c in single_cands:
        if c in df.columns:
            return "single", c

    # find one-hot style columns: many columns with 0/1 values and short names like 'N','D','G','C','A','H','M','O'
    # We'll look for columns with only values in {0,1} or {0,1,nan}
    onehot_cols = []
    for c in df.columns:
        # skip filename-like columns
        if c.lower() in ("filename", "file", "image", "image_path", "path"):
            continue
        vals = df[c].dropna().unique()
        # accept if all values are 0/1 or boolean
        if len(vals) > 0 and all(str(v) in ("0","1","0.0","1.0","True","False","true","false") for v in vals):
            onehot_cols.append(c)
    if len(onehot_cols) >= 2:
        return "onehot", onehot_cols

    # fallback: look for columns with a small set of strings (like 'N','D','G', etc.)
    for c in df.columns:
        if c.lower() in ("eye", "side"):
            continue
        if df[c].dtype == object:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 30:
                # candidate text label column
                return "single", c

    return "unknown", None

def build_report_text_for_row(row, label_info):
    ltype, lcol = label_info
    if ltype == "single":
        val = row.get(lcol, "")
        return str(val) if pd.notna(val) else ""
    elif ltype == "onehot":
        positive = []
        for c in lcol:
            try:
                if int(row.get(c, 0)) == 1:
                    positive.append(c)
            except Exception:
                pass
        if positive:
            return ", ".join(positive)
        return ""
    else:
        # try common columns that might contain keywords
        for alt in ["keyword","keywords","report","notes","finding","description"]:
            if alt in row.index and pd.notna(row[alt]):
                return str(row[alt])
        # last fallback: empty
        return ""

def main(csv_path=CSV_PATH, image_folder=IMAGE_FOLDER, output_dir=DATA_DIR, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns.tolist()[:20])

    # detect filename column (common names)
    fname_cols = ["filename", "file", "image", "image_path", "img"]
    fname_col = None
    for c in fname_cols:
        if c in df.columns:
            fname_col = c
            break
    # if not, try to detect a column containing strings that look like filenames
    if not fname_col:
        for c in df.columns:
            if df[c].dtype == object and df[c].str.contains(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", case=False, na=False).any():
                fname_col = c
                break
    if not fname_col:
        raise RuntimeError("Could not detect filename column in CSV. Please add a 'filename' column with image filenames.")

    # patient id
    pid_col = infer_patient_id_col(df)
    if pid_col:
        print("Using existing patient id column:", pid_col)
    else:
        print("No patient_id column detected. Deriving patient id from filenames.")
        df["_derived_pid"] = df[fname_col].apply(derive_patient_from_filename)
        pid_col = "_derived_pid"

    # label info
    label_info = infer_label_column(df)
    print("Inferred label info:", label_info)

    # construct a canonical dataframe with columns we need
    df2 = df.copy()
    # ensure filenames only (not full paths)
    df2["_filename"] = df2[fname_col].apply(lambda x: os.path.basename(str(x)))
    df2["_patient_id"] = df2[pid_col].astype(str)

    # build report_text / label
    df2["_report_text"] = df2.apply(lambda r: build_report_text_for_row(r, label_info), axis=1)

    # check left/right availability by patient
    # define side detection: check for "_left" or "_right" in filename OR a side column.
    def detect_side(fname, row):
        # check explicit columns
        for cand in ["side","eye","left_right","laterality","LR"]:
            if cand in row.index and pd.notna(row[cand]):
                v = str(row[cand]).lower()
                if "left" in v or "l" == v:
                    return "left"
                if "right" in v or "r" == v:
                    return "right"
        # fallback to filename tokens
        b = os.path.basename(fname).lower()
        if "_left" in b or "left" in b:
            return "left"
        if "_right" in b or "right" in b:
            return "right"
        # last fallback: unknown
        return "unknown"

    df2["_side"] = df2.apply(lambda r: detect_side(r["_filename"], r), axis=1)

    # Group by patient
    patients = df2.groupby("_patient_id")
    patient_records = []
    for pid, group in patients:
        rec = {"patient_id": pid}
        # left and right if exist
        lefts = group[group["_side"] == "left"]["_filename"].tolist()
        rights = group[group["_side"] == "right"]["_filename"].tolist()
        # if neither identified by side, try to split by first occurrence names if two images
        if not lefts and not rights:
            files = group["_filename"].tolist()
            if len(files) == 2:
                lefts = [files[0]]
                rights = [files[1]]
        rec["left_images"] = lefts
        rec["right_images"] = rights
        # use most common report_text among this patient's rows
        texts = group["_report_text"].astype(str)
        texts = texts[texts != ""]
        rec["report_text"] = texts.mode().iloc[0] if not texts.empty else ""
        # keep full rows for ingestion choices
        rec["rows"] = group.to_dict(orient="records")
        patient_records.append(rec)

    print(f"Total patients inferred: {len(patient_records)}")

    # Build patient-level DataFrame for splitting (this ensures both eyes stay in same split)
    p_df = pd.DataFrame(patient_records)

    # Shuffle and split patients
    train_p, temp_p = train_test_split(p_df, test_size=(1 - train_ratio), random_state=RANDOM_STATE, shuffle=True)
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_p, test_p = train_test_split(temp_p, test_size=(1 - relative_val), random_state=RANDOM_STATE, shuffle=True)

    print("Patients split counts -> train:", len(train_p), "val:", len(val_p), "test:", len(test_p))

    # Expand back to image-level rows for each split
    def expand_patient_records(pat_df):
        rows = []
        for _, r in pat_df.iterrows():
            for row in r["rows"]:
                # row is original csv row with original columns
                out = dict(row)
                out["patient_id"] = r["patient_id"]
                out["report_text_patient"] = r["report_text"]
                # normalized filename
                out["filename"] = os.path.basename(out.get(fname_col) if fname_col in out else out.get("_filename"))
                rows.append(out)
        return pd.DataFrame(rows)

    train_img_df = expand_patient_records(train_p)
    val_img_df = expand_patient_records(val_p)
    test_img_df = expand_patient_records(test_p)

    # Save CSVs
    out_train = os.path.join(output_dir, "odir_train.csv")
    out_val   = os.path.join(output_dir, "odir_val.csv")
    out_test  = os.path.join(output_dir, "odir_test.csv")
    train_img_df.to_csv(out_train, index=False)
    val_img_df.to_csv(out_val, index=False)
    test_img_df.to_csv(out_test, index=False)

    # Also create paired CSVs (one row per patient) for possible pair ingestion/eval
    def to_pairs_df(pat_df):
        pairs = []
        for _, r in pat_df.iterrows():
            pairs.append({
                "patient_id": r["patient_id"],
                "left_images": json.dumps(r["left_images"]),
                "right_images": json.dumps(r["right_images"]),
                "report_text": r["report_text"]
            })
        return pd.DataFrame(pairs)

    to_pairs_df(train_p).to_csv(os.path.join(output_dir, "odir_pairs_train.csv"), index=False)
    to_pairs_df(val_p).to_csv(os.path.join(output_dir, "odir_pairs_val.csv"), index=False)
    to_pairs_df(test_p).to_csv(os.path.join(output_dir, "odir_pairs_test.csv"), index=False)

    print("Saved files:")
    print(" -", out_train)
    print(" -", out_val)
    print(" -", out_test)
    print(" - paired CSVs: odir_pairs_*.csv")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH, help="Path to ODIR CSV (full_df.csv)")
    parser.add_argument("--images", default=IMAGE_FOLDER, help="Images folder")
    parser.add_argument("--out", default=DATA_DIR, help="Output dir")
    args = parser.parse_args()
    main(csv_path=args.csv, image_folder=args.images, output_dir=args.out)
