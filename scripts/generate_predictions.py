"""
scripts/generate_predictions.py
Generates model outputs for combined test sets (radiology, ophthalmology, pathology).
Saves JSONL predictions to results/predictions_full_test.jsonl
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# import run_rag_pipeline from your backend module
from backend.app.core.rag_pipeline import run_rag_pipeline

# CONFIG - adjust these paths to your test CSVs
TEST_FILES = {
    "radiology": "data/IUXRAY/indiana_merged_cleaned_test.csv",
    "ophthalmology": "data/ODIR-5K/odir_test.csv",
    "pathology": "data/pathology/pathology_test.csv"
}

OUT_PATH = Path("results/predictions_full_test.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_test_rows():
    rows = []
    for domain, fp in TEST_FILES.items():
        if not os.path.exists(fp):
            print(f"[warn] test CSV not found: {fp} (skipping domain)")
            continue
        df = pd.read_csv(fp)
        # expect columns: filename, report_text or image, question etc. adapt heuristics
        for _, r in df.iterrows():
            # pick text and image path heuristics
            image_col = None
            for c in ["filename","image","Left-Fundus","Right-Fundus","image_path"]:
                if c in df.columns:
                    image_col = c
                    break
            text_col = None
            for c in ["report_text","report","question","Right-Diagnostic Keywords","Left-Diagnostic Keywords"]:
                if c in df.columns:
                    text_col = c
                    break
            img = None
            if image_col:
                img_val = r.get(image_col, "")
                if isinstance(img_val, str) and img_val.strip() != "":
                    # try both domain-specific image folder naming
                    candidate_paths = [
                        f"data/IUXRAY/images/images_normalized/{img_val}",
                        f"data/ODIR-5K/preprocessed_images/{img_val}",
                        f"data/pathology/train/{img_val}",
                        img_val
                    ]
                    found = None
                    for p in candidate_paths:
                        if os.path.exists(p):
                            found = p
                            break
                    img = found
            text = r.get(text_col, "") if text_col else ""
            rows.append({"domain": domain, "image": img, "text": str(text)})
    return rows

def main():
    rows = load_test_rows()
    print("Total test rows ->", len(rows))
    with OUT_PATH.open("w", encoding="utf-8") as fo:
        for r in tqdm(rows):
            question = r["text"] or "Please interpret this image."
            image = r["image"]
            try:
                out = run_rag_pipeline(question, image_path=image, domain_hint=r["domain"])
                # write minimal record
                rec = {
                    "domain": r["domain"],
                    "image": image,
                    "question": question,
                    "generated": out.get("generated", {}),
                    "retrieved": out.get("retrieved", []),
                }
            except Exception as e:
                rec = {"domain": r["domain"], "image": image, "question": question, "error": str(e)}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Predictions saved to", OUT_PATH)

if __name__ == "__main__":
    main()
