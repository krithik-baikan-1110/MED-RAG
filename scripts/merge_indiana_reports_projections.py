import pandas as pd
import os

# Define input file paths
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "IUXRAY")
REPORTS_PATH = os.path.join(BASE_DIR, "indiana_reports.csv")
PROJECTIONS_PATH = os.path.join(BASE_DIR, "indiana_projections.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "indiana_merged_cleaned.csv")

# Load the CSV files
reports_df = pd.read_csv(REPORTS_PATH)
projections_df = pd.read_csv(PROJECTIONS_PATH)

print("Reports:", reports_df.shape, "| Projections:", projections_df.shape)

# Preview column names
print("Report Columns:", reports_df.columns.tolist())
print("Projection Columns:", projections_df.columns.tolist())

# Common key for merge
# Usually both have a field like `uid`, `study_id`, or `image_id`
# We’ll try automatic matching and fallback to manual mapping
common_cols = set(reports_df.columns).intersection(set(projections_df.columns))
if not common_cols:
    raise ValueError("❌ No common columns found to merge — please check the key (uid, study_id, etc.)")
else:
    merge_key = list(common_cols)[0]
    print(f"✅ Merging on key: {merge_key}")

# Merge the datasets
merged = pd.merge(projections_df, reports_df, on=merge_key, how="inner")

# Clean up report text
def clean_report(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    return text

text_fields = [col for col in ["impression", "findings", "comparison", "indication", "report"] if col in merged.columns]
print(f"Text fields used for report construction: {text_fields}")

if text_fields:
    merged["report_text"] = merged.apply(
        lambda row: " ".join(
            [
                clean_report(row[col])
                for col in text_fields
                if isinstance(row[col], str) and clean_report(row[col])
            ]
        ),
        axis=1,
    )
else:
    merged["report_text"] = ""

print(f"Rows after merge: {len(merged)}")
non_empty_text = merged["report_text"].str.len().gt(0).sum()
print(f"Rows with non-empty report_text: {non_empty_text}")
length_summary = merged["report_text"].str.len().describe()
print("report_text length summary:")
print(length_summary)

# Drop duplicates & empty rows
merged.drop_duplicates(subset=[merge_key, "report_text"], inplace=True)
lengths = merged["report_text"].str.len()
merged = merged[lengths > 30]

# Keep only essential columns
keep_cols = [merge_key, "image_path" if "image_path" in merged.columns else "filename", "report_text"]
merged = merged[[col for col in keep_cols if col in merged.columns]]

# Save merged file
merged.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Merged dataset saved as: {OUTPUT_PATH}")
print(f"Total records: {len(merged)}")
