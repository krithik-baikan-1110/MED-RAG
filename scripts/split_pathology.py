import pandas as pd
from sklearn.model_selection import train_test_split
import os

PATHOLOGY_DIR = "data/pathology"
TRAIN_CSV = os.path.join(PATHOLOGY_DIR, "trainrenamed.csv")
TEST_CSV = os.path.join(PATHOLOGY_DIR, "testrenamed.csv")

def split_pathology():
    print("🔍 Loading pathology CSV files...")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Combine train + test (because we want a new 80/10/10 split)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"📦 Total pathology samples: {len(full_df)}")

    # Step 1: Train + temp
    train_split, temp_split = train_test_split(
        full_df, test_size=0.20, random_state=42, shuffle=True
    )

    # Step 2: validation + test
    val_split, test_split = train_test_split(
        temp_split, test_size=0.50, random_state=42, shuffle=True
    )

    print("📊 Final splits:")
    print("  Train:", len(train_split))
    print("  Val:", len(val_split))
    print("  Test:", len(test_split))

    # Save new CSVs
    train_split.to_csv(os.path.join(PATHOLOGY_DIR, "pathology_train.csv"), index=False)
    val_split.to_csv(os.path.join(PATHOLOGY_DIR, "pathology_val.csv"), index=False)
    test_split.to_csv(os.path.join(PATHOLOGY_DIR, "pathology_test.csv"), index=False)

    print("✅ Pathology data successfully split & saved!")

if __name__ == "__main__":
    split_pathology()
