import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(csv_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1."

    df = pd.read_csv(csv_path)

    # STEP 1: Train vs temp split
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=random_state, shuffle=True
    )

    # STEP 2: Validation vs test split
    val_relative = val_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_relative), random_state=random_state, shuffle=True
    )

    print("Total:", len(df))
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # save splits
    base = csv_path.replace(".csv", "")
    train_df.to_csv(f"{base}_train.csv", index=False)
    val_df.to_csv(f"{base}_val.csv", index=False)
    test_df.to_csv(f"{base}_test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    split_dataset("data/IUXRAY/indiana_merged_cleaned.csv")
    # OR ODIR
    # split_dataset("data/ODIR-5K/full_df.csv")
