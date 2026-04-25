"""
Orchestrates Phase 1 (data wrangling) and Phase 2 (model training).
Executed once at Docker build time: RUN python pipeline.py
"""
import os

from sklearn.model_selection import train_test_split

from src.data.cleaning import clean, select_and_rename
from src.data.kaggle_repository import download_dataset, load_file
from src.data.labeling import apply_failure_labels, impute_sensor_columns
from src.training.train import train_and_save

OUTPUT_DIR = "data/cleaned"


def wrangle() -> None:
    print("=== Phase 1: Data Wrangling ===")

    print("Downloading dataset...")
    dataset_path = download_dataset()

    print("Loading train.csv...")
    df_raw = load_file(dataset_path, "train.csv")

    print("Selecting and renaming columns...")
    df = select_and_rename(df_raw)

    print("Cleaning...")
    df = clean(df)

    print("Imputing sensor columns (seed=42)...")
    df = impute_sensor_columns(df)

    print("Applying failure labels...")
    df = apply_failure_labels(df)

    anomaly_rate = df["failure"].mean() * 100
    print(f"Anomaly rate: {anomaly_rate:.1f}%  ({df['failure'].sum()} / {len(df)} rows)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stratified 70 / 20 / 10 split — preserves the failure ratio in every split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["failure"]
    )
    test_df, sample_df = train_test_split(
        temp_df, test_size=1 / 3, random_state=42, stratify=temp_df["failure"]
    )

    train_df.to_csv(f"{OUTPUT_DIR}/train.csv",  index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv",    index=False)
    sample_df.to_csv(f"{OUTPUT_DIR}/sample.csv", index=False)

    print(f"Saved  train={len(train_df)}  test={len(test_df)}  sample={len(sample_df)} rows")


def train() -> None:
    print("\n=== Phase 2: Model Training ===")
    train_and_save()


if __name__ == "__main__":
    wrangle()
    train()
    print("\nPipeline complete.")
