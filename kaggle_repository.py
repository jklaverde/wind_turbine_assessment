import kagglehub
import numpy as np
import pandas as pd


def _download_kaggle_content(repository_name: str) -> str:
    path = kagglehub.dataset_download(repository_name)
    print(path)
    return path


def _reduce_dataset(kaggle_file_path: str, file_name: str, columns_list: list[str]) -> pd.DataFrame:
    df = pd.read_csv(f"{kaggle_file_path}/{file_name}")
    df_reduced = df[columns_list]
    print(df_reduced)
    return df_reduced


def _impute_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng()
    df = df.copy()
    df["noise_level"] = rng.uniform(35, 110, size=len(df))
    df["humidity"]    = rng.uniform(10, 100, size=len(df))
    return df


def _apply_failure_labels(df: pd.DataFrame) -> pd.DataFrame:
    def label_failure(row) -> int:
        score = 0
        if row["temperature"] > 90: score += 3
        if row["humidity"]    > 85: score += 2
        if row["noise_level"] > 75: score += 2
        return 1 if score >= 3 else 0

    df = df.copy()
    df["failure"] = df.apply(label_failure, axis=1)
    return df


if __name__ == "__main__":
    kaggle_file_path = _download_kaggle_content(repository_name="mukund23/hackerearth-machine-learning-challenge")

    reduced_dataset = _reduce_dataset(kaggle_file_path=kaggle_file_path,
                                      file_name="train.csv",
                                      columns_list=["generator_temperature(°C)"])

    reduced_dataset = reduced_dataset.rename(columns={"generator_temperature(°C)": "temperature"})

    enriched_dataset = _impute_sensor_columns(reduced_dataset)
    labelled_dataset = _apply_failure_labels(enriched_dataset)

    print(labelled_dataset)
