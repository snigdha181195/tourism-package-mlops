import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi
from src.config import TARGET, RANDOM_STATE, TEST_SIZE

DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = data.columns.str.strip()
    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns])
    data = data.drop_duplicates().reset_index(drop=True)

    # Normalize known category inconsistencies.
    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in data.columns:
        data["MaritalStatus"] = data["MaritalStatus"].replace({"Unmarried": "Single"})

    # Numeric missing-value treatment.
    for col in data.select_dtypes(include="number").columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    # Categorical missing-value treatment.
    for col in data.select_dtypes(exclude="number").columns:
        if data[col].isna().any():
            mode = data[col].mode(dropna=True)
            data[col] = data[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return data

def split_and_save(df: pd.DataFrame, train_path: str, test_path: str):
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df

def upload_processed_files(train_path: str, test_path: str, repo_id: str, token: str | None):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    for file_path in [train_path, test_path]:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"processed/{Path(file_path).name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {Path(file_path).name}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/tourism.csv")
    parser.add_argument("--train-output", default="data/processed/train.csv")
    parser.add_argument("--test-output", default="data/processed/test.csv")
    parser.add_argument("--hf-repo-id", default=None)
    args = parser.parse_args()

    source = args.input
    if source.startswith("hf://"):
        _, repo_id, filename = source.split("/", 2)
        source = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    cleaned = clean_data(pd.read_csv(source))
    train_df, test_df = split_and_save(cleaned, args.train_output, args.test_output)
    print(f"Saved train={train_df.shape}, test={test_df.shape}")

    if args.hf_repo_id:
        upload_processed_files(args.train_output, args.test_output, args.hf_repo_id, os.getenv("HF_TOKEN"))
