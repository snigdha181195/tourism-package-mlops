import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

def upload_dataset(local_file: str, repo_id: str, token: str | None = None) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=Path(local_file).name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Register tourism dataset"
    )
    print(f"Uploaded {local_file} to dataset repo {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/raw/tourism.csv")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    upload_dataset(args.file, args.repo_id, os.getenv("HF_TOKEN"))
