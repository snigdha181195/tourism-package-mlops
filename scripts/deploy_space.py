import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ["HF_SPACE_REPO"]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False
)

files_to_upload = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
    "Dockerfile": "Dockerfile",
    "tourism_purchase_pipeline.joblib": "tourism_purchase_pipeline.joblib",
}

for local_filename, space_filename in files_to_upload.items():
    path = Path("deployment") / local_filename

    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=space_filename,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy {space_filename}",
    )
print(f"Deployment pushed to https://huggingface.co/spaces/{repo_id}")
