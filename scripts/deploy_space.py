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

for filename in ["app.py", "requirements.txt", "Dockerfile", "tourism_purchase_pipeline.joblib"]:
    path = Path("deployment") / filename
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy {filename}"
    )
print(f"Deployment pushed to https://huggingface.co/spaces/{repo_id}")
