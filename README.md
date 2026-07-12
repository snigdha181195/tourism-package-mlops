# Tourism Package Prediction — End-to-End MLOps Project

This repository predicts whether a customer will purchase a Wellness Tourism Package and implements the complete lifecycle required by the rubric:

- data registration in a Hugging Face dataset repository;
- cleaning, train/test splitting and processed-data publication;
- model experimentation, tuning, evaluation and model registration;
- Streamlit deployment through a Docker-based Hugging Face Space;
- GitHub Actions CI/CD on every push to `main`.

## Local execution

```bash
pip install -r requirements.txt
python -m src.data_preparation
python -m src.train
cp models/tourism_purchase_pipeline.joblib deployment/
streamlit run deployment/app.py
```

## Required GitHub secrets

Create these repository secrets:

- `HF_TOKEN`
- `HF_DATASET_REPO`, for example `username/tourism-package-data`
- `HF_MODEL_REPO`, for example `username/tourism-package-model`
- `HF_SPACE_REPO`, for example `username/tourism-package-predictor`

## Submission links

Replace these placeholders after publishing:

- GitHub repository: `PASTE_PUBLIC_GITHUB_REPOSITORY_LINK`
- Hugging Face Space: `PASTE_PUBLIC_HUGGING_FACE_SPACE_LINK`
- Hugging Face dataset: `PASTE_PUBLIC_HUGGING_FACE_DATASET_LINK`
- Hugging Face model: `PASTE_PUBLIC_HUGGING_FACE_MODEL_LINK`

Do not commit access tokens.
