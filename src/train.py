import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from huggingface_hub import HfApi

from src.config import TARGET, RANDOM_STATE

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True)
    }

def append_experiment(path: str, model_name: str, params: dict, metrics: dict):
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "best_params": json.dumps(params, sort_keys=True),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"]
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)

def train(train_path: str, test_path: str, model_path: str, metrics_path: str, experiments_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    prep = build_preprocessor(X_train)

    candidates = {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1),
            {
                "model__n_estimators": [80, 120],
                "model__max_depth": [8, 12],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", "log2"]
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [80, 120],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0]
            }
        ),
        
    }

    leaderboard = []
    best_model = None
    best_score = -1
    best_name = None
    best_params = None

    for name, (estimator, params) in candidates.items():
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        search = RandomizedSearchCV(
            pipe, params, n_iter=2,
            scoring="roc_auc", cv=2, random_state=RANDOM_STATE,
            n_jobs=1, verbose=0
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        append_experiment(experiments_path, name, search.best_params_, metrics)
        leaderboard.append({"model": name, **metrics, "best_params": search.best_params_})

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = search.best_estimator_
            best_name = name
            best_params = search.best_params_

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_model": best_name,
        "best_params": best_params,
        "test_metrics": evaluate(best_model, X_test, y_test),
        "leaderboard": leaderboard,
        "feature_columns": X_train.columns.tolist()
    }
    Path(metrics_path).write_text(json.dumps(payload, indent=2))
    return payload

def upload_model(model_path: str, metrics_path: str, repo_id: str, token: str | None):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    for path in [model_path, metrics_path]:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=Path(path).name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Register {Path(path).name}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--model-output", default="models/tourism_purchase_pipeline.joblib")
    parser.add_argument("--metrics-output", default="metrics/test_metrics.json")
    parser.add_argument("--experiments-output", default="experiments/experiments.csv")
    parser.add_argument("--hf-model-repo", default=None)
    args = parser.parse_args()

    result = train(args.train, args.test, args.model_output, args.metrics_output, args.experiments_output)
    print(json.dumps(result, indent=2))
    if args.hf_model_repo:
        upload_model(args.model_output, args.metrics_output, args.hf_model_repo, os.getenv("HF_TOKEN"))
