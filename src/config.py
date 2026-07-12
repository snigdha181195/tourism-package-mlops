from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "tourism.csv"
TRAIN_DATA = ROOT / "data" / "processed" / "train.csv"
TEST_DATA = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "tourism_purchase_pipeline.joblib"
METRICS_PATH = ROOT / "metrics" / "test_metrics.json"
EXPERIMENTS_PATH = ROOT / "experiments" / "experiments.csv"

TARGET = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.20

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "YOUR_HF_USERNAME/tourism-package-data")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "YOUR_HF_USERNAME/tourism-package-model")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "YOUR_HF_USERNAME/tourism-package-predictor")
