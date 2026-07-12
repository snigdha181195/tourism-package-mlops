import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import TARGET

model = joblib.load("models/tourism_purchase_pipeline.joblib")
test = pd.read_csv("data/processed/test.csv")
X, y = test.drop(columns=[TARGET]), test[TARGET]
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y, prob), 4))
print("Confusion matrix:")
print(confusion_matrix(y, pred))
print(classification_report(y, pred))
