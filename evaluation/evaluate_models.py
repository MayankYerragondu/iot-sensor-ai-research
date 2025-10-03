# evaluation/evaluate_models.py
import pandas as pd
import numpy as np
import joblib
import boto3
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BUCKET_NAME = "iot-glue-bucket-multi-model"
OUTPUT_DIR = "evaluation/results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

s3 = boto3.client("s3")

def evaluate_model(model, scaler, X_test, y_true):
    """Return metrics dict for anomaly detection model."""
    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)

    # For IsolationForest / OCSVM, predictions: -1=anomaly, 1=normal
    y_pred = np.where(preds == -1, 1, 0)  # 1=anomaly, 0=normal

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_alarm_rate = fp / (fp + tn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }

def main():
    # Example: local test data (replace with S3 fetch or Glue output)
    test_df = pd.read_csv("sample_pir_test.csv")  
    features = ["hour", "minute", "day_of_week", "time_diff"]
    X_test = test_df[features]
    y_true = test_df["label"]  # 1=anomaly, 0=normal

    # Example model names (expand as needed)
    model_names = ["if", "gmm", "ocsvm"]

    all_metrics = []
    for model_name in model_names:
        model_key = f"model/pir_alarm/device123_{model_name}.joblib"
        scaler_key = f"model/pir_alarm/device123_scaler.joblib"

        # Download model + scaler from S3
        s3.download_file(BUCKET_NAME, model_key, f"/tmp/{model_name}.joblib")
        s3.download_file(BUCKET_NAME, scaler_key, "/tmp/scaler.joblib")

        model = joblib.load(f"/tmp/{model_name}.joblib")
        scaler = joblib.load("/tmp/scaler.joblib")

        metrics = evaluate_model(model, scaler, X_test, y_true)
        metrics["model"] = model_name
        all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)
    out_path = os.path.join(OUTPUT_DIR, "pir_metrics.csv")
    df_metrics.to_csv(out_path, index=False)
    print(f"✅ Saved evaluation metrics to {out_path}")

if __name__ == "__main__":
    main()
