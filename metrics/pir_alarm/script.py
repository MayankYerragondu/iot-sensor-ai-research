import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def run_metrics(csv_path="70:2c:1f:32:1a:e4.csv"):
    # --- Load timestamp-only data ---
    df = pd.read_csv(csv_path)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    # Feature engineering (matches your pipeline)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    features = ["hour", "minute", "day_of_week", "time_diff"]


    # -----------------------------------------------------
    # 1. Assign ground-truth labels: 95% normal, 5% anomaly
    # -----------------------------------------------------
    n = len(df)
    df["label"] = 0  # normal
    anomaly_count = int(n * 0.05)

    # mark last 5% as anomalies
    df.loc[df.index[-anomaly_count:], "label"] = 1


    # -----------------------------------------------------
    # 2. Train/test split (optional, same as your structure)
    # -----------------------------------------------------
    try:
        train_df, _ = train_test_split(df, test_size=0.3, random_state=42)
    except ValueError:
        train_df = df.copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_all = scaler.transform(df[features])

    # -----------------------------------------------------
    # 3. Train Isolation Forest
    # -----------------------------------------------------
    model = IsolationForest(
        n_estimators=200,
        max_samples=min(512, len(X_train)),
        contamination=0.05,  # 5% expected anomaly rate
        random_state=42
    )
    model.fit(X_train)

    # Predict
    df["pred"] = np.where(model.predict(X_all) == -1, 1, 0)

    # -----------------------------------------------------
    # 4. Compute metrics
    # -----------------------------------------------------
    y_true = df["label"]
    y_pred = df["pred"]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 Isolation Forest Metrics (95/5 labels)")
    print("-----------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    return precision, recall, f1


if __name__ == "__main__":
    run_metrics()
