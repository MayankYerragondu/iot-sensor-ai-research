# eval_ocsvm_contact.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_ocsvm_contact(
    filename: str = "75:5d:z5:10:aa:34.csv",
    total_days: int = 180,
    anomaly_days: int = 20,
):
    # ---- Load & prepare ----
    df = pd.read_csv(filename)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    total_hours = total_days * 24
    anomaly_hours = anomaly_days * 24
    normal_hours = total_hours - anomaly_hours

    if len(df) != total_hours:
        raise ValueError(
            f"Expected {total_hours} rows, but CSV has {len(df)} rows. "
            "Make sure total_days in gen & eval match."
        )

    # Labels: last anomaly_hours are anomalies
    idx = np.arange(len(df))
    labels = (idx >= normal_hours).astype(int)  # 0 = normal, 1 = anomaly

    # ---- Features ----
    hour = df["ts"].dt.hour.values.astype(float)

    # Cyclic hour encoding
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # Alarm flag
    alarm_int = (df["csAlarm"] == "TRUE").astype(int)

    # Rolling behavior
    roll_24 = alarm_int.rolling(window=24, min_periods=1).mean().values
    roll_72 = alarm_int.rolling(window=72, min_periods=1).mean().values

    # Compute TRUE streak length
    alarm_values = alarm_int.values
    streak = np.zeros(len(alarm_values))

    count = 0
    for i in range(len(alarm_values)):
        if alarm_values[i] == 1:
            count += 1
        else:
            count = 0
        streak[i] = count

    # Feature matrix
    X = np.column_stack([
        hour_sin,
        hour_cos,
        roll_24,
        roll_72,
        streak
    ])

    # ---- Train on normal only ----
    X_train = X[labels == 0]
    X_full = X

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)

    # ---- OneClassSVM ----
    # nu ≈ expected fraction of anomalies in training (we keep small)
    model = OneClassSVM(
        kernel="rbf",
        nu=0.03,        # tweak up/down to change recall/precision
        gamma=0.5
    )
    model.fit(X_train_scaled)

    # SVM: -1 = anomaly, 1 = normal
    raw_pred = model.predict(X_full_scaled)
    preds = (raw_pred == -1).astype(int)

    # ---- Metrics ----
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f = f1_score(labels, preds, zero_division=0)

    print("\nFINAL OCSVM RESULT — contact_synthetic.csv")
    print("==========================================")
    print(f"Total rows        : {len(df)}")
    print(f"Normal hours      : {normal_hours}")
    print(f"Anomalous hours   : {anomaly_hours}")
    print(f"Predicted anomaly : {preds.sum()} / {anomaly_hours}")
    print("------------------------------------------")
    print(f"Precision         : {p:.4f}")
    print(f"Recall            : {r:.4f}")
    print(f"F1-score          : {f:.4f}")

    return p, r, f


if __name__ == "__main__":
    evaluate_ocsvm_contact()
