import pandas as pd
import numpy as np
import joblib
import boto3
import io
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape, Input
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict


# -------------------------------------------------------
# Helper: Remove outliers based on quantiles
# -------------------------------------------------------
def remove_outliers(df, cols, low=0.05, high=0.95):
    out = df.copy()
    for c in cols:
        ql, qh = out[c].quantile(low), out[c].quantile(high)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out


# -------------------------------------------------------
# Helper: Create supervised sequences (sliding window)
# -------------------------------------------------------
def create_sequences(data, n_steps=10):
    """
    Converts normalized feature data into sliding windows.

    X: sequence of past n_steps
    y: sequence of past n_steps (target is all timesteps: temp, humidity, lux)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i:i+n_steps, :3])  # predict sequence of 3 features
    return np.array(X), np.array(y)


# -------------------------------------------------------
# CNN model for sequence regression
# -------------------------------------------------------
def build_cnn(input_shape):
    """
    Simple 1D CNN model for time-series forecasting:
    - Conv1D layers extract local temporal patterns
    - MaxPooling reduces dimension
    - Dense layers map to output
    - Reshape to match (timesteps, 3 features)
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(32, 3, activation="relu"),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(input_shape[0]*3),
        Reshape((input_shape[0], 3))  # reshape back to sequence
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------
def run():
    s3 = boto3.client("s3")

    bucket = "iot-glue-bucket-multi-model"
    input_prefix = "output/cleaned/env_sensor/"
    output_prefix = "model/env_sensor/"

    # Step 1: List input files from S3
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=input_prefix)

    # Group files by device ID
    device_files = defaultdict(list)
    for obj in resp.get("Contents", []):
        parts = obj["Key"].split("/")
        if len(parts) >= 4:
            device_files[parts[3]].append(obj["Key"])

    # ---------------------------------------------------
    # Train CNN per device
    # ---------------------------------------------------
    for dev, keys in device_files.items():
        print(f"\n📡 Processing device: {dev}")

        # Step 2: Load all device CSVs
        dfs = [
            pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=k)["Body"].read()))
            for k in keys
        ]
        if not dfs:
            continue

        df = pd.concat(dfs).sort_values("window_start").reset_index(drop=True)

        # Step 3: Preprocess
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
        df[["avg_temperature","avg_humidity","avg_lux"]] = (
            df[["avg_temperature","avg_humidity","avg_lux"]].astype(float)
        )
        df.ffill(inplace=True)

        # Add time-based features
        df["month_of_day"] = df["window_start"].dt.month
        df["hour_of_day"] = df["window_start"].dt.hour
        df["day_of_week"] = df["window_start"].dt.dayofweek

        features = [
            "avg_temperature","avg_humidity","avg_lux",
            "month_of_day","hour_of_day","day_of_week"
        ]

        # Step 4: Outlier removal
        df = remove_outliers(df, features)
        if len(df) < 20:
            print(f"⚠️ Skipping {dev}, not enough data after cleaning")
            continue

        # Step 5: Normalize features
        scaled = MinMaxScaler().fit_transform(df[features])

        # Step 6: Create sequences
        X, y = create_sequences(scaled)
        if len(X) < 10:
            print(f"⚠️ Skipping {dev}, insufficient sequence data")
            continue

        # Train/validation split
        split = int(len(X) * 0.8)
        X_train, Y_train, X_val, Y_val = X[:split], y[:split], X[split:], y[split:]

        # Step 7: Build & train CNN
        model = build_cnn((X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=50, batch_size=128,
            callbacks=[early_stop],
            verbose=0
        )

        # Step 8: Save & upload model to S3
        local_path = f"/tmp/{dev}.keras"
        model.save(local_path)

        with open(local_path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=f"{output_prefix}{dev}.keras", Body=f)

        os.remove(local_path)
        print(f"📦 CNN model uploaded → s3://{bucket}/{output_prefix}{dev}.keras")


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
if __name__ == "__main__":
    run()
