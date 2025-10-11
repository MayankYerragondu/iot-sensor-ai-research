import pandas as pd
import numpy as np
import joblib
import boto3
import io
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict


# -------------------------------------------------------
# Helper: Remove outliers based on quantiles
# -------------------------------------------------------
def remove_outliers(df, cols, low=0.03, high=0.96):
    out = df.copy()
    for c in cols:
        ql, qh = out[c].quantile(low), out[c].quantile(high)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out


# -------------------------------------------------------
# Helper: Create sequences for supervised learning
# -------------------------------------------------------
def create_sequences(data, n_steps=10):
    """
    Turns time-series data into sliding windows.

    Parameters:
        data     : normalized feature array
        n_steps  : lookback window size (default 10)

    Returns:
        X : array of shape (samples, n_steps, features)
        y : last timestep labels (temperature, humidity, lux)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        # past n_steps features
        X.append(data[i:i+n_steps])
        # predict last step's target (3 values)
        y.append(data[i+n_steps-1, :3])
    return np.array(X), np.array(y)


# -------------------------------------------------------
# Main training pipeline
# -------------------------------------------------------
def run():
    s3 = boto3.client("s3")

    bucket = "iot-glue-bucket-multi-model"
    input_prefix = "output/cleaned/env_sensor/"
    output_prefix = "model/env_sensor/"

    # Step 1: List files from S3
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=input_prefix)

    # Group files per device_id
    device_files = defaultdict(list)
    for obj in resp.get("Contents", []):
        parts = obj["Key"].split("/")
        if len(parts) >= 4:
            device_id = parts[3]
            device_files[device_id].append(obj["Key"])

    # ---------------------------------------------------
    # Process each device separately
    # ---------------------------------------------------
    for dev, keys in device_files.items():
        print(f"\n📡 Processing device: {dev}")

        # Step 2: Load all CSVs for this device
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket, Key=key)
            dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read())))
        if not dfs:
            continue

        df = pd.concat(dfs).sort_values("window_start").reset_index(drop=True)

        # Step 3: Preprocess
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")

        # Convert sensor values to numeric
        df[["avg_temperature", "avg_humidity", "avg_lux"]] = (
            df[["avg_temperature", "avg_humidity", "avg_lux"]].astype(float)
        )

        # Fill missing values forward
        df.ffill(inplace=True)

        # Add time-based features
        df["month_of_day"] = df["window_start"].dt.month
        df["hour_of_day"] = df["window_start"].dt.hour
        df["day_of_week"] = df["window_start"].dt.dayofweek

        features = [
            "avg_temperature", "avg_humidity", "avg_lux",
            "month_of_day", "hour_of_day", "day_of_week"
        ]

        # Step 4: Outlier removal
        df = remove_outliers(df, features)
        if len(df) < 20:
            print(f"⚠️ Skipping {dev}, insufficient data after cleaning")
            continue

        # Step 5: Normalize features
        scaled = MinMaxScaler().fit_transform(df[features])

        # Step 6: Create sequences
        X, y = create_sequences(scaled, n_steps=10)
        if len(X) < 10:
            print(f"⚠️ Skipping {dev}, not enough sequence data")
            continue

        # Flatten sequence for RF (samples × features)
        ns, steps, nf = X.shape
        X_flat = X.reshape(ns, steps * nf)

        # Step 7: Train RandomForest
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_flat, y)

        # Step 8: Save locally and upload to S3
        local_path = f"/tmp/{dev}.joblib"
        joblib.dump(model, local_path)

        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key=f"{output_prefix}{dev}.joblib",
                Body=f
            )

        os.remove(local_path)
        print(f"📦 RF model uploaded → s3://{bucket}/{output_prefix}{dev}.joblib")


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    run()
