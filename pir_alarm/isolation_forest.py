import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import boto3
import io
import os
from collections import defaultdict
from urllib.parse import unquote


def run():
    s3 = boto3.client("s3")
    bucket_name = "iot-glue-bucket-multi-model"
    input_prefix = "output/cleaned/pir_alarm/"
    output_prefix = "model/pir_alarm/"

    # ---- Step 1: list all CSVs ----
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)
    device_files = defaultdict(list)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        parts = key.split("/")
        if len(parts) >= 4:
            encoded_id = parts[3].split("=")[-1]
            device_id = unquote(encoded_id)
            device_files[device_id].append(key)

    # ---- Step 2: train per device ----
    for device_id, keys in device_files.items():
        print(f"\n📡 Processing device: {device_id} with {len(keys)} file(s)")
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf-8-sig")
            dfs.append(df)

        if not dfs:
            print(f"⚠️ No data for {device_id}, skipping.")
            continue

        df = pd.concat(dfs, ignore_index=True)

        # ---- Feature engineering ----
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty:
            print(f"⚠️ Device {device_id} has no valid timestamps, skipping.")
            continue

        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

        features = ["hour", "minute", "day_of_week", "time_diff"]

        # ---- Handle small sample sizes ----
        if len(df) < 2:
            print(f"⚠️ Device {device_id} has only {len(df)} sample(s), skipping training.")
            continue

        # ---- Train/test split ----
        try:
            train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        except ValueError as e:
            print(f"⚠️ Split error for {device_id}: {e}, using all data for training.")
            train_df = df.copy()

        # ---- Scale features ----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features])

        # ---- Train IsolationForest ----
        model = IsolationForest(
            n_estimators=200,
            max_samples=min(512, len(X_train)),
            contamination=0.05,
            random_state=42,
        )
        model.fit(X_train)

        # ---- Save locally ----
        local_model = f"/tmp/{device_id}_if.joblib"
        local_scaler = f"/tmp/{device_id}_scaler.joblib"
        joblib.dump(model, local_model)
        joblib.dump(scaler, local_scaler)

        # ---- Upload to S3 ----
        for path in [local_model, local_scaler]:
            key_name = os.path.basename(path)
            with open(path, "rb") as f:
                s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{key_name}", Body=f.read())
            os.remove(path)
            print(f"✅ Uploaded and removed {path}")


if __name__ == "__main__":
    run()
