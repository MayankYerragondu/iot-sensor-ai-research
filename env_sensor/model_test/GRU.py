import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping
import boto3, io, os
from collections import defaultdict

def remove_outliers(df, columns, lower=0.05, upper=0.95):
    filtered_df = df.copy()
    for col in columns:
        q_low = filtered_df[col].quantile(lower)
        q_high = filtered_df[col].quantile(upper)
        filtered_df = filtered_df[(filtered_df[col] >= q_low) & (filtered_df[col] <= q_high)]
    return filtered_df

def create_sequences(data, n_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i:i+n_steps, :3]) 
    return np.array(X), np.array(y)

def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        GRU(64, activation="tanh", return_sequences=True),
        GRU(64, activation="tanh"),
        RepeatVector(input_shape[0]),
        GRU(64, activation="tanh", return_sequences=True),
        TimeDistributed(Dense(3))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def run():
    s3 = boto3.client("s3")
    bucket_name = "iot-glue-bucket-multi-model"
    input_prefix = "output/cleaned/env_sensor/"
    output_prefix = "model/env_sensor/"

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)
    device_files = defaultdict(list)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        parts = key.split("/")
        if len(parts) >= 4:
            device_id = parts[3]
            device_files[device_id].append(key)

    for device_id, keys in device_files.items():
        print(f"\n📡 Processing device: {device_id}")
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="utf-8-sig"))
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)

        # preprocess
        df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce", utc=True)
        df = df.sort_values("window_start").reset_index(drop=True)
        df[["avg_temperature", "avg_humidity", "avg_lux"]] = (
            df[["avg_temperature", "avg_humidity", "avg_lux"]]
            .replace("", np.nan)
            .astype(float)
        )
        df.ffill(inplace=True)
        df["month_of_day"] = df["window_start"].dt.month
