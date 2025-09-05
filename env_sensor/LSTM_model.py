import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import tensorflow as tf
import boto3
import io
import os
from datetime import datetime
from collections import defaultdict

def run():
    s3 = boto3.client('s3')
    bucket_name = 'iot-glue-bucket-multi-model'
    input_prefix = 'output/cleaned/env_sensor/'
    output_prefix = 'model/env_sensor/'

    # Step 1: List all files under the input prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)

    # Step 2: Organize files by device ID
    device_files = defaultdict(list)
    for obj in response.get('Contents', []):
        key = obj['Key']
        parts = key.split('/')
        if len(parts) >= 4:
            device_id = parts[3]
            device_files[device_id].append(key)

    # -------------------------------
    # Helper Functions
    # -------------------------------
    def remove_outliers(df, columns, lower=0.05, upper=0.95):
        filtered_df = df.copy()
        for col in columns:
            q_low = filtered_df[col].quantile(lower)
            q_high = filtered_df[col].quantile(upper)
            filtered_df = filtered_df[(filtered_df[col] >= q_low) & (filtered_df[col] <= q_high)]
        return filtered_df

    def build_encoder_decoder_lstm(input_shape):
        """2-layer encoder-decoder LSTM (64 units each)"""
        model = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.0),

            # Encoder
            LSTM(64, activation='tanh', return_sequences=True),
            LSTM(64, activation='tanh'),

            # Repeat context vector for decoder
            RepeatVector(input_shape[0]),

            # Decoder
            LSTM(64, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(3))  # Predict temperature, humidity, lux
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def create_sequences(data, n_steps=10):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i:i+n_steps, :3])  # Decoder predicts sequence of temp/humidity/lux
        return np.array(X), np.array(y)

    # -------------------------------
    # Train Model Per Device
    # -------------------------------
    for device_id, keys in device_files.items():
        print(f"\n📡 Processing device: {device_id} with {len(keys)} file(s)")

        # Load & concat all CSVs
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8-sig')
            dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        # Preprocess
        df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce", utc=True)
        df = df.sort_values("window_start").reset_index(drop=True)

        df[["avg_temperature", "avg_humidity", "avg_lux"]] = (
            df[["avg_temperature", "avg_humidity", "avg_lux"]]
            .replace("", np.nan)
            .astype(float)
        )
        df.ffill(inplace=True)

        df["month_of_day"] = df["window_start"].dt.month
        df["hour_of_day"] = df["window_start"].dt.hour
        df["day_of_week"] = df["window_start"].dt.dayofweek

        features = [
            "avg_temperature", 
            "avg_humidity", 
            "avg_lux", 
            "month_of_day",
            "hour_of_day", 
            "day_of_week", 
        ]

        df_clean = remove_outliers(df, features, lower=0.05, upper=0.95)
        if len(df_clean) < 20:
            print(f"⚠️  {device_id}, too little data after cleaning")
            continue

        # Normalize and prepare sequences
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_clean[features])
        X, y = create_sequences(scaled_data, n_steps=10)
        if len(X) < 10:
            print(f"⚠️  {device_id}, not enough sequence data")
            continue

        # Train/test split (80/20 temporal)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        # Build & train encoder–decoder LSTM
        model = build_encoder_decoder_lstm(input_shape=(X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=128,
            callbacks=[early_stop],
            verbose=0
        )

        # Save model to memory and upload to S3
        local_path = f"/tmp/{device_id}.keras"
        model.save(local_path)

        with open(local_path, "rb") as f:
            s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{device_id}.keras", Body=f)

        os.remove(local_path)
        print(f"📦 Uploaded model to s3://{bucket_name}/{output_prefix}{device_id}.keras")

if __name__ == "__main__":
    run()
