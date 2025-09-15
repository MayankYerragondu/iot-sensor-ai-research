import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime
import boto3
import io
from tensorflow.keras import Input
import os

def run():
    s3 = boto3.client('s3')
    bucket_name = 'iot-glue-bucket-multi-model'
    input_prefix = 'output/cleaned/env_sensor/'
    output_prefix = 'model/env_sensor/'

    # Step 1: List all files under the input prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)

    # Step 2: Organize files by device ID
    from collections import defaultdict
    device_files = defaultdict(list)

    for obj in response.get('Contents', []):
        key = obj['Key']
        parts = key.split('/')
        if len(parts) >= 4:
            device_id = parts[3]
            device_files[device_id].append(key)


    # -------------------------------
    # LSTM Model Generator Function
    # -------------------------------
    def remove_outliers(df, columns, lower=0.05, upper=0.95):
        filtered_df = df.copy()
        for col in columns:
            q_low = filtered_df[col].quantile(lower)
            q_high = filtered_df[col].quantile(upper)
            filtered_df = filtered_df[(filtered_df[col] >= q_low) & (filtered_df[col] <= q_high)]
        return filtered_df

    def build_lstm_model(input_shape, units=256):
        model = Sequential([
            Input(shape=input_shape),
            Masking(mask_value=0.),
            LSTM(units, activation='tanh'),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_sequences(data, n_steps=10):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps][:3])  # Predict temperature, humidity, lux
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
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8-sig')  # or utf-8
            print(df.columns.tolist())

            dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        # Preprocess
        # df["window_start"] = pd.to_datetime(df["window_start"])
        df["window_start"] = df["window_start"].astype(str)
        df["window_start"] = pd.to_datetime(df["window_start"], format="ISO8601", utc=True, errors="coerce")

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

        # Use all data for training and evaluation
        X_train, y_train = X, y
        X_test, y_test = X, y


        # Build & train
        model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), units=64)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

        # Save model to memory and upload to S3
        # after model.fit(...)
        local_path = f"/tmp/{device_id}.keras"
        model.save(local_path)  # uses extension to infer format

        # Upload to S3
        with open(local_path, "rb") as f:
            s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{device_id}.keras", Body=f)

        # Clean up disk
        os.remove(local_path)
        print(f"📦 Uploaded model to s3://{bucket_name}/{output_prefix}{device_id}.keras")

if __name__ == "__main__":
    run()
# # -------------------------------
# # Load and preprocess data
# # -------------------------------
# df = pd.read_csv("/Users/renqingyang/vs-code-project/sagemaker-project/hogar-multi-model-project/models/env_sensor/env_data.csv")
# df["window_start"] = pd.to_datetime(df["window_start"])
# df = df.sort_values("window_start").reset_index(drop=True)

# # Fill missing values and convert to float
# df[["avg_temperature", "avg_humidity", "avg_lux"]] = (
#     df[["avg_temperature", "avg_humidity", "avg_lux"]]
#     .replace("", np.nan)
#     .astype(float)
# )
# df.fillna(method="ffill", inplace=True)

# # Add time-based features
# df["month_of_day"] = df["window_start"].dt.month
# df["hour_of_day"] = df["window_start"].dt.hour
# df["day_of_week"] = df["window_start"].dt.dayofweek


# features = [
#     "avg_temperature", 
#     "avg_humidity", 
#     "avg_lux", 
#     "month_of_day",
#     "hour_of_day", 
#     "day_of_week", 
# ]

# df_clean = remove_outliers(df, features, lower=0.05, upper=0.95)  # Keep middle 90%

# # Select features for training


# # Normalize features
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df_clean[features])

# # Create LSTM sequences
# def create_sequences(data, n_steps=10):
#     X, y = [], []
#     for i in range(len(data) - n_steps):
#         X.append(data[i:i+n_steps])
#         y.append(data[i+n_steps][:3])  # Only predict temp, humidity, lux
#     return np.array(X), np.array(y)

# X, y = create_sequences(scaled_data, n_steps=10)

# # Train/test split (80/20)
# split_idx = int(len(X) * 0.8)
# X_train, X_test = X[:split_idx], X[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]

# # Build and train the model
# model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), units=64)
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stop])

# # -------------------------------
# # Evaluate model
# # -------------------------------
# y_pred = model.predict(X_test)

# # Inverse transform only the predicted part (first 3 columns)
# y_test_full = np.concatenate([y_test, X_test[:, -1, 3:]], axis=1)  # append time features back
# y_pred_full = np.concatenate([y_pred, X_test[:, -1, 3:]], axis=1)

# y_test_inv = scaler.inverse_transform(y_test_full)[:, :3]
# y_pred_inv = scaler.inverse_transform(y_pred_full)[:, :3]

# # Metrics
# rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
# mae = mean_absolute_error(y_test_inv, y_pred_inv)

# print(f"\n✅ Evaluation Metrics with Time Features:")
# print(f"RMSE: {rmse:.3f}")
# print(f"MAE:  {mae:.3f}")
