import pandas as pd
import numpy as np
import boto3
import io
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    TimeDistributed, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
from collections import defaultdict


# -------------------------------------------------------
# Helper: Remove outliers using quantile thresholds
# -------------------------------------------------------
def remove_outliers(df, cols, low=0.05, high=0.95):
    out = df.copy()
    for c in cols:
        ql, qh = out[c].quantile(low), out[c].quantile(high)
        out = out[(out[c] >= ql) & (out[c] <= qh)]
    return out


# -------------------------------------------------------
# Helper: Create sliding-window sequences
# -------------------------------------------------------
def create_sequences(data, n_steps=10):
    """
    Converts normalized features into supervised sequences.

    X: past n_steps timesteps of features
    y: same sequence’s temperature, humidity, lux
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i:i+n_steps, :3])  # predict all timesteps
    return np.array(X), np.array(y)


# -------------------------------------------------------
# Transformer Encoder-Decoder Model
# -------------------------------------------------------
def build_transformer(input_shape, heads=4, ff_dim=64):
    """
    Simple Transformer model for sequence-to-sequence regression.
    - Multi-head attention over input
    - Feed-forward dense block
    - TimeDistributed(Dense) to predict features per timestep
    """
    inp = Input(shape=input_shape)

    # Self-attention
    attn = MultiHeadAttention(num_heads=heads, key_dim=input_shape[1])(inp, inp)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inp + attn)

    # Feed-forward block
    ff = Dense(ff_dim, activation="relu")(out1)
    ff = Dense(input_shape[1])(ff)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff)

    # Output: 3 features per timestep
    out = TimeDistributed(Dense(3))(out2)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------------
def run():
    s3 = boto3.client("s3")

    bucket = "iot-glue-bucket-multi-model"
    input_prefix = "output/cleaned/env_sensor/"
    output_prefix = "model/env_sensor/"

    # Step 1: List files in S3
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=input_prefix)

    # Group by device ID
    device_files = defaultdict(list)
    for obj in resp.get("Contents", []): 
        parts = obj["Key"].split("/")
        if len(parts) >= 4:
            device_files[parts[3]].append(obj["Key"])

    # ---------------------------------------------------
    # Train model per device
    # ---------------------------------------------------
    for dev, keys in device_files.items():
        print(f"\n📡 Processing device: {dev}")

        # Step 2: Load device CSVs
        dfs = [
            pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=k)["Body"].read()))
            for k in keys
        ]
        if not dfs:
            continue

        df = pd.concat(dfs).sort_values("window_start").reset_index(drop=True)

        # Step 3: Preprocess
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
        df[["avg_temperature", "avg_humidity", "avg_lux"]] = (
            df[["avg_temperature", "avg_humidity", "avg_lux"]].astype(float)
        )
        df.ffill(inplace=True)

        # Add temporal features
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
            print(f"⚠️ Skipping {dev}: not enough data")
            continue

        # Step 5: Normalize & sequence creation
        scaled = MinMaxScaler().fit_transform(df[features])
        X, y = create_sequences(scaled)
        if len(X) < 10:
            print(f"⚠️ Skipping {dev}: insufficient sequence data")
            continue

        # Train/test split (80/20)
        s = int(len(X) * 0.8)
        Xt, Yt, Xv, Yv = X[:s], y[:s], X[s:], y[s:]

        # Step 6: Build and train model
        model = build_transformer((X.shape[1], X.shape[2]))
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            Xt, Yt,
            validation_data=(Xv, Yv),
            epochs=50, batch_size=128,
            callbacks=[early_stop],
            verbose=0
        )

        # Step 7: Save and upload model
        path = f"/tmp/{dev}.keras"
        model.save(path)
        with open(path, "rb") as f:
            s3.put_object(Bucket=bucket, Key=f"{output_prefix}{dev}.keras", Body=f)
        os.remove(path)

        print(f"📦 Transformer model uploaded → s3://{bucket}/{output_prefix}{dev}.keras")


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    run()
