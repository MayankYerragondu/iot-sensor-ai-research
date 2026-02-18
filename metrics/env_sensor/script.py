# env_eval_inmemory_single.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking
from tensorflow.keras import Input


# Build model
def build_lstm_autoencoder(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        LSTM(64, activation='tanh', return_sequences=True),
        LSTM(64, activation='tanh'),
        RepeatVector(input_shape[0]),
        LSTM(64, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(3))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# Create sequences
def create_sequences(data, n_steps=10):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i:i+n_steps, :3])
    return np.array(X), np.array(y)


# Main evaluation
def evaluate_env_model_inmemory_single(
    csv_path="env_synthetic.csv",
    device_id="72:a5:12:55:c5:00",
    n_steps=10,
    anomaly_days=20,
    total_days=180,
    window_minutes=5
):
    df = pd.read_csv(csv_path)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)

    # filter one device (though CSV contains only one)
    dev_df = df[df["devid"] == device_id].sort_values("window_start")
    dev_df.reset_index(drop=True, inplace=True)

    total_windows = (total_days * 24 * 60) // window_minutes
    anomaly_windows = (anomaly_days * 24 * 60) // window_minutes
    normal_windows = total_windows - anomaly_windows

    labels = np.array([0]*normal_windows + [1]*anomaly_windows)

    features = ["avg_temperature", "avg_humidity", "avg_lux"]
    scaled = MinMaxScaler().fit_transform(dev_df[features])

    X, y = create_sequences(scaled, n_steps)
    labels_seq = labels[n_steps:]

    # Train in-memory
    model = build_lstm_autoencoder((n_steps, X.shape[2]))
    model.fit(
        X, y,
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )

    # Predict
    preds = model.predict(X, verbose=0)
    mse = np.mean((preds - y)**2, axis=(1, 2))

    # Threshold from NORMAL region
    normal_mse = mse[:normal_windows - n_steps]
    threshold = normal_mse.mean() + 3 * normal_mse.std()

    pred_labels = (mse >= threshold).astype(int)

    p = precision_score(labels_seq, pred_labels, zero_division=0)
    r = recall_score(labels_seq, pred_labels, zero_division=0)
    f = f1_score(labels_seq, pred_labels, zero_division=0)

    print(f"\nDevice: {device_id}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f:.4f}")

    return {"precision": p, "recall": r, "f1": f, "threshold": threshold}


if __name__ == "__main__":
    evaluate_env_model_inmemory_single()
