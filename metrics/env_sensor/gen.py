# env_csv_gen_single.py
import pandas as pd
import numpy as np

def generate_env_csv_single(
    filename="env_synthetic.csv",
    device_id="72:a5:12:55:c5:00",
    total_days=30,
    anomaly_days=5,
    window_minutes=5
):
    np.random.seed(42)

    total_windows = (total_days * 24 * 60) // window_minutes
    anomaly_windows = (anomaly_days * 24 * 60) // window_minutes
    normal_windows = total_windows - anomaly_windows

    timestamps = pd.date_range(
        start="2024-01-01T00:00:00Z",
        periods=total_windows,
        freq=f"{window_minutes}min"
    )

    # --- IMPORTANT: convert to numpy arrays immediately ---
    hours = timestamps.hour.to_numpy()
    hour_sin = np.sin(2 * np.pi * hours / 24)

    temp = (20 + 5 * hour_sin + np.random.normal(0, 0.6, total_windows)).astype(float)
    hum  = (50 - 12 * hour_sin + np.random.normal(0, 2, total_windows)).astype(float)
    lux  = (
        np.clip(2000 * np.maximum(hour_sin, 0), 0, None)
        + np.random.normal(0, 60, total_windows)
    ).astype(float)

    # --- Apply anomalies safely on numpy arrays ---
    temp[normal_windows:] += np.random.normal(8, 2, anomaly_windows)
    hum[normal_windows:]  += np.random.normal(-10, 4, anomaly_windows)
    lux[normal_windows:]  += np.random.normal(400, 120, anomaly_windows)

    rows = []
    for i in range(total_windows):
        start = timestamps[i]
        end = start + pd.Timedelta(minutes=window_minutes)

        rows.append({
            "devid": device_id,
            "window_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "avg_temperature": round(float(temp[i]), 2),
            "avg_humidity": round(float(hum[i]), 2),
            "avg_lux": round(float(lux[i]), 2)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

    print(f"\nGenerated: {filename}")
    print(f"Device: {device_id}")
    print(f"Total windows: {total_windows}")
    print(f"Normal windows: {normal_windows}")
    print(f"Anomaly windows: {anomaly_windows}")

    return filename


if __name__ == "__main__":
    generate_env_csv_single()
