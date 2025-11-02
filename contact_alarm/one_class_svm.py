import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import boto3
import io
import os
from collections import defaultdict
from urllib.parse import unquote

def run():
    s3 = boto3.client('s3')
    bucket_name = 'iot-glue-bucket-multi-model'
    input_prefix = 'output/cleaned/contact_alarm/'
    output_prefix = 'model/contact_alarm/'

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)
    device_files = defaultdict(list)

    for obj in response.get('Contents', []):
        key = obj['Key']
        parts = key.split('/')
        if len(parts) >= 4:
            encoded_id = parts[3].split("=")[-1]
            device_id = unquote(encoded_id)
            device_files[device_id].append(key)

    for device_id, keys in device_files.items():
        print(f"\n📡 Processing device: {device_id} with {len(keys)} file(s)")
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8-sig')
            dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        distinct_ts = df['timestamp'].drop_duplicates().sort_values().reset_index(drop=True)

        hour_of_day = distinct_ts.dt.hour
        day_of_week = distinct_ts.dt.dayofweek
        time_diff_hours = distinct_ts.diff().dt.total_seconds().fillna(0) / 3600
        X_true = np.column_stack((hour_of_day, day_of_week, time_diff_hours))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_true)

        model = OneClassSVM(gamma="scale", nu=0.1)
        model.fit(X_scaled)

        local_model_path = f"/tmp/{device_id}_ocsvm.joblib"
        local_scaler_path = f"/tmp/{device_id}_scaler.joblib"
        joblib.dump(model, local_model_path)
        joblib.dump(scaler, local_scaler_path)

        for path in [local_model_path, local_scaler_path]:
            key_name = os.path.basename(path)
            with open(path, "rb") as f:
                s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{key_name}", Body=f.read())
            os.remove(path)
            print(f"✅ Uploaded and removed {path}")

if __name__ == "__main__":
    run()
