import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
from datetime import datetime
import boto3
import io
import os
from collections import defaultdict

def run():
    s3 = boto3.client('s3')
    bucket_name = 'iot-glue-bucket-multi-model'
    input_prefix = 'output/cleaned/contact_alarm/'
    output_prefix = 'model/contact_alarm/'

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

    # Step 3: Process each device
    for device_id, keys in device_files.items():
        print(f"\n📡 Processing device: {device_id} with {len(keys)} file(s)")

        # Load & concatenate CSVs
        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8-sig')
            dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        # Parse and process time features
        df['hour'] = pd.to_datetime(df['hour'], errors='coerce')
        df = df.dropna(subset=['hour'])  # Drop rows with bad dates

        distinct_hours = df['hour'].drop_duplicates().sort_values().reset_index(drop=True)

        hour_of_day = distinct_hours.dt.hour
        day_of_week = distinct_hours.dt.dayofweek
        time_diff_hours = distinct_hours.diff().dt.total_seconds().fillna(0) / 3600

        X_true = np.column_stack((hour_of_day, day_of_week, time_diff_hours))

        # Train model
        model = OneClassSVM(gamma="scale", nu=0.1)
        model.fit(X_true)

        # Save model to /tmp and upload to S3
        local_model_path = f"/tmp/{device_id}.joblib"
        joblib.dump(model, local_model_path)

        with open(local_model_path, "rb") as f:
            s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{device_id}_if.joblib", Body=f.read())

        os.remove(local_model_path)
        print(f"✅ Uploaded model to s3://{bucket_name}/{output_prefix}{device_id}_if.joblib and removed local copy.")


if __name__ == "__main__":
    run()