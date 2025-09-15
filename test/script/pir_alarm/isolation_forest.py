import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import boto3
import io
import os

def run():
    s3 = boto3.client('s3')
    bucket_name = 'iot-glue-bucket-multi-model'
    input_prefix = 'output/cleaned/pir_alarm/'
    output_prefix = 'model/pir_alarm/'
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

        train_df = pd.concat(dfs, ignore_index=True)

        # Ensure the CSV has a 'timestamp' column and convert it to datetime
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

        # Extract features from the timestamp
        train_df['hour'] = train_df['timestamp'].dt.hour
        train_df['minute'] = train_df['timestamp'].dt.minute
        # train_df['second'] = train_df['timestamp'].dt.second
        train_df['day_of_week'] = train_df['timestamp'].dt.dayofweek
        # train_df['day_of_month'] = train_df['timestamp'].dt.day

        # Calculate the time difference between consecutive events (in seconds)
        train_df['time_diff'] = train_df['timestamp'].diff().dt.total_seconds().fillna(0)

        # Split the data: 80% for training, 20% for testing
        train_data, test_data = train_test_split(train_df, test_size=0.3, random_state=42)

        # Use these features for training the anomaly detection model
        features = ['hour', 'minute', 'day_of_week', 'time_diff']

        train_features = train_data[features]
        test_features = test_data[features]

        # Scale the features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[features])


        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(train_scaled)

        # Save locally then upload
        local_model_path = f"/tmp/{device_id}.joblib"
        local_scaler_path = f"/tmp/{device_id}_scaler.joblib"
        joblib.dump(model, local_model_path)
        joblib.dump(scaler, local_scaler_path)

        # Upload both to S3
        for fname, key_suffix in [(local_model_path, f"{device_id}_if.joblib"),
                                (local_scaler_path, f"{device_id}_scaler.joblib")]:
            with open(fname, "rb") as f:
                s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{key_suffix}", Body=f.read())
            os.remove(fname)
            print(f"✅ Uploaded and removed {fname}")
            
if __name__ == "__main__":
    run()
    # test_features_scaled = scaler.transform(test_features)

    # # Initialize the Isolation Forest model
    # isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)

    # # Train the model on the training data
    # isolation_forest_model.fit(train_features_scaled)


    # # Assume this is the new input data you want to check
    # new_data = {
    #     'timestamp': [
    #         '2024-06-02T00:00:09.561Z',
    #         '2024-06-09T00:00:09.561Z',
    #         '2024-06-16T00:00:09.561Z',
    #         '2024-06-23T00:00:09.561Z',
    #         '2024-06-30T00:00:09.561Z'
    #     ],
    #     # Include any other features if you have them, otherwise we just process the timestamp
    # }
    # new_df = pd.DataFrame(new_data)
    # new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    # # Extract the same features from the new data
    # new_df['hour'] = new_df['timestamp'].dt.hour
    # new_df['minute'] = new_df['timestamp'].dt.minute
    # # new_df['second'] = new_df['timestamp'].dt.second
    # new_df['day_of_week'] = new_df['timestamp'].dt.dayofweek
    # # new_df['day_of_month'] = new_df['timestamp'].dt.day

    # # Calculate the time difference between consecutive events (in seconds)
    # new_df['time_diff'] = new_df['timestamp'].diff().dt.total_seconds().fillna(0)

    # # Prepare the new data for prediction
    # new_features = new_df[['hour', 'minute', 'day_of_week', 'time_diff']]
    # new_features_scaled = scaler.transform(new_features)  # Use the same scaler as training

    # # Predict whether the new data points are anomalous
    # new_df['anomaly'] = isolation_forest_model.predict(new_features_scaled)

    # # -1 indicates an anomaly, 1 indicates normal
    # print('isolation_forest_model: ')
    # print(new_df[['timestamp', 'anomaly']])

    # test_predictions = isolation_forest_model.predict(test_features_scaled)

    # wrong_predictions = (test_predictions == -1).sum()
    # total_predictions = len(test_predictions)
    # wrong_percentage = (wrong_predictions / total_predictions) * 100

    # print(f"Percentage of wrong predictions: {wrong_percentage:.2f}%")

    # save_path = '/Users/renqingyang/vs-code-project/sagemaker-project/hogar-multi-model-project/models/pir_alarm/isolation_forest_model.joblib'

    # joblib.dump(isolation_forest_model, save_path)