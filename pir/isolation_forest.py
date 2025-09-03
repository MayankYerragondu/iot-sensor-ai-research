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
    # Replace with the path to your local CSV file
    training_file_path = '/Users/renqingyang/vs-code-project/sagemaker-project/max_amount_data.csv'

    # Read CSV from local file system
    train_df = pd.read_csv(training_file_path)

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
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Initialize the Isolation Forest model
    isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)

    # Train the model on the training data
    isolation_forest_model.fit(train_features_scaled)


if __name__ == "__main__":
    run()