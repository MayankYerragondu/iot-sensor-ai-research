from sklearn.mixture import GaussianMixture

# Scale the features (reuse your StandardScaler)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[features])

# Train a GMM with 2 components (normal vs anomaly)
gmm = GaussianMixture(
    n_components=2,
    covariance_type='full',
    random_state=42
)
gmm.fit(train_scaled)

# Save model + scaler
local_model_path = f"/tmp/{device_id}_gmm.joblib"
local_scaler_path = f"/tmp/{device_id}_scaler.joblib"
joblib.dump(gmm, local_model_path)
joblib.dump(scaler, local_scaler_path)

# Upload to S3
for fname, key_suffix in [(local_model_path, f"{device_id}_gmm.joblib"),
                          (local_scaler_path, f"{device_id}_scaler.joblib")]:
    with open(fname, "rb") as f:
        s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{key_suffix}", Body=f.read())
    os.remove(fname)
    print(f"✅ Uploaded {fname}")
