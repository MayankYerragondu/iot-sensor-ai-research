from sklearn.svm import OneClassSVM

# Scale
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[features])

# Train OC-SVM with RBF kernel
ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.05,   # proportion of outliers
    gamma="scale"
)
ocsvm.fit(train_scaled)

# Save + upload
local_model_path = f"/tmp/{device_id}_ocsvm.joblib"
local_scaler_path = f"/tmp/{device_id}_scaler.joblib"
joblib.dump(ocsvm, local_model_path)
joblib.dump(scaler, local_scaler_path)

for fname, key_suffix in [(local_model_path, f"{device_id}_ocsvm.joblib"),
                          (local_scaler_path, f"{device_id}_scaler.joblib")]:
    with open(fname, "rb") as f:
        s3.put_object(Bucket=bucket_name, Key=f"{output_prefix}{key_suffix}", Body=f.read())
    os.remove(fname)
    print(f"✅ Uploaded {fname}")
