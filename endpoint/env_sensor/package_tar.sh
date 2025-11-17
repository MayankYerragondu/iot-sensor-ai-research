#!/bin/bash
set -e

BUCKET="iot-glue-bucket-multi-model"
S3_MODEL_PATH="model/env_sensor/"
S3_ENDPOINT_PATH="endpoint/env_sensor/"

mkdir -p models/raw
aws s3 cp "s3://${BUCKET}/${S3_MODEL_PATH}" models/raw/ \
  --recursive \
  --exclude "*" \
  --include "*.h5" \
  --include "*_scaler.joblib"

cd models/raw

for f in *.h5; do
  [[ -f "$f" ]] || continue
  base="${f%.h5}"
  mkdir -p "../${base}"
  mv "$f" "../${base}/"
  [[ -f "${base}_scaler.joblib" ]] && mv "${base}_scaler.joblib" "../${base}/" || true
  cp ../../inference.py "../${base}/"
done

cd ..

for dir in */; do
  [[ -d "$dir" ]] || continue
  [[ "$dir" == "raw/" ]] && continue
  base=$(basename "$dir")
  echo "📦 Packaging: $base"
  tar -czf "${base}.tar.gz" -C "$dir" .
  aws s3 cp "${base}.tar.gz" "s3://${BUCKET}/${S3_ENDPOINT_PATH}${base}.tar.gz"
done

echo "✅ All models packaged and uploaded to s3://${BUCKET}/${S3_ENDPOINT_PATH}"
