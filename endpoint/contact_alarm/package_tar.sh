#!/bin/bash
set -euo pipefail

BUCKET="iot-glue-bucket-multi-model"
S3_MODEL_PATH="model/contact_alarm/"
S3_ENDPOINT_PATH="endpoint/contact_alarm/"

# 1️⃣ Download all .joblib model artifacts
mkdir -p models/raw
aws s3 cp "s3://${BUCKET}/${S3_MODEL_PATH}" models/raw/ \
  --recursive \
  --exclude "*" \
  --include "*.joblib"

# 2️⃣ Organize each model into its own folder
cd models/raw
for f in *_if.joblib; do
  [[ -f "$f" ]] || continue
  base="${f%_if.joblib}"
  mkdir -p "../${base}"
  echo "📦 Organizing model: $base"

  # move both joblib files + copy inference.py
  mv "${base}_if.joblib" "../${base}/" 2>/dev/null || true
  cp ../../inference.py "../${base}/"
done

cd ..

for dir in */; do
  [[ -d "$dir" ]] || continue
  [[ "$dir" == "raw/" ]] && continue  # 🚫 Skip raw folder

  base=$(basename "$dir")
  echo "🎯 Packaging: $base"

  tar -czf "${base}.tar.gz" -C "$dir" .
  aws s3 cp "${base}.tar.gz" "s3://${BUCKET}/${S3_ENDPOINT_PATH}${base}.tar.gz"

  echo "✅ Uploaded: ${base}.tar.gz"
  rm -f "${base}.tar.gz"
done


cd ..
rm -rf models