import os
import joblib
import json
import pandas as pd

print("✅ Starting inference.py — SageMaker model server initialized")

# ========================
# MODEL LOADING
# ========================
def model_fn(model_dir):
    print(f"🔍 Loading model from: {model_dir}")

    # Recursively search for all .joblib files (some tarballs may have nested folders)
    all_files = []
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".joblib"):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise FileNotFoundError(f"No .joblib files found under {model_dir}. Contents: {list(os.walk(model_dir))}")

    # Select main model and optional scaler
    model_file = next((f for f in all_files if not f.endswith("_scaler.joblib")), None)
    scaler_file = next((f for f in all_files if f.endswith("_scaler.joblib")), None)

    if not model_file:
        raise FileNotFoundError("No main model (.joblib) found.")
    if not scaler_file:
        print("⚠️ No scaler file found — model will run on raw features.")

    print(f"📦 Model file: {model_file}")
    print(f"📦 Scaler file: {scaler_file}")

    # Load objects
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if scaler_file else None

    print("✅ Model and scaler loaded successfully")
    return {"model": model, "scaler": scaler}


# ========================
# INPUT PARSING
# ========================
def input_fn(request_body, content_type):
    print(f"🧭 Incoming content type: {content_type}")
    print(f"Raw request body: {request_body}")

    if content_type == "application/json":
        data = json.loads(request_body)
        ts = pd.to_datetime(data["timestamp"], utc=True)

        # Optional previous timestamp for computing time_diff
        if "previous_timestamp" in data:
            ts_prev = pd.to_datetime(data["previous_timestamp"], utc=True)
            time_diff = (ts - ts_prev).total_seconds()
        else:
            time_diff = 0

        features = {
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "time_diff": time_diff
        }

        df = pd.DataFrame([features])
        print(f"✅ Parsed features: {df.to_dict(orient='records')}")
        return df

    raise ValueError(f"Unsupported content type: {content_type}")


# ========================
# PREDICTION
# ========================
def predict_fn(input_data, model_and_scaler):
    model = model_and_scaler["model"]
    scaler = model_and_scaler.get("scaler")

    features = input_data[["hour", "day_of_week", "time_diff"]].values
    if scaler:
        print("🧮 Applying scaler transformation")
        features = scaler.transform(features)
    else:
        print("⚠️ Skipping scaler transformation (no scaler found)")

    # Generic prediction (works for IsolationForest, OneClassSVM, etc.)
    if hasattr(model, "predict"):
        preds = model.predict(features)
    elif hasattr(model, "score_samples"):
        preds = model.score_samples(features)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    print(f"✅ Prediction result: {preds.tolist()}")
    return preds.tolist()


# ========================
# OUTPUT SERIALIZATION
# ========================
def output_fn(prediction, accept):
    response = json.dumps({"prediction": prediction})
    print(f"📤 Sending response: {response}")
    return response


# ========================
# TORCHSERVE HANDLER (SAGEMAKER ENTRYPOINT)
# ========================
def handler(data, context):
    """TorchServe-compatible entrypoint for SageMaker"""
    try:
        print("🚀 Invoking custom handler")
        model = model_fn(context.model_dir)
        input_data = input_fn(data.read().decode("utf-8"), context.request_content_type)
        predictions = predict_fn(input_data, model)
        return output_fn(predictions, context.accept_header)
    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        raise
