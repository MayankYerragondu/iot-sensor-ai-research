import os
import joblib
import json
import pandas as pd

print("✅ Starting inference.py — model server initialized")

def model_fn(model_dir):
    print(f"🔍 Loading model from: {model_dir}")

    # Recursively walk the directory so nested paths (like /code/) are included
    all_files = []
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".joblib"):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise FileNotFoundError(f"No .joblib files found under {model_dir}. Found: {os.listdir(model_dir)}")

    # Find model and scaler explicitly
    model_file = next((f for f in all_files if not f.endswith("_scaler.joblib")), None)
    scaler_file = next((f for f in all_files if f.endswith("_scaler.joblib")), None)

    if not model_file:
        raise FileNotFoundError("No main model (.joblib) file found.")
    if not scaler_file:
        print("⚠️ No scaler file found — proceeding without scaling.")

    print(f"📦 Model file: {model_file}")
    print(f"📦 Scaler file: {scaler_file}")

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file) if scaler_file else None

    print("✅ Model and scaler loaded successfully")
    return {"model": model, "scaler": scaler}


def input_fn(request_body, content_type):
    print(f"🧭 Incoming content type: {content_type}")
    print(f"Raw request body: {request_body}")

    if content_type == "application/json":
        data = json.loads(request_body)
        ts = pd.to_datetime(data["timestamp"], utc=True)

        features = {
            "hour": ts.hour,
            "minute": ts.minute,
            "day_of_week": ts.dayofweek,
            "time_diff": 0
        }

        df = pd.DataFrame([features])
        print(f"✅ Parsed features: {df.to_dict(orient='records')}")
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_and_scaler):
    model = model_and_scaler["model"]
    scaler = model_and_scaler.get("scaler")

    features = input_data[["hour", "minute", "day_of_week", "time_diff"]].values

    if scaler:
        features = scaler.transform(features)

    preds = model.predict(features)
    print(f"✅ Prediction result: {preds.tolist()}")
    return preds.tolist()


def output_fn(prediction, accept):
    response = json.dumps({"prediction": prediction})
    print(f"📤 Sending response: {response}")
    return response


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
