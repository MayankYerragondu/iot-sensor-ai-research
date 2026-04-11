print("✅ inference.py loading modules")

print("✅ inference.py loading modules")

try:
    import os
    import json
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import joblib
    print("✅ All modules imported successfully")

except Exception as e:
    import traceback
    print("❌ Import error in inference.py:", e)
    traceback.print_exc()

print("✅ inference.py loaded")


import os
print("✅ inference.py loaded, listing model directory base:")
if os.path.exists("/opt/ml/models"):
    print("Contents of /opt/ml/models:", os.listdir("/opt/ml/models"))
else:
    print("/opt/ml/models does not exist yet.")


def model_fn(model_dir):
    print(f"🔍 model_fn: model_dir = {model_dir}")

    # Try the nested "model" subdirectory first
    nested = os.path.join(model_dir, "model")
    if os.path.isdir(nested):
        target = nested
    else:
        target = model_dir
    print(f"📂 Using directory: {target}")

    # Walk through the directory to locate .h5 and optional scaler
    model_path = None
    scaler_path = None
    for root, _, files in os.walk(target):
        for fname in files:
            if fname.endswith(".h5"):
                model_path = os.path.join(root, fname)
            elif fname.endswith("_scaler.joblib"):
                scaler_path = os.path.join(root, fname)

    if model_path is None:
        raise FileNotFoundError(f"No .h5 model file found under {target}")

    print(f"✅ Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    scaler = None
    if scaler_path:
        print(f"✅ Loading scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

    return {"model": model, "scaler": scaler}


def input_fn(request_body, content_type):
    print(f"📥 input_fn: content_type = {content_type}")
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    ts = pd.to_datetime(data["timestamp"], utc=True)
    features = {
        "avg_temperature": float(data["avg_temperature"]),
        "avg_humidity": float(data["avg_humidity"]),
        "avg_lux": float(data["avg_lux"]),
        "month_of_day": ts.month,
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
    }
    df = pd.DataFrame([features])
    print(f"✅ Parsed features: {df.to_dict(orient='records')}")
    return df


def predict_fn(input_data, model_and_scaler):
    model = model_and_scaler["model"]
    scaler = model_and_scaler.get("scaler", None)

    cols = ["avg_temperature", "avg_humidity", "avg_lux", "month_of_day", "hour_of_day", "day_of_week"]
    X = input_data[cols].values.astype(float)

    if scaler:
        X = scaler.transform(X)

    # Build a time-series shape, e.g. replicate across timesteps
    X_seq = np.expand_dims(np.tile(X, (10, 1)), axis=0)  # shape (1, 10, n_features)
    reconstructed = model.predict(X_seq, verbose=0)

    # Compute MSE on first 3 features (temp, humidity, lux)
    mse = np.mean(np.square(X_seq[:, :, :3] - reconstructed), axis=(1, 2))
    is_anomaly = (mse > 0.01).astype(int).tolist()

    # Compute additional metrics if needed
    result = {
        "reconstruction_error": mse.tolist(),
        "is_anomaly": is_anomaly
    }
    # Log the prediction results for debugging
    print(f"🔎 result = {result}")
    return result


# Output function to format the prediction result as JSON
def output_fn(prediction, accept):
    resp = json.dumps(prediction)
    print(f"📤 output = {resp}")
    return resp


# Main handler function that SageMaker will call
def handler(data, context):
    try:
        print("🚀 handler invoked")
        # Load the model and scaler
        model_ctx = model_fn(context.model_dir)
        # Log the raw input data for debugging
        inp = input_fn(data.read().decode("utf-8"), context.request_content_type)
        # Log the parsed input data for debugging
        print(f"✅ Parsed input data: {inp.to_dict(orient='records')}")
        preds = predict_fn(inp, model_ctx)
        return output_fn(preds, context.accept_header)
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        raise
