from openai import OpenAI
import os
import json
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import sagemaker


# --- OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- SageMaker Predictor Setup ---
pred = Predictor(
    endpoint_name="piralarm-mme-endpoint-2",
    sagemaker_session=sagemaker.Session(),
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)


# --- Query anomaly score ---
def query_anomaly_score(device_id: str, timestamp: str):
    """
    Calls your SageMaker multi-model endpoint
    and returns anomaly score + raw response.
    """
    try:
        # Convert device ID for the MME target_model name
        mm_filename = device_id.replace(":", "") + "_zwave-d96c234a:33-10.tar.gz"

        payload = {"timestamp": timestamp}

        resp = pred.predict(
            payload,
            target_model=mm_filename,
        )
        return resp
    except Exception as e:
        return {"error": str(e)}


# --- LLM Explanation ---
def handle_user_query(user_input: str, device_id: str, timestamp: str):
    # 1. Call SageMaker for anomaly score
    api_result = query_anomaly_score(device_id, timestamp)

    # 2. Build explanation prompt
    prompt = f"""
You are a smart-home IoT assistant.

User Question:
"{user_input}"

Sensor Model Result:
{json.dumps(api_result, indent=2)}

Explain:
- What the anomaly score means
- Whether it looks normal or suspicious
- Possible reasons for the reading
- Any recommendations to the user
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful IoT assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=600
    )

    return resp.choices[0].message.content.strip()


# --- Test Example ---
if __name__ == "__main__":
    answer = handle_user_query(
        user_input="Why did the sensor trigger at this time?",
        device_id="70:2c:1f:32:1a:e4",
        timestamp="2025-08-21T05:59:00Z"
    )
    print(answer)
