from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import sagemaker

predictor = Predictor(
    endpoint_name="env-sensor-mme-endpoint-2",
    sagemaker_session=sagemaker.Session(),
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

payload = {
    "timestamp": "2025-10-13T09:30:00Z",
    "avg_temperature": 25.3,
    "avg_humidity": 48.2,
    "avg_lux": 180.5
}

resp = predictor.predict(payload, target_model="70:2c:1f:32:2c:54_zwave-c04d323e:3-2.tar.gz")
print(resp)
