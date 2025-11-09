from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import sagemaker 

pred = Predictor(
    endpoint_name="contact-alarm-mme-endpoint",
    sagemaker_session=sagemaker.Session(),
    serializer=JSONSerializer(),          # sets ContentType: application/json
    deserializer=JSONDeserializer(),      # sets Accept: application/json
)

payload = {"timestamp": "2025-08-21T05:59:00Z"}
resp = pred.predict(
    payload,
    target_model="70:2c:1f:3b:df:6c_zwave-ed217efc:20-0.tar.gz",  # use a colon-free filename
)
print(resp)
