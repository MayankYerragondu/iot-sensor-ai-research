import sagemaker
from sagemaker.model import Model
from sagemaker.multidatamodel import MultiDataModel
import boto3

role_arn = "arn:aws:iam::743634506477:role/service-role/AmazonSageMaker-ExecutionRole-20240326T111222"
region = "us-east-1"
session = sagemaker.Session()
sm_client = boto3.client("sagemaker", region_name=region)

endpoint_name = "env-sensor-mme-endpoint-2"
model_name = "env-sensor-mme-2"

# 1️⃣ Cleanup old endpoint if it exists
for fn, param in [
    (sm_client.delete_endpoint, {"EndpointName": endpoint_name}),
    (sm_client.delete_endpoint_config, {"EndpointConfigName": endpoint_name}),
    (sm_client.delete_model, {"ModelName": model_name}),
]:
    try:
        fn(**param)
    except sm_client.exceptions.ClientError:
        pass

# 2️⃣ Define your base model using your custom Docker image
custom_image = "743634506477.dkr.ecr.us-east-1.amazonaws.com/iot-env-sensor-sagemaker-image:latest"

base_model = Model(
    image_uri=custom_image,
    role=role_arn,
    entry_point="inference.py",
    env={"SAGEMAKER_PROGRAM": "inference.py"},
    sagemaker_session=session,
)

# 3️⃣ Define your multi-model endpoint
mme = MultiDataModel(
    name=model_name,
    model_data_prefix="s3://iot-glue-bucket-multi-model/endpoint/env_sensor/",  # path to your .tar.gz models
    model=base_model,
    sagemaker_session=session,
)

# 4️⃣ Deploy to SageMaker
predictor = mme.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name=endpoint_name,
    update_endpoint=True,
)

print("✅ Endpoint deployed:", endpoint_name)
