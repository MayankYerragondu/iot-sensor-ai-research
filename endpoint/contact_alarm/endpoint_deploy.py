import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.multidatamodel import MultiDataModel
import boto3

role_arn = "arn:aws:iam::743634506477:role/service-role/AmazonSageMaker-ExecutionRole-20240326T111222"
session = sagemaker.Session()
region = session.boto_region_name

sm = boto3.client("sagemaker")

endpoint_name = "contact-alarm-mme-endpoint"
model_name    = "contact-alarm-mme"
endpoint_config_name = endpoint_name  # or whatever name you used

# 1. Delete the endpoint (if it exists)
try:
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"Deleted endpoint: {endpoint_name}")
except sm.exceptions.ClientError as e:
    print(f"Endpoint not found or already deleted: {endpoint_name}")

# 2. Delete the endpoint config
try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print(f"Deleted endpoint config: {endpoint_config_name}")
except sm.exceptions.ClientError as e:
    print(f"Endpoint config not found or already deleted: {endpoint_config_name}")

# 3. Delete the model
try:
    sm.delete_model(ModelName=model_name)
    print(f"Deleted model: {model_name}")
except sm.exceptions.ClientError as e:
    print(f"Model not found or already deleted: {model_name}")


# 1) Choose the sklearn inference image you’re using
image = image_uris.retrieve(framework="sklearn", region=region, version="1.2-1", py_version="py3")

# 2) Base container with your shared inference code (one copy for all models)
base_model = Model(
    image_uri=image,
    role=role_arn,
    entry_point="inference.py",        # your handler (Predictor uses this)
    env={"SAGEMAKER_PROGRAM": "inference.py"},
    sagemaker_session=session,
)

# 3) Multi-model wrapper that points to the S3 PREFIX containing many model tarballs
mme = MultiDataModel(
    name=model_name,
    model_data_prefix="s3://iot-glue-bucket-multi-model/endpoint/contact_alarm/",  # <-- prefix (no trailing slash needed)
    model=base_model,
    sagemaker_session=session,
)

# 4) Deploy the MME (pick a supported instance; m5 is a safe minimum for sklearn)
predictor = mme.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name,
    update_endpoint=True,   
)
