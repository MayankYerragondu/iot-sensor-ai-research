#!/bin/bash
set -e

echo "Running Terraform plan and apply..."
terraform init
terraform apply -auto-approve

# Fetch the bucket name from Terraform output AFTER apply
EXTRACTION_SCRIPT_PATH="./glue_job_scripts/extract_sensors.py"
CLEANING_SCRIPT_PATH="./glue_job_scripts/data_cleaning.py"
PIR_ALARM_TRAINING_SCRIPT_PATH="./pir_alarm/isolation_forest.py"
CONTACT_ALARM_TRAINING_SCRIPT_PATH="./contact_alarm/one_class_svm.py"
ENV_SENSOR_TRAINING_SCRIPT_PATH="./env_sensor/LSTM_model.py"


echo "=== Fetching Terraform Outputs ==="
ECR_IMAGE_URI=$(terraform output -raw repository_url)
SAGEMAKER_ROLE_ARN=$(terraform output -raw sagemaker_execution_role_arn)
STEP_FUNCTION_ARN=$(terraform output -raw sagemaker_stepfunction_arn)
BUCKET_NAME=$(terraform output -raw iot_glue_bucket)

echo "Uploading extract_sensors.py to S3..."
aws s3 cp "$EXTRACTION_SCRIPT_PATH" "s3://${BUCKET_NAME}/scripts/extract_sensors.py"

echo "Uploading data_cleaning.py to S3..."
aws s3 cp "$CLEANING_SCRIPT_PATH" "s3://${BUCKET_NAME}/scripts/data_cleaning.py"

echo "Uploading isolation_forest.py to S3..."
aws s3 cp "$PIR_ALARM_TRAINING_SCRIPT_PATH" "s3://${BUCKET_NAME}/scripts/isolation_forest.py"

echo "Uploading one_class_svm.py to S3..."
aws s3 cp "$CONTACT_ALARM_TRAINING_SCRIPT_PATH" "s3://${BUCKET_NAME}/scripts/one_class_svm.py"

echo "Uploading LSTM_model.py to S3..."
aws s3 cp "$ENV_SENSOR_TRAINING_SCRIPT_PATH" "s3://${BUCKET_NAME}/scripts/LSTM_model.py"



# === Build and push Docker image to ECR ===
echo "=== Logging into ECR and pushing custom image ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${ECR_IMAGE_URI%/*}"

docker build -t sagemaker-custom-image:latest .
docker tag sagemaker-custom-image:latest "$ECR_IMAGE_URI"
docker push "$ECR_IMAGE_URI"

# === Trigger SageMaker Training via Step Function ===
echo "=== Starting Step Function execution for SageMaker Training ==="

aws stepfunctions start-execution \
  --state-machine-arn "$STEP_FUNCTION_ARN" \
  --input "{
    \"TrainingJobName\": \"glue-sagemaker-flow-$(date +%s)\",
    \"TrainingImage\": \"$ECR_IMAGE_URI\",
    \"SageMakerRoleArn\": \"$SAGEMAKER_ROLE_ARN\",
    \"InputS3Uri\": \"s3://${BUCKET_NAME}/output/cleaned/\",
    \"OutputS3Uri\": \"s3://${BUCKET_NAME}/model/\"
  }"

echo "✅ Deploy completed."


