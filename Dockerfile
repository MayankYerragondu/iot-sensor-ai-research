# Use AWS-provided SageMaker PyTorch container as base (even if you're not using PyTorch)
# FROM public.ecr.aws/sagemaker/sagemaker-distribution:latest-cpu

# # Install the SageMaker Training Toolkit for script mode support
# RUN pip install --no-cache-dir sagemaker-training

FROM --platform=linux/amd64 python:3.8-slim

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    boto3 \
    tensorflow

ENV PYTHONDONTWRITEBYTECODE=1

# Create working directory for SageMaker code
WORKDIR /opt/ml/code


# Create the necessary directory structure first
RUN mkdir -p script/contact_alarm \
    && mkdir -p script/pir_alarm \
    && mkdir -p script/env_sensor

# Copy only the required files
COPY training_script_entry.py .
COPY script/contact_alarm/one_class_svm.py script/contact_alarm/one_class_svm.py
COPY script/contact_alarm/__init__.py script/contact_alarm/__init__.py

COPY script/pir_alarm/isolation_forest.py script/pir_alarm/isolation_forest.py
COPY script/pir_alarm/__init__.py script/pir_alarm/__init__.py

COPY script/env_sensor/LSTM_model.py script/env_sensor/LSTM_model.py
COPY script/env_sensor/__init__.py script/env_sensor/__init__.py

# Define the script SageMaker should run
ENV SAGEMAKER_PROGRAM training_script_entry.py

# Use exec form ENTRYPOINT for signal handling (recommended)
ENTRYPOINT ["python", "training_script_entry.py"]
