# TensorFlow 2.19 inference base image (AWS DLC)
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-cpu

# Install extra dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.19.0 \
    numpy \
    pandas \
    scikit-learn==1.5.2 \
    joblib \
    boto3

# Clean up caches
RUN rm -rf /root/.cache /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy your inference script
COPY inference.py .

# SageMaker entry point
ENV SAGEMAKER_PROGRAM=inference.py
