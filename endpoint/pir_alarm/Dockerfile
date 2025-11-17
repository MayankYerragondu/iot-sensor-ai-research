FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312

# (Optional) system deps you really need
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# (Optional) tweak TorchServe defaults by replacing the properties file the DLC uses:
# COPY etc/sagemaker-ts.properties /etc/sagemaker-ts.properties
# ^ Only if you need global timeouts/ports/workers changes.




# # Use a CPU-only PyTorch image
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312

# # Set working directory
# WORKDIR /opt/ml/code

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     curl \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install Python dependencies
# COPY code/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your inference code
# COPY code/inference.py .
# COPY code/handler.py .

# COPY code/model-config.yaml /opt/ml/model/config.yaml

# # Set environment variables for SageMaker
# ENV SAGEMAKER_PROGRAM inference.py
# ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
# # Ensure logs are flushed immediately
# ENV PYTHONUNBUFFERED 1  
# ENV TORCHSERVE_CONFIG /opt/ml/model/config.yaml

# # Define the entry point for SageMaker
# # ENTRYPOINT ["python", "inference.py"]