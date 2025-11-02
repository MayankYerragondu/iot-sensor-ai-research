docker build -t iot-env-sensor-sagemaker-image .                                                     
docker tag iot-env-sensor-sagemaker-image:latest 743634506477.dkr.ecr.us-east-1.amazonaws.com/iot-env-sensor-sagemaker-image:latest
docker push 743634506477.dkr.ecr.us-east-1.amazonaws.com/iot-env-sensor-sagemaker-image:latest     


sh package_tar.sh
python endpoint_deploy.py