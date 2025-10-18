
output "repository_url" {
  value = module.ecr_sagemaker.repository_url
}

output "iot_glue_bucket" {
  value = module.s3.bucket_name
}

output "sagemaker_custom_image_uri" {
  value = "${module.ecr_sagemaker.repository_url}:latest"
}

output "sagemaker_execution_role_arn" {
  value = module.iam.sagemaker_execution_role_arn
}

output "sagemaker_stepfunction_arn" {
  value = aws_sfn_state_machine.sagemaker_training.arn
}