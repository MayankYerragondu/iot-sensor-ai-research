provider "aws" {
  region = "us-east-1"
}

module "s3" {
  source = "./s3"
}

module "iam" {
  source = "./iam"
}

module "glue_jobs" {
  source         = "./glue_jobs"
  bucket_name    = module.s3.bucket_name
  glue_role_arn  = module.iam.glue_role_arn
}

module "ecr_sagemaker" {
  source = "./ecr"
  name   = "sagemaker-custom-image"
  tags = {
    Project = "iot-glue-training"
  }
}

module "glue_workflow" {
  source                  = "./glue_workflow"
  glue_role_arn           = module.iam.glue_role_arn
  glue_script_path        = module.s3.extract_script_path
  cleaning_script_path    = module.s3.cleaning_script_path
  bucket_name             = module.s3.bucket_name
  extract_job_name        = module.glue_jobs.extract_job_name
  cleaning_job_name       = module.glue_jobs.cleaning_job_name
}



resource "aws_sfn_state_machine" "sagemaker_training" {
  name     = "sagemaker-training-job"
  role_arn = module.iam.stepfunction_role_arn

  depends_on = [
    module.iam
  ]
  definition = file("${path.module}/stepfunction/training_job.tmpl.json")
  type       = "STANDARD"
}
