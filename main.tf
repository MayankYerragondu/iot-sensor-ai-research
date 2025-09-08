provider "aws" {
  region = "us-east-1"
}

module "s3" {
  source = "./s3"
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
