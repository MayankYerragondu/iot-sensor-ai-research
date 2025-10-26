variable "training_job_name" {}
variable "sagemaker_role_arn" {}
variable "training_image_name" {}
variable "input_s3_uri" {}
variable "output_s3_uri" {}
variable "instance_type" {
  default = "ml.m5.large"
}
variable "hyperparameters" {
  type    = map(string)
  default = {}
}
