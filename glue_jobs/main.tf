terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "bucket_name" {}
variable "glue_role_arn" {}

resource "aws_glue_job" "extract" {
  name     = "iot-sensor-extract-1"
  role_arn = var.glue_role_arn

  command {
    name            = "glueetl"
    script_location = "s3://${var.bucket_name}/scripts/extract_sensors.py"
    python_version  = "3"
  }

  default_arguments = {
    "--TempDir"      = "s3://${var.bucket_name}/temp/"
    "--job-language" = "python"
    "--output_bucket"  = "${var.bucket_name}/output/"
    "--bucket_name"  = var.bucket_name
  }

  glue_version      = "4.0"
  number_of_workers = 2
  worker_type       = "G.1X"
}

resource "aws_glue_job" "cleaning" {
  name     = "iot-data-cleaning-1"
  role_arn = var.glue_role_arn

  command {
    name            = "glueetl"
    script_location = "s3://${var.bucket_name}/scripts/data_cleaning.py"
    python_version  = "3"
  }

  default_arguments = {
    "--TempDir"      = "s3://${var.bucket_name}/temp/"
    "--job-language" = "python"
    "--output_bucket"  = "${var.bucket_name}/output/"
    "--bucket_name"  = var.bucket_name
  }

  glue_version      = "4.0"
  number_of_workers = 2
  worker_type       = "G.1X"
}

output "extract_job_name" {
  value = aws_glue_job.extract.name
}

output "cleaning_job_name" {
  value = aws_glue_job.cleaning.name
}