variable "bucket_name" {}
variable "glue_role_arn" {}

resource "aws_glue_job" "extract" {
  name     = "iot-sensor-extract"
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
  }

  glue_version      = "4.0"
  number_of_workers = 2
  worker_type       = "G.1X"
}

resource "aws_glue_job" "cleaning" {
  name     = "iot-data-cleaning"
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
  }

  glue_version      = "4.0"
  number_of_workers = 2
  worker_type       = "G.1X"
}
