terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_s3_bucket" "iot_glue_bucket" {
  bucket        = "iot-glue-bucket-multi-model"
  force_destroy = true
}


resource "aws_s3_object" "glue_script" {
  bucket = aws_s3_bucket.iot_glue_bucket.id
  key    = "scripts/extract_sensors.py"
  source = "${path.module}/../glue_job_scripts/extract_sensors.py"
  etag   = filemd5("${path.module}/../glue_job_scripts/extract_sensors.py")
}

resource "aws_s3_object" "data_cleaning_script" {
  bucket = aws_s3_bucket.iot_glue_bucket.id
  key    = "scripts/data_cleaning.py"
  source = "${path.module}/../glue_job_scripts/data_cleaning.py"
  etag   = filemd5("${path.module}/../glue_job_scripts/data_cleaning.py")
}


output "bucket_name" {
  value = aws_s3_bucket.iot_glue_bucket.id
}

output "extract_script_path" {
  value = aws_s3_object.glue_script.key
}

output "cleaning_script_path" {
  value = aws_s3_object.data_cleaning_script.key
}