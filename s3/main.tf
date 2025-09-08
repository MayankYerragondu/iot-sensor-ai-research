resource "aws_s3_bucket" "iot_glue_bucket" {
  bucket        = "iot-glue-bucket-multi-model"
  force_destroy = false
}

resource "aws_s3_object" "glue_script" {
  bucket = aws_s3_bucket.iot_glue_bucket.id
  key    = "scripts/extract_sensors.py"
  source = "${path.module}/../glue_job/extract_sensors.py"
  etag   = filemd5("${path.module}/../glue_job/extract_sensors.py")
}

resource "aws_s3_object" "data_cleaning_script" {
  bucket = aws_s3_bucket.iot_glue_bucket.id
  key    = "scripts/data_cleaning.py"
  source = "${path.module}/../glue_job/data_cleaning.py"
  etag   = filemd5("${path.module}/../glue_job/data_cleaning.py")
}
