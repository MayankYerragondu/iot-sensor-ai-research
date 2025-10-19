

resource "aws_sagemaker_training_job" "this" {
  name              = var.training_job_name
  role_arn          = var.sagemaker_role_arn
  algorithm_specification {
    training_input_mode = "File"
    training_image_name = var.training_image_name
  }

  input_data_config {
    channel_name = "training"
    data_source {
      s3_data_source {
        s3_data_type     = "S3Prefix"
        s3_uri           = var.input_s3_uri
        s3_data_distribution_type = "FullyReplicated"
      }
    }
  }

  output_data_config {
    s3_output_path = var.output_s3_uri
  }

  resource_config {
    instance_type  = var.instance_type
    instance_count = 1
    volume_size_in_gb = 10
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }

  hyperparameters = var.hyperparameters

  tags = {
    Project = "iot-ml-training"
  }
}
