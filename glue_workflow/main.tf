variable "glue_role_arn" {}
variable "bucket_name" {}
variable "glue_script_path" {}
variable "cleaning_script_path" {}
variable "extract_job_name" {}
variable "cleaning_job_name" {}

resource "aws_glue_workflow" "iot_workflow" {
  name        = "iot-glue-workflow"
  description = "Workflow to run sensor extraction and then cleaning"
}

resource "aws_glue_trigger" "start_extract" {
  name          = "start-extract-sensors"
  type          = "ON_DEMAND"
  workflow_name = aws_glue_workflow.iot_workflow.name
  enabled       = true

  actions {
    job_name = var.extract_job_name
  }
}

resource "aws_glue_trigger" "run_cleaning_after_extract" {
  name          = "run-cleaning-after-extract"
  type          = "CONDITIONAL"
  workflow_name = aws_glue_workflow.iot_workflow.name
  enabled       = true

  predicate {
    conditions {
      job_name = var.extract_job_name
      state    = "SUCCEEDED"
    }
  }

  actions {
    job_name = var.cleaning_job_name
  }

  depends_on = [
    aws_glue_trigger.start_extract
  ]
}
