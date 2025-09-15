data "aws_caller_identity" "current" {}


resource "aws_iam_role" "glue_role" {
  name = "iot-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "glue.amazonaws.com" },
      Action    = ["sts:AssumeRole"]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_s3" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role" "stepfunction" {
  name = "stepfunction_sagemaker_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = { Service = "states.amazonaws.com" },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "stepfunction_policy" {
  name = "StepFunctionSageMakerPolicy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:AddTags",                // Add this line
          "s3:GetObject",
          "s3:PutObject",
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_stepfunction_policy" {
  role       = aws_iam_role.stepfunction.name
  policy_arn = aws_iam_policy.stepfunction_policy.arn
}

resource "aws_iam_role" "sagemaker_execution" {
  name = "sagemaker_execution_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "sagemaker.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  depends_on = [aws_iam_role.sagemaker_execution]
}

resource "aws_iam_policy" "sagemaker_pull_policy" {
  name = "SageMakerECRPullPolicy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_pull_policy_attachment" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_pull_policy.arn
  depends_on = [aws_iam_role.sagemaker_execution]
}

resource "aws_iam_policy" "stepfunction_events_policy" {
  name        = "StepFunctionEventsPolicy"
  description = "Additional Step Functions permissions (if needed)"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          # Add other permissions if needed, or leave empty if replaced
          "states:StartExecution",
          "states:DescribeExecution",
          "events:PutRule",
          "events:PutTargets",
          "events:DescribeRule",
          "events:DeleteRule",
          "events:RemoveTargets"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_stepfunction_events_policy" {
  role       = aws_iam_role.stepfunction.name
  policy_arn = aws_iam_policy.stepfunction_events_policy.arn
}

data "aws_iam_policy_document" "pass_sagemaker_role" {
  statement {
    effect = "Allow"
    actions = ["iam:PassRole"]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/sagemaker_execution_role"
    ]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "stepfunction_passrole_policy" {
  name   = "StepFunctionPassSageMakerExecutionRole"
  policy = data.aws_iam_policy_document.pass_sagemaker_role.json
}

resource "aws_iam_role_policy_attachment" "attach_passrole" {
  role       = aws_iam_role.stepfunction.name
  policy_arn = aws_iam_policy.stepfunction_passrole_policy.arn
}

resource "aws_iam_policy" "stepfunction_glue_workflow_policy" {
  name = "StepFunctionGlueWorkflowPolicy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "glue:StartWorkflowRun",
          "glue:GetWorkflowRun",
          "glue:GetWorkflow",
          "glue:ListWorkflows"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_stepfunction_glue_policy" {
  role       = aws_iam_role.stepfunction.name
  policy_arn = aws_iam_policy.stepfunction_glue_workflow_policy.arn
}



output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}


output "stepfunction_role_arn" {
  value = aws_iam_role.stepfunction.arn
}

output "glue_role_arn" {
  value = aws_iam_role.glue_role.arn
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}
