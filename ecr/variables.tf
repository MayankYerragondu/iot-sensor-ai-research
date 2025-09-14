variable "name" {
  description = "iot-image-repo"
  type        = string
}

variable "tags" {
  description = "Tags to apply to the repository"
  type        = map(string)
  default     = {}
}
