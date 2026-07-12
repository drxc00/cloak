variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_access_key" {
  description = "AWS IAM access key for the terraform service account"
  type        = string
  sensitive   = true
  default     = "AKIA0123456789ABCDEF"
}

variable "aws_secret_key" {
  description = "AWS IAM secret key"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub PAT for Terraform provider authentication"
  type        = string
  sensitive   = true
  default     = "ghp_oNZd7mP3xKqR9vT2yB8wC5aE1fD4jH6lN0s"
}

variable "huggingface_token" {
  description = "HuggingFace API token for model registry access"
  type        = string
  sensitive   = true
  default     = "hf_xPqR7mN3vK5wQ9sT2yA8bC1dE4fG6hJ0lM9nOpQr"
}

variable "datadog_api_key" {
  description = "Datadog API key for metrics and APM"
  type        = string
  sensitive   = true
}

variable "grafana_service_account_token" {
  description = "Grafana service account token for dashboard provisioning"
  type        = string
  sensitive   = true
  default     = "glsa_abcdefghijklmnopqrstuvwxyz12345_a1b2c3d4"
}

variable "slack_webhook_url" {
  description = "Slack incoming webhook for Terraform notifications"
  type        = string
  sensitive   = true
}

variable "dockerhub_token" {
  description = "Docker Hub personal access token for image pulls"
  type        = string
  sensitive   = true
  default     = "dckr_pat_abcdefghijklmnopqrstuvwxyzABCDE"
}
