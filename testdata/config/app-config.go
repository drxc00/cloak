package config

import (
	"fmt"
	"os"
	"time"
)

type AppConfig struct {
	Port         int
	ReadTimeout  time.Duration
	WriteTimeout time.Duration

	DBHost     string
	DBPort     int
	DBName     string
	DBUser     string
	DBPassword string
	DBSSLMode  string

	RedisURL      string
	RedisPassword string

	GitHubToken        string
	GitHubPackageToken string
	GitLabDeployToken  string
	OpenAIKey          string
	AnthropicAPIKey    string
	HuggingFaceToken   string
	DeepSeekKey        string
	GroqKey            string
	AwsAccessKeyID     string
	AwsSecretAccessKey string
	SlackBotToken      string
	SlackUserToken     string
	SlackWebhookURL    string
	StripeSecretKey    string
	SendGridAPIKey     string
	TwilioAccountSID   string
	GrafanaToken       string
	DockerHubToken     string
	JWTSecret          string
	ServiceAPIToken    string
}

func Load() (*AppConfig, error) {
	cfg := &AppConfig{
		Port:               8080,
		ReadTimeout:        30 * time.Second,
		WriteTimeout:       30 * time.Second,
		DBHost:             envOrDefault("DB_HOST", "db-primary.internal.corp"),
		DBPort:             5432,
		DBName:             envOrDefault("DB_NAME", "app_production"),
		DBUser:             envOrDefault("DB_USER", "app_prod_user"),
		DBPassword:         os.Getenv("DB_PASSWORD"),
		DBSSLMode:          "verify-full",
		RedisURL:           envOrDefault("REDIS_URL", "redis://redis-cluster.internal.corp:6379/0"),
		RedisPassword:      os.Getenv("REDIS_PASSWORD"),
		GitHubToken:        os.Getenv("GITHUB_TOKEN"),
		GitHubPackageToken: os.Getenv("GITHUB_PACKAGE_TOKEN"),
		GitLabDeployToken:  os.Getenv("GITLAB_DEPLOY_TOKEN"),
		OpenAIKey:          os.Getenv("OPENAI_API_KEY"),
		AnthropicAPIKey:    os.Getenv("ANTHROPIC_API_KEY"),
		HuggingFaceToken:   os.Getenv("HUGGINGFACE_TOKEN"),
		DeepSeekKey:        os.Getenv("DEEPSEEK_API_KEY"),
		GroqKey:            os.Getenv("GROQ_API_KEY"),
		AwsAccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		AwsSecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		SlackBotToken:      os.Getenv("SLACK_BOT_TOKEN"),
		SlackUserToken:     os.Getenv("SLACK_USER_TOKEN"),
		SlackWebhookURL:    os.Getenv("SLACK_WEBHOOK_URL"),
		StripeSecretKey:    os.Getenv("STRIPE_SECRET_KEY"),
		SendGridAPIKey:     os.Getenv("SENDGRID_API_KEY"),
		TwilioAccountSID:   os.Getenv("TWILIO_ACCOUNT_SID"),
		GrafanaToken:       os.Getenv("GRAFANA_TOKEN"),
		DockerHubToken:     os.Getenv("DOCKERHUB_TOKEN"),
		JWTSecret:          os.Getenv("JWT_SECRET"),
		ServiceAPIToken:    os.Getenv("SERVICE_API_TOKEN"),
	}

	// Validate required secrets are present.
	if cfg.DBPassword == "" {
		return nil, fmt.Errorf("DB_PASSWORD is required")
	}
	if cfg.GitHubToken == "" {
		return nil, fmt.Errorf("GITHUB_TOKEN is required")
	}

	return cfg, nil
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
