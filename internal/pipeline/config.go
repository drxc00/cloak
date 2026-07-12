package pipeline

type Config struct {
	EnabledTypes     map[string]bool
	EntropyThreshold float64
	Thorough         bool // enables the LLM fallback stage
	DryRun           bool
}

func defaultConfig() *Config {
	return &Config{
		EnabledTypes: map[string]bool{
			"EMAIL": true, "PHONE": true, "MAC_ADDRESS": true,
			"IPv4": true, "IPv6": true, "SSN": true, "CREDIT_CARD": true,
			"JWT": true, "PRIVATE_KEY": true, "TOKEN": true,
			"API_KEY": true, "CREDENTIALS": true,

			// Secrets-stage vendor rules — keep in sync with secrets.Rules().
			"GITHUB_PAT": true, "GITHUB_FINE_GRAINED_PAT": true, "GITHUB_OAUTH": true,
			"GITHUB_APP_TOKEN": true, "GITHUB_REFRESH_TOKEN": true, "GITLAB_TOKEN": true,
			"AWS_ACCESS_KEY": true, "OPENAI_API_KEY": true, "ANTHROPIC_API_KEY": true,
			"OPENROUTER_API_KEY": true, "DEEPSEEK_API_KEY": true, "GROQ_API_KEY": true,
			"HUGGINGFACE_TOKEN": true, "PERPLEXITY_API_KEY": true,
			"NPM_ACCESS_TOKEN": true, "PYPI_API_TOKEN": true, "DOCKERHUB_PAT": true,
			"SLACK_BOT_TOKEN": true, "SLACK_USER_TOKEN": true, "SLACK_WEBHOOK_URL": true,
			"SENDGRID_API_KEY": true, "TWILIO_ACCOUNT_SID": true,
			"NOTION_INTEGRATION_TOKEN": true, "STRIPE_ACCESS_TOKEN": true,
			"HEROKU_API_KEY": true, "GRAFANA_SERVICE_ACCOUNT_TOKEN": true,
			"ATLASSIAN_API_TOKEN": true, "DOPPLER_TOKEN": true, "DB_CREDENTIALS": true,
			"REPLICATE_API_TOKEN": true,
			"DATADOG_API_KEY": true, "MAILGUN_API_KEY": true,
			"VAULT_TOKEN": true, "LINEAR_API_KEY": true, "AGE_SECRET_KEY": true,
			"GENERIC_API_KEY": true,
		},
		EntropyThreshold: 3.5,
	}
}

type Option func(*Config)

func WithDisabled(types ...string) Option {
	return func(c *Config) {
		for _, t := range types {
			c.EnabledTypes[t] = false
		}
	}
}

func WithThorough(v bool) Option { return func(c *Config) { c.Thorough = v } }
func WithDryRun(v bool) Option   { return func(c *Config) { c.DryRun = v } }

func NewConfig(opts ...Option) *Config {
	c := defaultConfig()
	for _, opt := range opts {
		opt(c)
	}
	return c
}
