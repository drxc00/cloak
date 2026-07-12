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
			"NAME": true, "EMAIL": true, "PHONE": true,
			"MAC_ADDRESS": true, "API_KEY": true, "PASSWORD": true,
			"USERNAME": true, "CREDENTIALS": true, "HOSTNAME": true,
			"SSN": true, "CREDIT_CARD": true, "JWT": true, "IBAN": true, "IPv4": true, "IPv6": true,
			"ADDRESS": true, "PRIVATE_KEY": true, "TOKEN": true,
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
