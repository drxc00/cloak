package secrets

import "regexp"

// genericSecretRegex builds the shared key[=:]value pattern used by the
// keyword-driven rules (PASSWORD, GENERIC_API_KEY), parameterized by the
// key-stem alternation so each rule can scope its own keyword set.
func genericSecretRegex(keyStems string) *regexp.Regexp {
	return regexp.MustCompile(
		`(?i)[\w.-]{0,50}?(?:` + keyStems + `)[\w.-]{0,50}(?:[ \t\w.-]{0,20})[\s'"]{0,3}` +
			`(?:=|>|:{1,3}=|\|\||:|=>|\?=|,)` +
			`[\s='"]{0,5}` +
			`([\w.=-]{10,150}|[a-z0-9][a-z0-9+/]{11,}={0,3})` +
			`(?:[\s'";,.!?)\]}]|\\[nr]|$)`,
	)
}

func Rules() []Rule {
	return []Rule{
		{
			Type:        "GITHUB_PAT",
			Keywords:    []string{"ghp_"},
			Regex:       regexp.MustCompile(`\bghp_[0-9a-zA-Z]{10,40}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
			Allowlist: &Allowlist{
				ExactSecrets: setOf("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
			},
		},
		{
			Type:        "GITHUB_FINE_GRAINED_PAT",
			Keywords:    []string{"github_pat_"},
			Regex:       regexp.MustCompile(`\bgithub_pat_\w{10,}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "GITHUB_OAUTH",
			Keywords:    []string{"gho_"},
			Regex:       regexp.MustCompile(`\bgho_[0-9a-zA-Z]{36}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "GITHUB_APP_TOKEN",
			Keywords:    []string{"ghu_", "ghs_"},
			Regex:       regexp.MustCompile(`\b(?:ghu|ghs)_[0-9a-zA-Z]{36}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "GITHUB_REFRESH_TOKEN",
			Keywords:    []string{"ghr_"},
			Regex:       regexp.MustCompile(`\bghr_[0-9a-zA-Z]{36}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		// GitLab issues several token types beyond just personal access
		// tokens now: deploy tokens, runner registration tokens, and
		// agent tokens all share the glXXX- prefix family.
		{
			Type:        "GITLAB_TOKEN",
			Keywords:    []string{"glpat-", "gldt-", "glrt-", "glagent-"},
			Regex:       regexp.MustCompile(`\b(?:glpat|gldt|glrt|glagent)-[0-9a-zA-Z_\-]{10,50}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "AWS_ACCESS_KEY",
			Keywords:    []string{"akia", "asia", "abia", "acca", "a3t"},
			Regex:       regexp.MustCompile(`\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\b`),
			SecretGroup: 1,
			MinEntropy:  3.0,
			// Allowlist: &Allowlist{
			// 	Regexes:      []*regexp.Regexp{regexp.MustCompile(`(?i)EXAMPLE$`)},
			// 	ExactSecrets: setOf("AKIAIOSFODNN7EXAMPLE"),
			// },
		},
		{
			// OpenAI: legacy `sk-` keys and newer proj/svcacct/admin keys
			// all embed the literal marker T3BlbkFJ mid-token (base64 for
			// "OpenAI") — anchoring on that marker is far more reliable
			// than matching on prefix + length alone, since it holds
			// across every key generation OpenAI has shipped.
			Type:        "OPENAI_API_KEY",
			Keywords:    []string{"t3blbkfj"},
			Regex:       regexp.MustCompile(`\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,74}T3BlbkFJ[A-Za-z0-9_-]{20,74}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// Anthropic: sk-ant-api03- (standard key), sk-ant-oat01-
			// (OAuth token from first-party CLI tools), sk-ant-admin01-
			Type:        "ANTHROPIC_API_KEY",
			Keywords:    []string{"sk-ant-"},
			Regex:       regexp.MustCompile(`\bsk-ant-(?:api03|oat01|admin01)-[A-Za-z0-9_-]{10,}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "OPENROUTER_API_KEY",
			Keywords:    []string{"sk-or-v1-"},
			Regex:       regexp.MustCompile(`\bsk-or-v1-[a-f0-9]{64}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// DeepSeek collides fully with legacy OpenAI's bare sk-
			// shape, so the keyword gate is load-bearing here without
			// "deepseek" nearby, don't trust this pattern alone.
			Type:        "DEEPSEEK_API_KEY",
			Keywords:    []string{"deepseek"},
			Regex:       regexp.MustCompile(`\bsk-[a-f0-9]{32}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "GROQ_API_KEY",
			Keywords:    []string{"gsk_"},
			Regex:       regexp.MustCompile(`\bgsk_[A-Za-z0-9]{48,52}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "HUGGINGFACE_TOKEN",
			Keywords:    []string{"hf_"},
			Regex:       regexp.MustCompile(`\bhf_[A-Za-z0-9]{10,40}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "PERPLEXITY_API_KEY",
			Keywords:    []string{"pplx-"},
			Regex:       regexp.MustCompile(`\bpplx-[A-Za-z0-9]{10,}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "REPLICATE_API_TOKEN",
			Keywords:    []string{"r8_"},
			Regex:       regexp.MustCompile(`\br8_[A-Za-z0-9]{10,40}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "NPM_ACCESS_TOKEN",
			Keywords:    []string{"npm_"},
			Regex:       regexp.MustCompile(`\bnpm_[A-Za-z0-9]{36}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "PYPI_API_TOKEN",
			Keywords:    []string{"pypi-agei"},
			Regex:       regexp.MustCompile(`\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{10,}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "DOCKERHUB_PAT",
			Keywords:    []string{"dckr_pat_"},
			Regex:       regexp.MustCompile(`\bdckr_pat_[A-Za-z0-9_\-]{10,64}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "SLACK_BOT_TOKEN",
			Keywords:    []string{"xoxb-"},
			Regex:       regexp.MustCompile(`\bxoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*\b`),
			SecretGroup: 0,
			MinEntropy:  2.0,
		},
		{
			Type:        "SLACK_USER_TOKEN",
			Keywords:    []string{"xoxp-", "xoxe-"},
			Regex:       regexp.MustCompile(`\bxox[pe](?:-[0-9]{10,13}){3}-[a-zA-Z0-9-]{28,34}\b`),
			SecretGroup: 0,
			MinEntropy:  2.0,
		},
		{
			Type:        "SLACK_WEBHOOK_URL",
			Keywords:    []string{"hooks.slack.com"},
			Regex:       regexp.MustCompile(`hooks\.slack\.com/(?:services|workflows|triggers)/[A-Za-z0-9+/]{43,135}`),
			SecretGroup: 0,
			MinEntropy:  2.0,
		},
		{
			Type:        "SENDGRID_API_KEY",
			Keywords:    []string{"sg."},
			Regex:       regexp.MustCompile(`\bSG\.[A-Za-z0-9_\-]{18,30}\.[A-Za-z0-9_\-]{35,55}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// Twilio's Account SID has a real prefix; the paired Auth
			// Token doesn't, so we only reliably catch the SID here
			// the bare token needs the generic rule + "auth_token"
			// context keyword to have a chance.
			Type:        "TWILIO_ACCOUNT_SID",
			Keywords:    []string{"twilio"},
			Regex:       regexp.MustCompile(`\bAC[a-f0-9]{32}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "NOTION_INTEGRATION_TOKEN",
			Keywords:    []string{"secret_", "ntn_"},
			Regex:       regexp.MustCompile(`\b(?:secret_|ntn_)[A-Za-z0-9]{43,50}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "STRIPE_ACCESS_TOKEN",
			Keywords:    []string{"sk_test", "sk_live", "sk_prod", "rk_test", "rk_live", "rk_prod"},
			Regex:       regexp.MustCompile(`\b(?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99}\b`),
			SecretGroup: 0,
			MinEntropy:  2.0,
		},
		{
			Type: "HEROKU_API_KEY",
			// Heroku keys are bare UUIDv4s with no distinguishing prefix,
			// so a "heroku"-adjacent keyword plus the UUID shape is the
			// only reliable signal available.
			Keywords:    []string{"heroku"},
			Regex:       regexp.MustCompile(`(?i)heroku[a-z0-9_\-]{0,20}\s*[:=]\s*['"]?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})['"]?`),
			SecretGroup: 1,
			MinEntropy:  3.0,
		},
		{
			// Grafana's the rare one with an actual trailing checksum —
			// worth validating in code the same way you validate credit
			// cards with Luhn, rather than trusting the regex alone.
			Type:        "GRAFANA_SERVICE_ACCOUNT_TOKEN",
			Keywords:    []string{"glsa_"},
			Regex:       regexp.MustCompile(`\bglsa_[A-Za-z0-9]{32}_[a-f0-9]{8}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "ATLASSIAN_API_TOKEN",
			Keywords:    []string{"atatt3"},
			Regex:       regexp.MustCompile(`\bATATT3[A-Za-z0-9_=\-]{100,300}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "DOPPLER_TOKEN",
			Keywords:    []string{"dp.st.", "dp.sa.", "dp.ct."},
			Regex:       regexp.MustCompile(`\bdp\.(?:st|sa|ct)\.[A-Za-z0-9]{40,44}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// Datadog issues two key flavours: API keys start with 'pub'
			// and application keys start with 'pvt'. Both have 32 hex
			// chars after the prefix. Require "datadog" context or the
			// exact prefix to avoid matching "public" and "pvtool".
			Type:        "DATADOG_API_KEY",
			Keywords:    []string{"datadog", "dd_api_key", "dd_app_key", "pub", "pvt"},
			Regex:       regexp.MustCompile(`\b(?:pub|pvt)[a-f0-9]{32}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// Mailgun: API keys start with key- followed by alphanum;
			// signing keys are bare hex strings drawn from the same
			// account page, so they need "mailgun" proximity to fire.
			Type:        "MAILGUN_API_KEY",
			Keywords:    []string{"key-", "mailgun"},
			Regex:       regexp.MustCompile(`\bkey-[a-f0-9]{32}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// HashiCorp Vault: tokens can be service (`hvs.`) or batch
			// (`hvb.`). Both are prefix-dot-base62 strings.
			Type:        "VAULT_TOKEN",
			Keywords:    []string{"hvs.", "hvb."},
			Regex:       regexp.MustCompile(`\bhv[bs]\.[A-Za-z0-9]{24,100}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "LINEAR_API_KEY",
			Keywords:    []string{"lin_api_"},
			Regex:       regexp.MustCompile(`\blin_api_[A-Za-z0-9]{40}\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			// Age encryption: AGE-SECRET-KEY-1 followed by base64.
			// The Bech32-encoded private key variant (age1...) has a
			// different shape — it starts after AGE-SECRET-KEY-1...
			Type:        "AGE_SECRET_KEY",
			Keywords:    []string{"age-secret-key-1"},
			Regex:       regexp.MustCompile(`\bAGE-SECRET-KEY-1[A-Za-z0-9+/]{40,80}(?:={0,2})?\b`),
			SecretGroup: 0,
			MinEntropy:  3.0,
		},
		{
			Type:        "PASSWORD",
			Keywords:    []string{"password", "passwd", "pass"},
			Regex:       genericSecretRegex(`pass(?:w(?:or)?d)?`),
			SecretGroup: 1,
			MinEntropy:  3.0,
			Allowlist:   genericAllowlist(),
		},
		{
			Type: "GENERIC_API_KEY",
			Keywords: []string{
				"api", "key", "token", "secret",
				"credential", "creds", "auth", "access",
			},
			Regex:       genericSecretRegex(`access|auth|api|credential|creds|key|secret|token`),
			SecretGroup: 1,
			MinEntropy:  3.0,
			Allowlist:   genericAllowlist(),
		},
		{
			Type:        "DB_CREDENTIALS",
			Keywords:    []string{"postgres://", "mysql://"},
			Regex:       regexp.MustCompile(`(\w+)://([\w.-]+):([\w.-]+)@`),
			SecretGroup: 3, // username and password, not the scheme
		},
	}
}
