# =============================================================================
# README.md — Internal onboarding documentation
# =============================================================================
# ⚠️  Example credentials in this doc are placeholders for illustration.
#     cloak should NOT redact documented examples — only real-looking values.
# =============================================================================

## Getting Started

1. Clone the repository
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Fill in your credentials — contact #infra on Slack for vault access.
4. Run `make dev`

### Required Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DB_PASSWORD` | PostgreSQL password | `P@ssw0rd!2024#Staging` |
| `GITHUB_TOKEN` | GitHub classic PAT | `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| `GITHUB_PACKAGE_TOKEN` | GitHub fine-grained PAT for packages | `github_pat_...` |
| `GITLAB_DEPLOY_TOKEN` | GitLab deploy token | `gldt-...` |
| `OPENAI_API_KEY` | OpenAI project API key | `sk-proj-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-api03-...` |
| `AWS_ACCESS_KEY_ID` | AWS IAM access key (20 chars) | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key (40 chars) | (example: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`) |
| `SLACK_BOT_TOKEN` | Slack bot token | `xoxb-...` |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook | `hooks.slack.com/services/...` |
| `STRIPE_SECRET_KEY` | Stripe secret key | `sk_live_...` |
| `SENDGRID_API_KEY` | SendGrid API key | `SG....` |
| `HF_TOKEN` | HuggingFace token | `hf_...` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | `sk-...` |
| `GROQ_API_KEY` | Groq API key | `gsk_...` |
| `TWILIO_ACCOUNT_SID` | Twilio account SID | `AC...` |
| `JWT_SECRET` | Internal JWT signing secret | (256-bit Base64) |

> ⚠️  **Never commit real credentials.** The values shown above are examples.
> Use `cloak redact` before pasting logs into ChatGPT or Claude.

### Quick Start (Development)

```bash
# Start all services locally
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Verify it works
curl http://localhost:3000/health
```

### Environment File Template (.env.example)

```bash
DB_PASSWORD=replace_me
GITHUB_TOKEN=ghp_replace_me_with_real_token
OPENAI_API_KEY=sk-proj-replace_me
ANTHROPIC_API_KEY=sk-ant-api03-replace_me
AWS_ACCESS_KEY_ID=AKIAREPLACEME
AWS_SECRET_ACCESS_KEY=replace_me_with_40_char_secret
SLACK_BOT_TOKEN=xoxb-replace-me
SLACK_WEBHOOK_URL=hooks.slack.com/services/replace/me/here
STRIPE_SECRET_KEY=sk_live_replace_me
```
