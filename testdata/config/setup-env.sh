#!/usr/bin/env bash

# ── Database ───────────────────────────────────────────────────────────
export DB_HOST="db-staging.internal.corp"
export DB_PORT=5432
export DB_NAME="app_staging"
export DB_USER="staging_user"
export DB_PASSWORD="P@ssw0rd!2024#Staging"

# ── Redis ──────────────────────────────────────────────────────────────
export REDIS_URL="redis://:redis-staging-pass-42@redis-staging.internal.corp:6379/0"

# ── GitHub ────────────────────────────────────────────────────────────
export GITHUB_TOKEN="ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz"
export GITHUB_PACKAGE_TOKEN="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"
export GITLAB_DEPLOY_TOKEN="gldt-b29a3bb209abad344c65df00301fb2881acf4e8"

# ── LLM Providers ──────────────────────────────────────────────────────
export OPENAI_API_KEY="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"
export ANTHROPIC_API_KEY="sk-ant-api03-b58a67063e5dce2d5d16179e8fa080b388cf321e0fb2ef2ade272d7679cdfea4f92e9196a124fd122c8fc15914fbba6e8f1d"
export HUGGINGFACE_TOKEN="hf_xPqR7mN3vK5wQ9sT2yA8bC1dE4fG6hJ0lM9nOpQr"
export DEEPSEEK_API_KEY="sk-9c798937ff804cbf41babbe137cbe088"
export GROQ_API_KEY="gsk_2a2cf79313484d5be1d2f6c0ed17d9271486ce4eace749b9"

# ── AWS ────────────────────────────────────────────────────────────────
export AWS_ACCESS_KEY_ID="AKIALALEMEL33243OLIB"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_REGION="us-east-1"

# ── Slack ──────────────────────────────────────────────────────────────
export SLACK_BOT_TOKEN="xoxb-781236542736-2364535789652-GkwFDQoHqzXDVsC6GzqYUypD"
export SLACK_USER_TOKEN="xoxp-41684372915-1320496754-45609968301-e708ba56e1517a99f6b5fb07349476ef"
export SLACK_WEBHOOK_URL="hooks.slack.com/services/T024TTTTT/BBB72BBL/AZAAA9u0pA4ad666eMgbi555"

# ── Payments & Email ───────────────────────────────────────────────────
export STRIPE_SECRET_KEY="sk_live_f347ed9b7f842105caf30cd5adb41fe7144cface99f6046984b3d2e1"
export SENDGRID_API_KEY="SG.3dd3e4ea31571b192a6cc0cf8.b297f8d481d51fa9997cbfa6487a2e738e50009bd41a1c2d3"

# ── Twilio ─────────────────────────────────────────────────────────────
export TWILIO_ACCOUNT_SID="ACabcdef1234567890abcdef1234567890"
export TWILIO_AUTH_TOKEN="dGhpcy1pcy1hLWZha2UtdHdpbGlvLWF1dGgtdG9rZW4tZm9yLXRlc3Rpbmc"

# ── Infrastructure ─────────────────────────────────────────────────────
export GRAFANA_TOKEN="glsa_abcdefghijklmnopqrstuvwxyz12345_a1b2c3d4"
export DOCKERHUB_TOKEN="dckr_pat_abcdefghijklmnopqrstuvwxyzABCDE"

# ── Monitoring & Delivery ──────────────────────────────────────────────
export DD_API_KEY="pub3c9f5a01b7d482e6f1a8c49d2e7b036f"
export DD_APP_KEY="pvt8a7b6c5d4e3f2a1b09c8d7e6f5a4b3c2d1e0f"
export MAILGUN_API_KEY="key-3c9f5a01b7d482e6f1a8c49d2e7b036f"

# ── Secrets Management ─────────────────────────────────────────────────
export VAULT_TOKEN="hvs.CAGgioFNoP2vRZ5d3z7xLmK8wQ4sY1bH6tU9nJ"
export LINEAR_API_KEY="lin_api_M9nOpQrStUvWxYz01234567AbCdEfGhIjKlMnN"
export AGE_KEY="AGE-SECRET-KEY-1QF3h5G6t7u8v9w0x1y2z3A4B5C6D7E8F9G0H1I2J3K4L5M6N7"

# ── Internal ───────────────────────────────────────────────────────────
export JWT_SECRET="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n"
export SERVICE_API_TOKEN="abcdefghijklmnopqrstuvwxyz0123456789"
export INTERNAL_KEY="Zf3D0LXCM3EIMbgJpUNnkRtOfOueHznB"
