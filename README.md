# Cloak

**Redact sensitive data before it reaches an AI model.**

Cloak is a CLI tool that filters logs, tickets, config files, and any other text for sensitive information — names, emails, IP addresses, API keys, passwords, credentials, and more — before it gets pasted or sent into an LLM. It replaces each match with a clear `[REDACTED - TYPE]` marker, and it can run as a one-off filter or as a standing local proxy in front of your AI provider.

```
$ echo "My name is Neil Villanueva, contact me at neil@company.com" | cloak
My name is [REDACTED - NAME], contact me at [REDACTED - EMAIL]
```

## Why

Teams paste production logs, customer tickets, and config files into LLMs every day to move faster. Most of the time nobody notices what just left the building. Cloak puts a filter in that path automatically, so scrubbing sensitive data isn't something anyone has to remember to do by hand.

## Features

- **Fast by default** — structured formats (IPs, SSNs, credit cards, JWTs, private keys) are caught with regex in microseconds; API keys and tokens are caught via prefix + entropy detection, also in microseconds
- **Context-aware** — a small local model catches names, addresses, and other entities regex can't, without needing a GPU or a Python runtime
- **Structure-preserving** — redacts values in place inside JSON, YAML, CSV, and log lines without reformatting or breaking syntax
- **Cloak Proxy** — point any OpenAI/Anthropic/Gemini-compatible SDK at Cloak instead of the provider directly, and every request gets filtered transparently, using your real API key
- **Configurable** — turn specific redaction types on/off, tune detection thresholds, or run in dry-run mode to preview what would be redacted

## Installation

```bash
go install github.com/<your-org>/cloak/cmd/cloak@latest
```

Or download a prebuilt binary from [Releases](#) for your platform.

## Quick start

**Filter a file:**
```bash
cloak redact input.log > output.log
```

**Pipe from stdin:**
```bash
tail -f app.log | cloak
```

**Dry run (report only, no rewrite):**
```bash
cloak redact input.log --dry-run
```

**Run as a proxy in front of an AI provider:**
```bash
cloak proxy --port 8317
```
Then point your SDK at it instead of the real provider:
```bash
export OPENAI_BASE_URL=http://localhost:8317
```
Cloak filters the outgoing request, forwards it to the real provider with your real API key, and streams the response back untouched.

## How it works

Cloak runs detection as a staged pipeline. Each stage only looks at what the previous one didn't already redact, so the common case clears the whole pipeline in single-digit milliseconds:

1. **Regex matcher** — IPs, SSNs, credit cards, JWTs, MAC addresses, IBANs, PEM key blocks
2. **Secret scanner** — API keys and tokens via known prefixes + Shannon entropy scoring, including unlabeled/encoded credentials
3. **NER model** — names, addresses, usernames in prose or file paths, via a small local model
4. **LLM fallback** *(optional, off by default)* — for genuinely ambiguous, context-dependent cases; enable with `--thorough`

## Configuration

```yaml
# cloak.yaml
redact:
  types:
    NAME: true
    EMAIL: true
    IP: true
    API_KEY: true
    HOSTNAME: false   # disable if your logs are internal-only and hostnames are fine to keep
  entropy_threshold: 3.5
  thorough: false     # enables the LLM fallback stage

proxy:
  listen: "localhost:8317"
  providers:
    openai:    "https://api.openai.com"
    anthropic: "https://api.anthropic.com"
    gemini:    "https://generativelanguage.googleapis.com"
```

## Supported redaction types

`NAME` · `EMAIL` · `PHONE` · `IP` · `MAC_ADDRESS` · `API_KEY` · `PASSWORD` · `USERNAME` · `CREDENTIALS` · `HOSTNAME` · `SSN` · `CREDIT_CARD` · `IBAN` · `ADDRESS` · `PRIVATE_KEY` · `TOKEN`

## Project layout

```
cloak/
├── cmd/cloak/           # CLI entrypoint
├── internal/
│   ├── redact/          # Pipeline orchestrator
│   ├── stages/           # Detection stages (regex, secrets, NER, LLM)
│   ├── proxy/            # Cloak Proxy (provider adapters, streaming)
│   ├── config/            # Config loading
│   └── report/            # Dry-run reporting
├── testdata/              # Benchmark cases used in tests
└── README.md
```

## Benchmarking

Cloak's detection stages are tested against a benchmark suite of real-world-style inputs (stack traces, `.env` files, nginx logs, Slack exports, git diffs, and deliberate false-positive traps) — see `testdata/benchmark_cases.json`. Contributions of new edge cases are welcome.

## What Cloak is not

Cloak isn't a content moderation tool, and it doesn't judge what you're doing with an AI model — it only controls what sensitive data reaches it. It's also not a replacement for proper secrets management; treat it as a last line of defense for the moment something sensitive is about to be pasted or sent.

## Contributing

Issues and PRs welcome. If you're adding a new redaction type or provider adapter, please include test cases in `testdata/`.

## License

[MIT](#)