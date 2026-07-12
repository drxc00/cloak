# Cloak

**Redact sensitive data before it reaches an AI model.**

Cloak is a CLI tool that scans text, files, or stdin for sensitive information — names, emails, IP addresses, API keys, credentials, and more — and replaces each match with a `[REDACTED - TYPE]` marker.

```bash
echo "My name is Neil Villanueva, contact me at neil@company.com" | cloak redact
My name is [REDACTED - NAME], contact me at [REDACTED - EMAIL]
```

## Why

Teams paste production logs, customer tickets, and config files into LLMs every day to move faster. Most of the time nobody notices what just left the building. Cloak puts a filter in that path, so scrubbing sensitive data isn't something anyone has to remember to do by hand.

## Features

- **Fast by default** — structured formats (IPs, SSNs, credit cards, JWTs, private keys) are caught with regex in microseconds; API keys and tokens (GitHub, AWS, OpenAI, Anthropic, Slack, Stripe, and 30+ other vendor formats) are caught via prefix + Shannon-entropy scoring
- **Context-aware (opt-in)** — a small local GLiNER model, run out-of-process via a Rust sidecar, catches names, addresses, usernames, and organizations that regex can't — no GPU or Python runtime needed at redact-time
- **Structure-preserving** — redacts values in place without reformatting or breaking surrounding syntax
- **Configurable** — disable specific redaction types per run, or preview matches without touching the file

## Installation

Requires Go 1.25+.

```bash
git clone https://github.com/drxc00/cloak.git
cd cloak
make build          # builds ./bin/cloak
```

Or build with plain `go build`:

```bash
go build -o bin/cloak ./cmd/cloak/
```

Prebuilt binaries for Linux/macOS (amd64/arm64) are published on [Releases](https://github.com/drxc00/cloak/releases) for tagged versions.

### Enabling NER (`--thorough`)

Regex + secret-scanning work out of the box. Detecting names/addresses/usernames additionally requires the `cloak-nerd` sidecar and its model, which `cloak init` downloads (~150–300 MB) into `~/.cache/cloak/`:

```bash
cloak init
```

This fetches, per your OS/arch:
- `cloak-nerd` — the Rust ONNX Runtime sidecar binary
- `model.onnx` — the GLiNER PII model (FP32; UINT8 quantization measurably hurt detection quality, so it's off by default)
- `tokenizer.json`, `gliner_config.json` — its tokenizer and prompt config

Re-run `cloak init` any time to re-download and replace these files.

## Quick start

**Filter a file:**
```bash
cloak redact input.log > output.log
```

**Pipe from stdin:**
```bash
tail -f app.log | cloak redact
```

**Redact inline text:**
```bash
cloak redact --text "My name is Neil, call me at 555-0123"
```

**Enable NER (names, addresses, usernames, organizations):**
```bash
cloak redact --thorough input.log
```

**Disable specific types:**
```bash
cloak redact --disable HOSTNAME,USERNAME input.log
```

## How it works

Cloak runs detection as a staged pipeline. Each stage only looks at text the previous stage didn't already claim:

1. **Regex matcher** — emails, phone numbers, IPv4/IPv6, MAC addresses, SSNs, credit cards, JWTs, PEM key blocks
2. **Secret scanner** — vendor API keys/tokens by known prefix, plus generic high-entropy credential detection
3. **NER model** *(opt-in via `--thorough`, requires `cloak init`)* — names, addresses, usernames, hostnames, organizations, via a local GLiNER model run through the `cloak-nerd` sidecar

> `--dry-run` is accepted as a flag but not yet wired up — it currently has no effect and redaction always applies. Tracked as a known gap.

## Supported redaction types

**Structured PII (regex, always on):** `EMAIL` · `PHONE` · `IPv4` · `IPv6` · `MAC_ADDRESS` · `SSN` · `CREDIT_CARD`

**Secrets & credentials (always on):** `JWT` · `PRIVATE_KEY` · `TOKEN` · `API_KEY` · `CREDENTIALS` · `GENERIC_API_KEY` · `DB_CREDENTIALS`, plus 30+ vendor-specific patterns (GitHub, GitLab, AWS, OpenAI, Anthropic, OpenRouter, DeepSeek, Groq, Hugging Face, Perplexity, npm, PyPI, Docker Hub, Slack, SendGrid, Twilio, Notion, Stripe, Heroku, Grafana, Atlassian, Doppler, Replicate, Datadog, Mailgun, Vault, Linear, age — full list in `internal/pipeline/config.go`)

**NER, opt-in via `--thorough`:** `NAME` · `USERNAME` · `ADDRESS` · `HOSTNAME` · `ORGANIZATION`

Any type can be turned off per run with `--disable TYPE1,TYPE2`.

## Project layout

```
cloak/
├── cmd/cloak/                 # CLI entrypoint (redact, init)
├── internal/
│   ├── pipeline/               # Stage orchestration, config, match merging
│   └── stages/
│       ├── regex/              # Structured PII patterns
│       ├── secrets/             # Vendor API key + entropy-based detection
│       └── ner/                 # Shells out to the cloak-nerd sidecar
├── nerd/                       # cloak-nerd: Rust GLiNER ONNX inference sidecar
├── scripts/                    # Python model export/benchmark (build-time only)
├── testdata/                   # Benchmark cases + integration fixtures
├── integration/                # End-to-end pipeline tests
└── .github/workflows/          # Release CI (builds + publishes all artifacts)
```

## Development

```bash
make build              # go build -> bin/cloak
make nerd                # cargo build --release -> bin/cloak-nerd
make test                 # unit tests for detection stages
make test-integration      # end-to-end pipeline tests against testdata/benchmark_cases.jsonl
make bench                  # redaction throughput benchmark
make vet                     # go vet
make fmt                      # go fmt
```

To exercise the full NER path locally (requires Python for model export):

```bash
make model-export     # exports knowledgator/gliner-pii-edge-v1.0 to ./model-export/
make model-benchmark    # scores the exported model against testdata/benchmark_cases.jsonl
make init-dev             # copies bin/cloak-nerd + model-export/* into ~/.cache/cloak/
```

`make clean` removes `bin/`, `model-export/`, and Rust build artifacts.

## Releasing

Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds `cloak` (Go, linux/darwin × amd64/arm64), `cloak-nerd` (Rust, same matrix), exports the GLiNER model (FP32), benchmarks it as a release gate, and publishes everything plus a `checksums.txt` to GitHub Releases.

## Benchmarking

Detection stages are tested against a suite of real-world-style inputs (stack traces, `.env` files, nginx logs, git diffs, and deliberate false-positive traps) in `testdata/benchmark_cases.jsonl`. Contributions of new edge cases are welcome.

## What Cloak is not

Cloak isn't a content moderation tool, and it doesn't judge what you're doing with an AI model — it only controls what sensitive data reaches it. It's also not a replacement for proper secrets management; treat it as a last line of defense for the moment something sensitive is about to be pasted or sent.

## Contributing

Issues and PRs welcome. If you're adding a new redaction type, please include test cases in `testdata/`.

## License

[MIT](LICENSE)
