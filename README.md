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

- **Layered detection** — structured formats (IPs, SSNs, credit cards, JWTs, private keys) via regex; API keys and tokens (GitHub, AWS, OpenAI, Anthropic, Slack, Stripe, and 30+ other vendor formats) via prefix + Shannon-entropy scoring; names, addresses, and usernames via a local token-classification model run through a Rust sidecar — no GPU or Python runtime needed at redact-time
- **`--fast` for regex-only mode** — skip the NER model when you only need structured PII and secret redaction
- **Structure-preserving** — redacts values in place without reformatting or breaking surrounding syntax
- **Configurable** — disable specific redaction types per run with `--disable`

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

Prebuilt binaries for Linux/macOS (amd64/arm64) are published on [GitHub Releases](https://github.com/drxc00/cloak/releases) for tagged versions.

### Installing the NER model

Regex + secret scanning work out of the box. Named-entity detection (names, addresses, usernames) runs by default and requires the `cloak-nerd` sidecar plus its model, which `cloak init` downloads into `~/.cache/cloak/`:

```bash
cloak init
```

This fetches:
- `cloak-nerd` — the Rust ONNX Runtime sidecar binary (from [GitHub Releases](https://github.com/drxc00/cloak/releases))
- `model.onnx` — the PII token-classification model, ~130 MB (from [Hugging Face Hub](https://huggingface.co/drxc0/cloak-ner-v1))
- `tokenizer.json`, `model_config.json` — tokenizer and sidecar config

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

**Skip NER for regex-only mode:**
```bash
cloak redact --fast input.log
```

**Disable specific types:**
```bash
cloak redact --disable NAME,USERNAME input.log
```

## How it works

Cloak runs detection as a staged pipeline. Each stage only looks at text the previous stage didn't already claim:

1. **Regex matcher** — emails, phone numbers, IPv4/IPv6, MAC addresses, SSNs, credit cards, IBANs, JWTs, PEM key blocks
2. **Secret scanner** — vendor API keys/tokens by known prefix, plus generic high-entropy credential detection
3. **NER model** *(on by default; skip with `--fast`)* — names, addresses, usernames via a local token-classification model run through the `cloak-nerd` sidecar. Requires `cloak init` to download the model first; if the model isn't installed this stage is silently skipped.

## Supported redaction types

**Structured PII (regex, always on):**
`EMAIL` · `PHONE` · `IPv4` · `IPv6` · `MAC_ADDRESS` · `SSN` · `CREDIT_CARD` · `IBAN`

**Secrets & credentials (always on):**
`JWT` · `PRIVATE_KEY` · `TOKEN` · `API_KEY` · `CREDENTIALS` · `PASSWORD` ·
`GENERIC_API_KEY` · `DB_CREDENTIALS`, plus 30+ vendor-specific patterns:
GitHub (PAT, OAuth, app tokens) · GitLab · AWS · OpenAI · Anthropic ·
OpenRouter · DeepSeek · Groq · Hugging Face · Perplexity · npm · PyPI ·
Docker Hub · Slack (bot, user, webhook) · SendGrid · Twilio · Notion ·
Stripe · Heroku · Grafana · Atlassian · Doppler · Replicate · Datadog ·
Mailgun · Vault · Linear · age

**NER (on by default, requires `cloak init`):**
`NAME` · `USERNAME` · `ADDRESS`

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
├── nerd/                       # cloak-nerd: Rust token-classification ONNX inference sidecar
├── scripts/                    # Python training, export, and benchmarking scripts
├── testdata/                   # Benchmark cases + integration fixtures
├── integration/                # End-to-end pipeline tests
└── .github/workflows/          # Release CI
```

## Development

```bash
make build              # go build -> bin/cloak
make nerd               # cargo build --release -> bin/cloak-nerd
make test               # unit tests for detection stages
make test-integration   # end-to-end pipeline tests
make bench              # redaction throughput benchmark
make vet                # go vet
make fmt                # go fmt
```

To train and export the NER model locally (requires Python and a GPU):

```bash
make model-data         # builds the unified dataset → data/
make model-train        # trains the token classifier → trained/
make model-export       # exports INT8-quantized ONNX → model-export/
make model-parity       # verifies PyTorch ↔ ONNX agreement
make model-benchmark    # scores against testdata/benchmark_cases.jsonl
make model-upload       # uploads to Hugging Face Hub
make init-dev           # copies model + sidecar into ~/.cache/cloak/
```

`make clean` removes `bin/`, build artifacts, and datasets. Trained models and ONNX exports are preserved — use `make clean-all` to remove those too.

## Releasing

Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds `cloak`
(Go, linux/darwin × amd64/arm64) and `cloak-nerd` (Rust, same matrix), then
publishes all binaries plus a `checksums.txt` to GitHub Releases.

Model training and export are manual, local steps. The ONNX model is published
to [Hugging Face Hub](https://huggingface.co/drxc0/cloak-ner-v1) and
`cloak init` downloads it automatically.

## Benchmarking

Detection stages are tested against a suite of real-world-style inputs (stack traces, `.env` files, nginx logs, git diffs, and deliberate false-positive traps) in `testdata/benchmark_cases.jsonl`. Contributions of new edge cases are welcome.

## What Cloak is not

Cloak isn't a content moderation tool, and it doesn't judge what you're doing with an AI model — it only controls what sensitive data reaches it. It's also not a replacement for proper secrets management; treat it as a last line of defense for the moment something sensitive is about to be pasted or sent.

## Acknowledgements

The NER model is trained on several open datasets:

- **[ai4privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-400k)** —
  a multilingual PII corpus (English, German, Spanish, French, Italian, Dutch)
  covering names, addresses, usernames, and structured PII in natural-language
  prose. Licensed under CC BY-NC-SA 4.0.

- **[Few-NERD](https://github.com/thunlp/Few-NERD)** — a large-scale,
  fine-grained named-entity recognition dataset derived from Wikipedia articles.
  Used under CC BY-NC-SA 4.0 for person-name vocabulary and hard-negative
  signal (organization, location, and product mentions that teach the model
  what *isn't* a person name).

- **[Hypixel Player List](https://hypixel.net/)** and
  **[Epicube Player List](https://epicube.com/)** — publicly available gamer-tag
  dumps used to enrich the username vocabulary pool.

- Curated name lists derived from public sources for first-name and username
  diversity in synthetic augmentation templates.

Synthetic training rows are generated from these vocabularies via 20+ template
factories targeting log files, `.env` blocks, stack traces, git diffs, JSON
payloads, Kubernetes events, CI/CD output, SQL results, Docker logs, Terraform
output, AWS CLI output, code comments, and Slack messages.

## Contributing

Issues and PRs welcome. If you're adding a new redaction type, please include test cases in `testdata/`.

## License

[MIT](LICENSE)
