# cloak-nerd

GLiNER PII inference sidecar for [Cloak](https://github.com/drxc00/cloak).

Reads NDJSON requests from stdin, writes NDJSON responses to stdout — no HTTP,
no gRPC, just pipes. Designed to be spawned by the Cloak Go pipeline as a
subprocess.

## Protocol

**Input** (stdin, one JSON object per line):

```json
{"text": "Contact Maria Santos at mariasantos88@yahoo.com", "labels": ["NAME", "EMAIL"]}
```

**Output** (stdout, one JSON object per line):

```json
{"entities": [{"start": 8, "end": 21, "type": "NAME", "score": 0.94}]}
```

Entity offsets are byte positions in the original text. If `cloak-nerd` was
started with a `--threshold`, only entities scoring at or above it are
returned.

## Usage

```bash
cloak-nerd --model model.onnx --tokenizer tokenizer.json --threshold 0.5
```

The process reads lines until EOF, then exits. Model and tokenizer are loaded
once at startup and reused across all requests.

## Model

cloak-nerd expects a GLiNER PII model exported to ONNX (UINT8 quantized).
Export one with the scripts in the repository root:

```bash
pip install -r ../scripts/requirements.txt
python ../scripts/export_model.py \
  --model knowledgator/gliner-pii-edge-v1.0 \
  --out ./model/
```

This produces `model.onnx` and `tokenizer.json`.

## Build

```bash
cargo build --release
```

A static MUSL build for portable Linux deployment:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

---

*Created with assistance from Claude.*
