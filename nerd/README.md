# cloak-nerd

Token-classification PII inference sidecar for [Cloak](https://github.com/drxc00/cloak).

Reads NDJSON requests from stdin, writes NDJSON responses to stdout — no HTTP,
no gRPC, just pipes. Designed to be spawned by the Cloak Go pipeline as a
subprocess.

## Protocol

**Input** (stdin, one JSON object per line):

```json
{"text": "Contact Maria Santos at mariasantos88@yahoo.com", "labels": ["NAME", "ADDRESS", "USERNAME"]}
```

**Output** (stdout, one JSON object per line):

```json
{"entities": [{"start": 8, "end": 21, "type": "NAME", "score": 0.94}]}
```

Entity offsets are byte positions in the original text. The `labels` field
acts as an **allow-list filter** over the classifier's fixed output classes.
If a requested label is not a class the model knows, the sidecar simply never
emits it.

If `cloak-nerd` was started with a `--threshold`, only entities scoring at or
above it are returned.

## Model

cloak-nerd expects a **token-classification** ONNX model with a standard
`id2label` config, exported from the training pipeline:

```bash
pip install -r ../scripts/requirements.txt
python ../scripts/train_model.py --out ./trained/
python ../scripts/export_model.py --checkpoint ./trained/ --out ./model/ --quantize
```

This produces three files the sidecar needs:
- `model.onnx` — INT8 (or FP32) token-classification graph
- `tokenizer.json` — Fast tokenizer with byte offset support
- `model_config.json` — Contains `id2label`, `max_len`, `model_type`, and
  the ONNX input/output names

The sidecar asserts `model_type == "token-classifier-v1"` on startup to
refuse incompatible (old GLiNER) artifacts.

## Usage

```bash
cloak-nerd --model model.onnx --tokenizer tokenizer.json --config model_config.json --threshold 0.5
```

The process reads lines until EOF, then exits. Model and tokenizer are loaded
once at startup and reused across all requests.

## Build

```bash
cargo build --release
```

A static MUSL build for portable Linux deployment:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

## Architecture

The sidecar uses a BIO-tagged transformer encoder:
1. Tokenize text with byte offsets (fast tokenizer).
2. Run ONNX forward pass → per-subword-token logits.
3. Softmax → argmax label per token.
4. BIO decode: merge contiguous non-O tokens into entity spans by byte
   offsets — no word-boundary reconstruction needed.
5. Filter by allow-list labels and threshold; return byte-offset spans.

For inputs longer than `max_len`, overlapping windows are used with span
deduplication.

---

*Created with assistance from Claude.*
