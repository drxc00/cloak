.PHONY: all build test nerd clean init-dev model-data model-train model-export model-parity model-benchmark model-upload model-pipeline

# Default target.
all: build

build:
	go build -ldflags="-s -w" -o bin/cloak ./cmd/cloak/

test:
	go test ./internal/stages/... -count=1

test-integration:
	go test ./integration/ -v -run TestBenchmarkCases

bench:
	go test ./integration/ -bench=BenchmarkRedact -benchmem

vet:
	go vet ./...

fmt:
	go fmt ./...

nerd:
	cd nerd && cargo build --release
	cp nerd/target/release/cloak-nerd bin/

nerd-clean:
	cd nerd && cargo clean

# Build the unified ~580K-row dataset from all sources.
# Runs augment + Few-NERD conversion, merges with ai4privacy, then stratified split.
model-data:
	python3 scripts/training/prepare_dataset.py \
		--out scripts/training/datasets/augmented/ \
		--seed 42 \
		--synthetic-rows 50000 \
		--fewnerd-person-rows 50000 \
		--fewnerd-negative-rows 100000
	python3 scripts/training/build_combined.py \
		--out data/combined.jsonl --seed 42
	python3 scripts/training/split_dataset.py \
		--in data/combined.jsonl --out-dir data/ \
		--train 0.80 --val 0.10 --test 0.10 --seed 42

# --- Train ---
#
# distilbert-base-multilingual-cased → trained/
# Tune BATCH_SIZE/MAX_LEN/MAX_TRAIN_ROWS for your GPU.
#
# Quick iteration (smoke test, ~2 min):
#   MAX_TRAIN_ROWS=2000 EPOCHS=1 make model-train
#
# Decent model (50K rows, 4 epochs, ~45 min on 4GB):
#   MAX_TRAIN_ROWS=50000 make model-train
#
# Full dataset (465K rows, 4 epochs, ~8 hr on 4GB):
#   make model-train

BATCH_SIZE ?= 16
MAX_LEN    ?= 128
EPOCHS     ?= 4
MAX_TRAIN_ROWS ?= 0

_TRAIN_ROWS_FLAG = $(if $(filter 0,$(MAX_TRAIN_ROWS)),,$(if $(MAX_TRAIN_ROWS),--max-train-rows $(MAX_TRAIN_ROWS)))

model-train:
	python3 -m venv /tmp/model-train-venv
	/tmp/model-train-venv/bin/pip install -r scripts/training/requirements.txt
	/tmp/model-train-venv/bin/python scripts/training/train_model.py \
		--backbone distilbert-base-multilingual-cased \
		--out ./trained/ \
		--data-dir ./data/ \
		--batch-size $(BATCH_SIZE) \
		--max-len $(MAX_LEN) \
		--epochs $(EPOCHS) \
		$(_TRAIN_ROWS_FLAG) \
		--seed 42

# --- Export to ONNX (INT8 quantized) ---

model-export:
	python3 -m venv /tmp/model-export-venv
	/tmp/model-export-venv/bin/pip install -r scripts/requirements.txt
	/tmp/model-export-venv/bin/python scripts/export_model.py \
		--checkpoint ./trained/ \
		--out ./model-export/ \
		--quantize

# --- Parity check (PyTorch vs ONNX) ---

model-parity:
	/tmp/model-export-venv/bin/python scripts/parity_check.py \
		--checkpoint ./trained/ \
		--onnx ./model-export/

# --- Benchmark against test cases ---

model-benchmark:
	/tmp/model-export-venv/bin/python scripts/benchmark_model.py \
		--model-dir ./model-export/ \
		--cases testdata/benchmark_cases.jsonl \
		--nerd-bin ./bin/cloak-nerd

# Full training pipeline from data to benchmark.
model-pipeline: model-data model-train model-export model-parity model-benchmark

init-dev: build nerd
	@mkdir -p $$HOME/.cache/cloak/bin
	@mkdir -p $$HOME/.cache/cloak/models
	cp bin/cloak-nerd $$HOME/.cache/cloak/bin/cloak-nerd
	@if [ -f model-export/model.onnx ]; then \
		cp model-export/model.onnx $$HOME/.cache/cloak/models/model.onnx; \
		cp model-export/tokenizer.json $$HOME/.cache/cloak/models/tokenizer.json; \
		cp model-export/model_config.json $$HOME/.cache/cloak/models/model_config.json; \
		echo "✓ Model copied from model-export/"; \
	else \
		echo "⚠  Run 'make model-train model-export' first to produce model files"; \
	fi
	@echo "✓ Dev environment ready — use 'bin/cloak redact ...'"

release-snapshot: build nerd
	@echo "=== Release snapshot ==="
	@echo "Data:    make model-data"
	@echo "Train:   make model-train"
	@echo "Export:  make model-export"
	@echo "Parity:  make model-parity"
	@echo "Bench:   make model-benchmark"
	@echo "Upload:  make model-upload"
	@echo "Rust:    built → bin/cloak-nerd"
	@echo "Go:      built → bin/cloak"
	@echo "Assets:  model-export/{model.onnx,tokenizer.json,model_config.json}"
	@echo "         bin/cloak, bin/cloak-nerd"

clean:
	rm -rf bin/ augmented/ data/
	cd nerd && cargo clean

# clean-all removes trained models and exports too — only run if you mean it.
clean-all: clean
	rm -rf model-export/ trained/ onnx-*/
