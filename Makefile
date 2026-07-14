.PHONY: all build test nerd clean init-dev model-data model-train model-train-edge model-train-full model-export model-export-edge model-export-full model-parity model-parity-edge model-parity-full model-benchmark model-benchmark-edge model-benchmark-full model-upload model-pipeline

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

# --- Train one or both model variants ---
#
# edge: distilbert-base-multilingual-cased  →  trained-edge/
# full: microsoft/deberta-v3-base           →  trained-full/

model-train-edge:
	python3 -m venv /tmp/model-train-venv
	/tmp/model-train-venv/bin/pip install -r scripts/training/requirements.txt
	/tmp/model-train-venv/bin/python scripts/training/train_model.py \
		--backbone distilbert-base-multilingual-cased \
		--out ./trained-edge/ \
		--data-dir ./data/ \
		--seed 42

model-train-full:
	python3 -m venv /tmp/model-train-venv
	/tmp/model-train-venv/bin/pip install -r scripts/training/requirements.txt
	/tmp/model-train-venv/bin/python scripts/training/train_model.py \
		--backbone microsoft/deberta-v3-base \
		--out ./trained-full/ \
		--data-dir ./data/ \
		--seed 42

model-train: model-train-edge model-train-full

# --- Export to ONNX ---
#
# edge: INT8 quantized  →  model-export-edge/
# full: FP32             →  model-export-full/

model-export-edge:
	python3 -m venv /tmp/model-export-venv
	/tmp/model-export-venv/bin/pip install -r scripts/requirements.txt
	/tmp/model-export-venv/bin/python scripts/export_model.py \
		--checkpoint ./trained-edge/ \
		--out ./model-export-edge/ \
		--quantize

model-export-full:
	/tmp/model-export-venv/bin/python scripts/export_model.py \
		--checkpoint ./trained-full/ \
		--out ./model-export-full/

model-export: model-export-edge model-export-full

# --- Parity check (PyTorch vs ONNX) ---

model-parity-edge:
	/tmp/model-export-venv/bin/python scripts/parity_check.py \
		--checkpoint ./trained-edge/ \
		--onnx ./model-export-edge/

model-parity-full:
	/tmp/model-export-venv/bin/python scripts/parity_check.py \
		--checkpoint ./trained-full/ \
		--onnx ./model-export-full/

model-parity: model-parity-edge model-parity-full

# --- Benchmark against test cases ---

model-benchmark-edge:
	/tmp/model-export-venv/bin/python scripts/benchmark_model.py \
		--model-dir ./model-export-edge/ \
		--cases testdata/benchmark_cases.jsonl \
		--nerd-bin ./bin/cloak-nerd

model-benchmark-full:
	/tmp/model-export-venv/bin/python scripts/benchmark_model.py \
		--model-dir ./model-export-full/ \
		--cases testdata/benchmark_cases.jsonl \
		--nerd-bin ./bin/cloak-nerd

model-benchmark: model-benchmark-edge model-benchmark-full

# --- Upload models to Hugging Face Hub ---
#
# Requires HF write access. Set HF_REPO (or use the default).
# Run 'huggingface-cli login' first to authenticate.
HF_REPO ?= drxc0/cloak-ner-v1

model-upload:
	python3 -m venv /tmp/model-upload-venv
	/tmp/model-upload-venv/bin/pip install huggingface_hub
	/tmp/model-upload-venv/bin/python -c '\
		import os;\
		from huggingface_hub import HfApi;\
		repo = os.environ.get("HF_REPO", "$(HF_REPO)");\
		api = HfApi();\
		api.upload_folder(folder_path="./model-export-edge", repo_id=repo, path_in_repo="edge");\
		api.upload_folder(folder_path="./model-export-full", repo_id=repo, path_in_repo="full");\
		print("✓ Edge + Full uploaded to " + repo)'

# Full training pipeline from data to benchmark.
model-pipeline: model-data model-train model-export model-parity model-benchmark

init-dev: build nerd
	@mkdir -p $$HOME/.cache/cloak/bin
	@mkdir -p $$HOME/.cache/cloak/models
	cp bin/cloak-nerd $$HOME/.cache/cloak/bin/cloak-nerd
	@if [ -f model-export-edge/model.onnx ]; then \
		cp model-export-edge/model.onnx $$HOME/.cache/cloak/models/model.onnx; \
		cp model-export-edge/tokenizer.json $$HOME/.cache/cloak/models/tokenizer.json; \
		cp model-export-edge/model_config.json $$HOME/.cache/cloak/models/model_config.json; \
		echo "edge" > $$HOME/.cache/cloak/models/variant.txt; \
		echo "✓ Edge model copied from model-export-edge/"; \
	elif [ -f model-export-full/model.onnx ]; then \
		cp model-export-full/model.onnx $$HOME/.cache/cloak/models/model.onnx; \
		cp model-export-full/tokenizer.json $$HOME/.cache/cloak/models/tokenizer.json; \
		cp model-export-full/model_config.json $$HOME/.cache/cloak/models/model_config.json; \
		echo "full" > $$HOME/.cache/cloak/models/variant.txt; \
		echo "✓ Full model copied from model-export-full/"; \
	else \
		echo "⚠  Run 'make model-train model-export' first to produce model files"; \
	fi
	@echo "✓ Dev environment ready — use 'bin/cloak redact --thorough ...'"

release-snapshot: build nerd
	@echo "=== Release snapshot ==="
	@echo "Data:    make model-data"
	@echo "Train:   make model-train  (both edge + full)"
	@echo "Export:  make model-export (both variants)"
	@echo "Parity:  make model-parity"
	@echo "Bench:   make model-benchmark"
	@echo "Upload:  make model-upload"
	@echo "Rust:    built → bin/cloak-nerd"
	@echo "Go:      built → bin/cloak"
	@echo "Assets:  model-export-edge/{model.onnx,tokenizer.json,model_config.json}"
	@echo "         model-export-full/{model.onnx,tokenizer.json,model_config.json}"
	@echo "         bin/cloak, bin/cloak-nerd"

clean:
	rm -rf bin/ model-export/ model-export-edge/ model-export-full/ \
		trained/ trained-edge/ trained-full/ onnx-*/ augmented/ data/
	cd nerd && cargo clean
