.PHONY: all build test nerd clean init-dev

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

model-export:
	python3 -m venv /tmp/model-export-venv
	/tmp/model-export-venv/bin/pip install -r scripts/requirements.txt
	/tmp/model-export-venv/bin/python scripts/export_model.py \
		--model knowledgator/gliner-pii-edge-v1.0 \
		--out ./model-export/ \
		--no-quantize

model-benchmark:
	/tmp/model-export-venv/bin/python scripts/benchmark_model.py \
		--model-dir ./model-export/ \
		--cases testdata/benchmark_cases.jsonl

init-dev: build nerd
	@mkdir -p $$HOME/.cache/cloak/bin
	@mkdir -p $$HOME/.cache/cloak/models
	cp bin/cloak-nerd $$HOME/.cache/cloak/bin/cloak-nerd
	@if [ -f model-export/model.onnx ]; then \
		cp model-export/model.onnx $$HOME/.cache/cloak/models/model.onnx; \
		cp model-export/tokenizer.json $$HOME/.cache/cloak/models/tokenizer.json; \
		cp model-export/gliner_config.json $$HOME/.cache/cloak/models/gliner_config.json; \
		echo "✓ Model files copied from model-export/"; \
	else \
		echo "⚠  Run 'make model-export' first to produce model files"; \
	fi
	@echo "✓ Dev environment ready — use 'bin/cloak redact --thorough ...'"

release-snapshot: build nerd
	@echo "=== Release snapshot ==="
	@echo "Python:  make model-export && make model-benchmark"
	@echo "Rust:    built → bin/cloak-nerd"
	@echo "Go:      built → bin/cloak"
	@echo "Assets:  model-export/model.onnx, model-export/tokenizer.json, model-export/gliner_config.json"
	@echo "         bin/cloak, bin/cloak-nerd"

clean:
	rm -rf bin/ model-export/
	cd nerd && cargo clean
