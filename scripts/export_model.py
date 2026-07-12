#!/usr/bin/env python3
"""
Export a GLiNER PII checkpoint to ONNX.

Ships the FP32 graph by default — UINT8 quantization measurably hurt
detection quality on the edge model, so it's opt-in via --quantize now,
not the default.

Produces three files in the output directory:
  model.onnx          — FP32 ONNX graph (the one cloak-nerd loads), or the
                        UINT8 quantized graph if --quantize was passed
  tokenizer.json       — fast tokenizer, including the `<<ENT>>`/`<<SEP>>`
                         prompt tokens baked in at training time
  gliner_config.json   — max_width / ent_token / sep_token cloak-nerd needs
                         to reconstruct the exact prompt + span layout

Usage:
  python export_model.py --model knowledgator/gliner-pii-edge-v1.0 --out ./edge-out/
  python export_model.py --model knowledgator/gliner-pii-small-v1.0 --out ./small-out/ --quantize
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Export GLiNER PII model to ONNX")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path (e.g. knowledgator/gliner-pii-edge-v1.0)")
    parser.add_argument("--out", required=True,
                        help="Output directory for model.onnx + tokenizer.json + gliner_config.json")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Apply UINT8 quantization (default: off — hurts accuracy on the edge model)")
    parser.add_argument("--no-quantize", action="store_false", dest="quantize",
                        help="Skip quantization, ship the FP32 graph as model.onnx (default)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⟳  Exporting {args.model} → {out_dir}/")

    try:
        from gliner import GLiNER
    except ImportError:
        sys.exit(
            "ERROR: gliner not installed.\n"
            "  pip install -r scripts/requirements.txt"
        )

    model = GLiNER.from_pretrained(
        args.model,
        load_onnx_model=False,
        load_tokenizer=True,
    )

    # `export_to_onnx` always writes the FP32 graph to `onnx_filename` and,
    # when quantize=True, additionally writes a UINT8 graph to
    # `quantized_filename`. It also saves gliner_config.json and the
    # tokenizer (with the `<<ENT>>`/`<<SEP>>` prompt tokens already baked
    # in) into `save_dir` as a side effect.
    result = model.export_to_onnx(
        save_dir=str(out_dir),
        onnx_filename="model_fp32.onnx",
        quantized_filename="model.onnx",
        quantize=args.quantize,
    )

    onnx_path = out_dir / "model.onnx"
    fp32_path = out_dir / "model_fp32.onnx"
    if not args.quantize or result.get("quantized_path") is None:
        # No quantized graph was produced — ship the FP32 graph as model.onnx.
        if not fp32_path.exists():
            sys.exit(f"ERROR: ONNX export failed — {fp32_path} not created")
        fp32_path.replace(onnx_path)
    elif fp32_path.exists():
        # Quantized graph is what we ship — drop the unused intermediate FP32 graph.
        fp32_path.unlink()

    tok_path = out_dir / "tokenizer.json"
    config_path = out_dir / "gliner_config.json"

    if not onnx_path.exists():
        sys.exit(f"ERROR: ONNX export failed — {onnx_path} not created")
    if not tok_path.exists():
        sys.exit(f"ERROR: tokenizer.json not found in {out_dir} — export_to_onnx should have written it")
    if not config_path.exists():
        sys.exit(f"ERROR: gliner_config.json not found in {out_dir} — export_to_onnx should have written it")

    onnx_mb = onnx_path.stat().st_size / (1024 * 1024)
    tok_kb = tok_path.stat().st_size / 1024

    checksums = {
        "model.onnx": sha256_hex(onnx_path),
        "tokenizer.json": sha256_hex(tok_path),
        "gliner_config.json": sha256_hex(config_path),
    }
    manifest = {
        "model": args.model,
        "quantized": args.quantize,
        "files": checksums,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"✓  model.onnx          {onnx_mb:.1f} MB  sha256:{checksums['model.onnx'][:16]}...")
    print(f"✓  tokenizer.json      {tok_kb:.0f} KB  sha256:{checksums['tokenizer.json'][:16]}...")
    print(f"✓  gliner_config.json  sha256:{checksums['gliner_config.json'][:16]}...")
    print(f"✓  manifest.json")

if __name__ == "__main__":
    main()
