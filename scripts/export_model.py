#!/usr/bin/env python3
"""
Export a trained fixed-label token classifier to ONNX.

Replaces the GLiNER export path with standard token-classification export
via optimum-cli.  Supports optional INT8 dynamic quantization.

Produces in the output directory:
  model.onnx          — ONNX graph (FP32, or INT8 if --quantize)
  tokenizer.json       — fast tokenizer from the training checkpoint
  model_config.json    — sidecar config (id2label, max_len, input_names, model_type)

Usage:
  python scripts/export_model.py --checkpoint ./trained/ --out ./onnx-fp32/
  python scripts/export_model.py --checkpoint ./trained/ --out ./onnx-int8/ --quantize
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

_here = Path(__file__).resolve().parent
_training = _here / "training"
if str(_training) not in sys.path:
    sys.path.insert(0, str(_training))

from label_map import ID2LABEL, LABEL2ID  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rewrite_embedded(path: Path) -> None:
    """Re-save an ONNX model so all tensors are embedded (no external data)."""
    data_path = path.with_suffix(path.suffix + ".data")
    if not data_path.exists():
        return  # already embedded
    print("  Re-saving ONNX without external data …")
    model = onnx.load(str(path))
    onnx.save(model, str(path))
    # onnx.save with default args embeds all tensors, but external-data
    # files are not auto-cleaned. Remove the stale data file.
    data_path.unlink(missing_ok=True)
    print(f"  ✓  External data merged — {data_path.name} removed")


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a token-classification checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the trained HF checkpoint (contains config.json) — output of train_model.py",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--quantize", action="store_true", default=False,
        help="Apply INT8 dynamic quantization to model.onnx",
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version"
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (checkpoint / "config.json").exists():
        sys.exit(f"ERROR: {checkpoint} does not contain config.json — is it a trained HF checkpoint?")

    # ------------------------------------------------------------------
    # 1. Export FP32 with torch.onnx.export directly (avoids optimum version
    #    conflicts between torch ≥2.10 and optimum <2.0).
    # ------------------------------------------------------------------
    print(f"⟳  Loading PyTorch model from {checkpoint} ...")
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    pt_model = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    pt_model.eval()

    # Trace with a dummy input.
    dummy_text = "John Smith lives at 742 Evergreen Terrace, Springfield 62704."
    dummy_enc = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=64)

    fp32_onnx = out_dir / "model.onnx"

    print(f"  Exporting to {fp32_onnx} ...")
    torch.onnx.export(
        pt_model,
        (dummy_enc["input_ids"], dummy_enc["attention_mask"]),
        str(fp32_onnx),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"✓  FP32 model.onnx exported")
    if not fp32_onnx.exists():
        sys.exit(f"ERROR: torch.onnx.export did not produce {fp32_onnx}")

    # Re-save without external data so the model ships as a single file.
    # torch.onnx.export may spill large tensors into model.onnx.data when
    # the total size exceeds the default threshold; cloaks init only
    # downloads model.onnx, so external data breaks the sidecar at load.
    _rewrite_embedded(fp32_onnx)

    # ------------------------------------------------------------------
    # 2. Validate ONNX model & extract input names
    # ------------------------------------------------------------------
    onnx_model = onnx.load(str(fp32_onnx))
    onnx.checker.check_model(onnx_model)

    onnx_inputs = [inp.name for inp in onnx_model.graph.input]
    onnx_outputs = [out.name for out in onnx_model.graph.output]
    print(f"  ONNX inputs:  {onnx_inputs}")
    print(f"  ONNX outputs: {onnx_outputs}")

    # Verify output shape is [batch, seq, num_labels]
    output_shape = [
        dim.dim_value if dim.dim_value else str(dim.dim_param)
        for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    expected_num_labels = len(LABEL2ID)
    if output_shape[-1] != expected_num_labels:
        sys.exit(
            f"ERROR: ONNX output last dim is {output_shape[-1]}, "
            f"but LABEL2ID has {expected_num_labels} labels. Shapes: {output_shape}"
        )
    print(f"  Output shape: {output_shape} ✓ ({expected_num_labels} labels)")

    # ------------------------------------------------------------------
    # 3. Copy tokenizer from checkpoint
    # ------------------------------------------------------------------
    tok_src = checkpoint / "tokenizer.json"
    if not tok_src.exists():
        sys.exit(f"ERROR: tokenizer.json not found in {checkpoint}")
    shutil.copy2(tok_src, out_dir / "tokenizer.json")
    print(f"✓  tokenizer.json")

    # ------------------------------------------------------------------
    # 4. Quantize (optional)
    # ------------------------------------------------------------------
    if args.quantize:
        print("⟳  Applying INT8 dynamic quantization ...")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        fp32_backup = out_dir / "model_fp32.onnx"
        shutil.move(str(fp32_onnx), str(fp32_backup))
        quantize_dynamic(
            str(fp32_backup),
            str(fp32_onnx),
            weight_type=QuantType.QInt8,
        )
        fp32_backup.unlink()
        _rewrite_embedded(fp32_onnx)
        mb = fp32_onnx.stat().st_size / (1024 * 1024)
        print(f"✓  INT8 model.onnx  ({mb:.1f} MB)")
    else:
        mb = (out_dir / "model.onnx").stat().st_size / (1024 * 1024)
        print(f"✓  FP32 model.onnx  ({mb:.1f} MB)")

    # ------------------------------------------------------------------
    # 5. Write sidecar model config (§6.12)
    # ------------------------------------------------------------------
    # Read max_len from the checkpoint's config.json.
    hf_config = json.loads((checkpoint / "config.json").read_text())
    max_len = hf_config.get("max_position_embeddings", 512)

    # Read training manifest if available for the actual training max_len.
    manifest_path = checkpoint / "training_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        max_len = manifest.get("max_len", max_len)

    model_config = {
        "model_type": "token-classifier-v1",
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "max_len": max_len,
        "onnx_inputs": onnx_inputs,
        "onnx_outputs": onnx_outputs,
    }
    config_path = out_dir / "model_config.json"
    config_path.write_text(json.dumps(model_config, indent=2) + "\n")
    print(f"✓  model_config.json  (model_type={model_config['model_type']}, max_len={max_len})")

    # ------------------------------------------------------------------
    # 6. Manifest + checksums
    # ------------------------------------------------------------------
    checksums = {
        "model.onnx": sha256_hex(out_dir / "model.onnx"),
        "tokenizer.json": sha256_hex(out_dir / "tokenizer.json"),
        "model_config.json": sha256_hex(config_path),
    }
    export_manifest = {
        "checkpoint": str(checkpoint),
        "quantized": args.quantize,
        "files": checksums,
    }
    manifest_out = out_dir / "manifest.json"
    manifest_out.write_text(json.dumps(export_manifest, indent=2) + "\n")

    onnx_mb = (out_dir / "model.onnx").stat().st_size / (1024 * 1024)
    tok_kb = (out_dir / "tokenizer.json").stat().st_size / 1024
    print(f"\n{'─'*50}")
    print(f"✓  model.onnx          {onnx_mb:.1f} MB  sha256:{checksums['model.onnx'][:16]}...")
    print(f"✓  tokenizer.json      {tok_kb:.0f} KB  sha256:{checksums['tokenizer.json'][:16]}...")
    print(f"✓  model_config.json  sha256:{checksums['model_config.json'][:16]}...")
    print(f"✓  manifest.json")


if __name__ == "__main__":
    main()
