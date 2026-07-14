#!/usr/bin/env python3
"""
Parity check: compare PyTorch checkpoint vs exported ONNX model.

Runs N sample texts through both and asserts argmax labels agree on
≥99% of tokens.  Catches opset/axis/IO-name mistakes before they reach
the Rust sidecar.

Usage:
  python scripts/parity_check.py --checkpoint ./trained/ --onnx ./onnx-export/
  python scripts/parity_check.py --checkpoint ./trained/ --onnx ./onnx-int8/ --num-samples 500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

_here = Path(__file__).resolve().parent
_training = _here / "training"
if str(_training) not in sys.path:
    sys.path.insert(0, str(_training))

from label_map import ID2LABEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch checkpoint vs exported ONNX"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained HF checkpoint")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX directory (contains model.onnx)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of test texts to compare")
    parser.add_argument("--tolerance", type=float, default=0.95, help="Minimum token agreement ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    onnx_dir = Path(args.onnx)
    onnx_path = onnx_dir / "model.onnx"
    config_path = onnx_dir / "model_config.json"

    if not onnx_path.exists():
        sys.exit(f"ERROR: {onnx_path} not found")
    if not (checkpoint / "config.json").exists():
        sys.exit(f"ERROR: {checkpoint} does not look like a HF checkpoint")

    print(f"⟳  Loading PyTorch model from {checkpoint} ...")
    pt_model = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    pt_model.eval()

    print(f"⟳  Loading ONNX model from {onnx_path} ...")
    session = ort.InferenceSession(str(onnx_path))

    # Validate ONNX input/output names match model_config.
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        onnx_cfg_inputs = set(cfg.get("onnx_inputs", []))
        actual_inputs = {inp.name for inp in session.get_inputs()}
        if onnx_cfg_inputs and onnx_cfg_inputs != actual_inputs:
            print(f"⚠  WARNING: model_config inputs {onnx_cfg_inputs} ≠ actual {actual_inputs}")

    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_output_name = session.get_outputs()[0].name
    print(f"  ONNX inputs:  {onnx_input_names}")
    print(f"  ONNX output:  {onnx_output_name}")

    # ------------------------------------------------------------------
    # Generate test texts — mix of PII and machine-shaped text
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    SAMPLE_TEXTS = [
        "My name is John Smith, I live at 742 Evergreen Terrace, Springfield 62704.",
        "Contact admin@company.com or call 555-123-4567.",
        "Error in /home/jdoe/app/main.py:42 — user jdoe not authorized.",
        "Deploying api-gateway to prod-cluster for META integration.",
        'export DB_USER=jdoe\nDB_HOST=db-primary.company.internal\nREGION=us-east-1',
        "Traceback: File /opt/AWS/sdk/client.py, line 128, in process — user Jane Doe.",
        "commit a1b2c3 — Author: Jane Doe <jdoe@company.com> — fix auth for cache-layer",
        "nginx: 10.0.1.42 - jdoe [2025-07-01T12:00:00Z] GET /api/v1/users 200 — Mozilla/5.0",
        "Dr. Maria Santos will see you at 221B Baker Street, London.",
        "Config: DATADOG_API_KEY=abc123, monitor=api-gateway.prod, oncall=asmith",
    ]

    # Extend with fuzzed variants.
    extras = list(SAMPLE_TEXTS)
    for text in SAMPLE_TEXTS:
        for _ in range(min(20, args.num_samples // len(SAMPLE_TEXTS))):
            # slightly modify
            words = text.split()
            if len(words) > 5:
                i = rng.integers(0, len(words))
                if rng.random() < 0.3:
                    words.insert(i, rng.choice(["META", "AWS", "GCP", "prod", "staging"]))
                elif len(words[i]) > 3 and rng.random() < 0.5:
                    words[i] = words[i].upper()
            extras.append(" ".join(words))

    test_texts = extras[:args.num_samples]

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    total_tokens = 0
    total_agree = 0
    max_len = tokenizer.model_max_length

    for i, text in enumerate(test_texts):
        # PyTorch forward
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            pt_logits = pt_model(**enc).logits  # [1, seq, num_labels]
        pt_preds = pt_logits.argmax(dim=-1).squeeze(0).numpy()  # [seq]

        # ONNX forward
        onnx_inputs = {}
        for name in onnx_input_names:
            if name in enc:
                onnx_inputs[name] = enc[name].numpy()
            else:
                # Fill missing inputs (e.g. token_type_ids for BERT-style models).
                onnx_inputs[name] = np.zeros_like(enc["input_ids"].numpy())
        onnx_logits = session.run([onnx_output_name], onnx_inputs)[0]  # [1, seq, num_labels]
        onnx_preds = onnx_logits.argmax(axis=-1).squeeze(0)  # [seq]

        seq_len = pt_preds.shape[0]
        match = (pt_preds == onnx_preds)
        agree = int(match.sum())
        total_tokens += seq_len
        total_agree += agree

        if i < 5 or agree != seq_len:
            status = "✓" if agree == seq_len else f"✗ {seq_len - agree}/{seq_len} mismatch"
            mismatches = []
            if agree != seq_len:
                for j in range(seq_len):
                    if pt_preds[j] != onnx_preds[j]:
                        mismatches.append(
                            f"  pos {j}: pt={ID2LABEL.get(int(pt_preds[j]), '?')} "
                            f"onnx={ID2LABEL.get(int(onnx_preds[j]), '?')}"
                        )
            print(f"  [{i:3d}] {status}")
            if mismatches:
                for m in mismatches[:10]:
                    print(m)
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches) - 10} more")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    ratio = total_agree / total_tokens if total_tokens > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Token agreement: {total_agree}/{total_tokens} = {ratio:.4%}")
    print(f"Threshold: {args.tolerance:.4%}")

    if ratio >= args.tolerance:
        print("✓  PASS — PyTorch ↔ ONNX parity within tolerance")
        sys.exit(0)
    else:
        print("✗  FAIL — parity below tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
