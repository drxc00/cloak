#!/usr/bin/env python3
"""
Benchmark the exported token-classification model against Cloak's test cases.

Drives the *real* shipped decode path — either via the cloak-nerd sidecar
over NDJSON stdin/stdout (preferred), or via a pure-Python port of the
§6.11 decoder that operates on the ONNX graph directly.

Reads testdata/benchmark_cases.jsonl, runs the model on each case's input
text, and reports per-case and aggregate precision / recall / F1.

Only NER-relevant entity types (NAME, ADDRESS, USERNAME) are scored.
Cases that depend on regex/secrets stages are marked "non-ner" and skipped.

Usage:
  python scripts/benchmark_model.py --model-dir ./onnx-int8/ --cases testdata/benchmark_cases.jsonl
  python scripts/benchmark_model.py --model-dir ./onnx-int8/ --cases testdata/benchmark_cases.jsonl --nerd-bin ./bin/cloak-nerd
"""

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Cloak types that the NER model handles (Core-3).
# ---------------------------------------------------------------------------
NER_TYPES = {"NAME", "ADDRESS", "USERNAME"}

# Cases that require NER capabilities.
NER_CASE_IDS = {1, 13, 20, 24, 25}

# ---------------------------------------------------------------------------
# Parsing expected output
# ---------------------------------------------------------------------------
REDACTED_RE = re.compile(r"\[REDACTED - ([A-Z_]+)\]")


def parse_expected(expected: str) -> list[tuple[int, int, str]]:
    """Find all [REDACTED - TYPE] markers and return (start, end, type) tuples."""
    entities: list[tuple[int, int, str]] = []
    clean = REDACTED_RE.sub("", expected)
    offset = 0
    for m in REDACTED_RE.finditer(expected):
        ent_type = m.group(1)
        start = m.start() - offset
        end = start + (m.end() - m.start())
        entities.append((start, end, ent_type))
        offset += m.end() - m.start()
    return entities


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    expected: list[tuple[int, int, str]],
    predicted: list[tuple[int, int, str]],
) -> dict[str, float]:
    """Compute span-level precision, recall, F1."""
    expected_set: set[tuple[int, int, str]] = set(expected)
    predicted_set: set[tuple[int, int, str]] = set(predicted)

    # Match by span overlap + type match.
    tp = 0
    matched_pred: set[int] = set()

    for ei, (es, ee, et) in enumerate(expected):
        for pi, (ps, pe, pt) in enumerate(predicted):
            if pi in matched_pred:
                continue
            if et == pt and es < pe and ps < ee:
                tp += 1
                matched_pred.add(pi)
                break

    fn = len(expected) - tp
    fp = len(predicted) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Sidecar-backed prediction (preferred — tests the real decode path)
# ---------------------------------------------------------------------------

def predict_via_sidecar(
    texts: list[str],
    nerd_bin: Path,
    model_dir: Path,
    threshold: float = 0.5,
) -> list[list[tuple[int, int, str]]]:
    """
    Spawn cloak-nerd, send NDJSON requests, collect responses.
    Returns one list of (start, end, type) per input text.
    """
    model_onnx = model_dir / "model.onnx"
    tokenizer_json = model_dir / "tokenizer.json"
    config_json = model_dir / "model_config.json"

    if not model_onnx.exists():
        sys.exit(f"ERROR: {model_onnx} not found — run export_model.py first")
    if not tokenizer_json.exists():
        sys.exit(f"ERROR: {tokenizer_json} not found")
    if not config_json.exists():
        sys.exit(f"ERROR: {config_json} not found — run export_model.py first")

    cmd = [
        str(nerd_bin),
        "--model", str(model_onnx),
        "--tokenizer", str(tokenizer_json),
        "--config", str(config_json),
        "--threshold", str(threshold),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    results: list[list[tuple[int, int, str]]] = []

    for text in texts:
        request = {"text": text, "labels": list(NER_TYPES)}
        req_json = json.dumps(request) + "\n"
        proc.stdin.write(req_json.encode())  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

        line = proc.stdout.readline()  # type: ignore[union-attr]
        resp = json.loads(line)

        entities: list[tuple[int, int, str]] = []
        for ent in resp.get("entities", []):
            t = ent["type"]
            # Sidecar uses Core-3 labels from the allow-list directly.
            if t in NER_TYPES:
                entities.append((ent["start"], ent["end"], t))
        results.append(entities)

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait(timeout=30)

    return results


# ---------------------------------------------------------------------------
# Pure-Python ONNX decode (fallback — §6.11 decoder ported to Python)
# ---------------------------------------------------------------------------

def predict_via_onnx(
    texts: list[str],
    model_dir: Path,
    threshold: float = 0.5,
) -> list[list[tuple[int, int, str]]]:
    """
    Run the ONNX graph directly via onnxruntime with the §6.11 BIO decode.
    Used when a cloak-nerd binary is not available.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = model_dir / "model.onnx"
    config_path = model_dir / "model_config.json"
    tok_path = model_dir / "tokenizer.json"

    if not onnx_path.exists():
        sys.exit(f"ERROR: {onnx_path} not found")
    if not config_path.exists():
        sys.exit(f"ERROR: {config_path} not found")

    config = json.loads(config_path.read_text())
    id2label = {int(k): v for k, v in config["id2label"].items()}
    max_len = config.get("max_len", 384)

    session = ort.InferenceSession(str(onnx_path))
    onnx_input_names = [inp.name for inp in session.get_inputs()]
    onnx_output_name = session.get_outputs()[0].name

    tokenizer = AutoTokenizer.from_pretrained(str(tok_path.parent))

    results: list[list[tuple[int, int, str]]] = []

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors="np",
        )

        offsets = enc["offset_mapping"][0]  # [seq, 2]
        seq_len = enc["input_ids"].shape[1]

        # Build ONNX inputs.
        onnx_inputs: dict[str, Any] = {}
        for name in onnx_input_names:
            if name in enc:
                onnx_inputs[name] = enc[name]
            else:
                onnx_inputs[name] = np.zeros((1, seq_len), dtype=np.int64)

        logits = session.run([onnx_output_name], onnx_inputs)[0]  # [1, seq, num_labels]
        probs = _softmax(logits[0])  # [seq, num_labels]
        pred_ids = probs.argmax(axis=-1)  # [seq]

        # §6.11 BIO decoder — merge by contiguous non-O tokens using byte offsets.
        entities: list[dict[str, Any]] = []
        open_span: dict[str, Any] | None = None

        for pos in range(seq_len):
            tok_start, tok_end = offsets[pos]
            if tok_start == tok_end:  # special token
                continue

            label_id = int(pred_ids[pos])
            label_str = id2label.get(label_id, "O")
            score = float(probs[pos, label_id])

            if score < threshold:
                label_str = "O"

            if label_str == "O":
                if open_span is not None:
                    open_span["end"] = tok_end
                    entities.append(open_span)
                    open_span = None
                continue

            if label_str.startswith("B-"):
                if open_span is not None:
                    open_span["end"] = tok_end
                    entities.append(open_span)
                open_span = {
                    "type": label_str[2:],
                    "start": int(tok_start),
                    "end": int(tok_end),
                    "score": score,
                }
            elif label_str.startswith("I-"):
                etype = label_str[2:]
                if open_span is not None and open_span["type"] == etype:
                    open_span["end"] = int(tok_end)
                    open_span["score"] = min(open_span["score"], score)
                elif open_span is not None and open_span["type"] != etype:
                    entities.append(open_span)
                    open_span = None

        if open_span is not None:
            entities.append(open_span)

        results.append([
            (e["start"], e["end"], e["type"])
            for e in sorted(entities, key=lambda x: x["start"])
        ])

    return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark token-classification model against Cloak test cases"
    )
    parser.add_argument("--model-dir", required=True, help="Directory with model.onnx + tokenizer.json + model_config.json")
    parser.add_argument("--cases", required=True, help="Path to benchmark_cases.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entity score threshold")
    parser.add_argument("--nerd-bin", default=None, help="Path to cloak-nerd binary (preferred); falls back to pure-Python ONNX decode")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load cases
    cases_path = Path(args.cases)
    cases: list[dict[str, Any]] = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    texts = [c["input"] for c in cases]

    # Choose prediction backend
    if args.nerd_bin and Path(args.nerd_bin).exists():
        print(f"⟳  Using sidecar: {args.nerd_bin}")
        all_preds = predict_via_sidecar(texts, Path(args.nerd_bin), model_dir, args.threshold)
    else:
        if args.nerd_bin:
            print(f"⚠  {args.nerd_bin} not found — falling back to pure-Python ONNX decode")
        else:
            print("⟳  Pure-Python ONNX decode (sidecar not specified)")
        all_preds = predict_via_onnx(texts, model_dir, args.threshold)

    # Evaluate
    print(f"\n{'='*70}")
    print(f"{'Case':<20} {'Expected':<10} {'Predicted':<10} {'TP/FP/FN':<15} {'Result'}")
    print(f"{'='*70}")

    ner_tp = ner_fp = ner_fn = 0
    case1_ok = False

    for case, preds in zip(cases, all_preds):
        cid = case["id"]
        is_ner = cid in NER_CASE_IDS
        expected = parse_expected(case["expected_output"])

        if is_ner:
            # Only count NER-relevant expected types.
            ner_expected = [(s, e, t) for s, e, t in expected if t in NER_TYPES]
            ner_preds = [(s, e, t) for s, e, t in preds if t in NER_TYPES]
            scores = evaluate(ner_expected, ner_preds)
            ner_tp += int(scores["tp"])
            ner_fp += int(scores["fp"])
            ner_fn += int(scores["fn"])
            status = "PASS" if scores["f1"] >= 0.5 else "FAIL"

            # Case #1 gate: "Neil Patrick Villanueva" → NAME, META untouched.
            if cid == 1:
                name_detected = any(t == "NAME" for _, _, t in preds)
                meta_redacted = any(
                    t not in NER_TYPES or s <= text.find("META") < e
                    for s, e, t in preds
                    for text in [case["input"]]
                )
                # meta_redacted is True if META was hit by a non-NER or a NER span covers it.
                # Let's check more carefully: is any NER span covering "META"?
                meta_pos = case["input"].find("META")
                ner_hits_meta = any(
                    s <= meta_pos < e for s, e, _ in preds
                )
                name_detected = any(
                    t == "NAME" and case["input"][s:e] in ("Neil", "Neil Patrick Villanueva")
                    for s, e, t in preds
                )
                case1_ok = name_detected and not ner_hits_meta
        else:
            scores = {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0}
            status = "non-ner"

        print(
            f"{f'#{cid} {case['category']}':<20} "
            f"{len([e for e in expected if e[2] in NER_TYPES]):<10} "
            f"{len([e for e in preds if e[2] in NER_TYPES]):<10} "
            f"{scores['tp']}/{scores['fp']}/{scores['fn']:<15} "
            f"{status}"
        )

    # Aggregate
    tp = ner_tp
    fp = ner_fp
    fn = ner_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*70}")
    print("NER cases only (#1, #13, #20, #24, #25):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    if case1_ok:
        print("✓  Case #1: NAME detected, META untouched")
    else:
        print("✗  Case #1: FAILED — the §1.1 regression is not fixed")

    # Exit code for CI
    if f1 < 0.5 or not case1_ok:
        print("\n⚠️  Quality gate FAILED")
        sys.exit(1)
    else:
        print("\n✓  Model passes quality gate")
        sys.exit(0)


if __name__ == "__main__":
    main()
