#!/usr/bin/env python3
"""
Benchmark an exported ONNX GLiNER model against Cloak's integration test cases.

Reads testdata/benchmark_cases.jsonl, runs the NER model on each case's input
text, and reports per-case and aggregate precision / recall / F1.

Only evaluates NER-relevant entity types (name, address, hostname, username)
— cases that depend on regex/secrets stages are marked as "non-ner" and
skipped for scoring.

Usage:
  python scripts/benchmark_model.py --model-dir ./edge-export/ --cases testdata/benchmark_cases.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# GLiNER entity labels → Cloak types
# ---------------------------------------------------------------------------
GLINER_LABEL_TO_CLOAK: Dict[str, str] = {
    "person": "NAME",
    "full name": "NAME",
    "name": "NAME",
    "first name": "NAME",
    "last name": "NAME",
    "username": "USERNAME",
    "user": "USERNAME",
    "email address": "EMAIL",
    "email": "EMAIL",
    "phone number": "PHONE",
    "ip address": "IPv4",
    "address": "ADDRESS",
    "physical address": "ADDRESS",
    "hostname": "HOSTNAME",
    "domain name": "HOSTNAME",
    "organization": "ORGANIZATION",
}

# Cloak types that this model should handle.
NER_TYPES = {"NAME", "USERNAME", "ADDRESS", "HOSTNAME", "ORGANIZATION"}

# Cases that require NER capabilities (these should improve once NER is wired).
NER_CASE_IDS = {1, 13, 20, 24, 25}

# ---------------------------------------------------------------------------
# Parsing expected output
# ---------------------------------------------------------------------------
REDACTED_RE = re.compile(r"\[REDACTED - ([A-Z_]+)\]")


def parse_expected(expected: str) -> List[Tuple[int, int, str]]:
    """Find all [REDACTED - TYPE] markers and return (start, end, type) tuples."""
    entities = []
    # Strip markers to compute byte offsets.
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
    expected: List[Tuple[int, int, str]],
    predicted: List[Tuple[int, int, str]],
) -> Dict[str, float]:
    """Compute precision, recall, F1 on span-level matching."""
    tp = 0
    fp = 0
    fn = 0

    matched_expected: Set[int] = set()
    matched_predicted: Set[int] = set()

    for ei, (es, ee, et) in enumerate(expected):
        for pi, (ps, pe, pt) in enumerate(predicted):
            if pi in matched_predicted:
                continue
            # Span overlap with type match
            if et == pt and es < pe and ps < ee:
                tp += 1
                matched_expected.add(ei)
                matched_predicted.add(pi)
                break

    fn = len(expected) - len(matched_expected)
    fp = len(predicted) - len(matched_predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX GLiNER against test cases")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.onnx + tokenizer.json")
    parser.add_argument("--cases", required=True, help="Path to benchmark_cases.jsonl")
    parser.add_argument("--threshold", type=float, default=0.3, help="Entity confidence threshold")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "model.onnx").exists():
        sys.exit(f"ERROR: model.onnx not found in {model_dir}")

    # Load cases
    cases_path = Path(args.cases)
    cases = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    # Load model
    try:
        from gliner import GLiNER
    except ImportError:
        sys.exit("ERROR: gliner not installed. pip install -r scripts/requirements.txt")

    print(f"⟳  Loading model from {model_dir} ...")
    model = GLiNER.from_pretrained(
        str(model_dir),
        load_onnx_model=True,
        load_tokenizer=True,
    )

    # All GLiNER labels we care about
    # (use the most specific labels that cover our types)
    candidate_labels = [
        "person", "full name", "name",
        "username", "user",
        "address", "physical address",
        "hostname", "domain name",
        "organization",
    ]

    print(f"\n{'='*70}")
    print(f"{'Case':<20} {'Expected':<10} {'Predicted':<10} {'TP/FP/FN':<15} {'Result'}")
    print(f"{'='*70}")

    ner_total_tp = 0
    ner_total_fp = 0
    ner_total_fn = 0

    for case in cases:
        cid = case["id"]
        text = case["input"]
        expected = case["expected_output"]

        expected_entities = parse_expected(expected)
        is_ner_case = cid in NER_CASE_IDS

        # Run model
        raw_preds = model.predict_entities(text, candidate_labels, threshold=args.threshold)

        # Convert GLiNER output to Cloak types
        predicted = []
        for ent in raw_preds:
            cloak_type = GLINER_LABEL_TO_CLOAK.get(ent["label"])
            if cloak_type is None:
                continue
            predicted.append((ent["start"], ent["end"], cloak_type))

        if is_ner_case:
            scores = evaluate(expected_entities, predicted)
            ner_total_tp += scores["tp"]
            ner_total_fp += scores["fp"]
            ner_total_fn += scores["fn"]

            status = "PASS" if scores["f1"] >= 0.5 else "FAIL"
        else:
            scores = {"tp": 0, "fp": 0, "fn": 0, "f1": 0.0}
            status = "non-ner"

        print(
            f"{f'#{cid} {case['category']}':<20} "
            f"{len(expected_entities):<10} "
            f"{len(predicted):<10} "
            f"{scores['tp']}/{scores['fp']}/{scores['fn']:<15} "
            f"{status}"
        )

    # Aggregate
    print(f"\n{'='*70}")
    tp = ner_total_tp
    fp = ner_total_fp
    fn = ner_total_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nNER cases only (#1, #13, #20, #24, #25):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    # Exit code for CI
    if f1 < 0.5:
        print("\n⚠️  F1 below 0.5 — export may be degraded. Check model and threshold.")
        sys.exit(1)
    else:
        print("\n✓  Model passes quality gate.")
        sys.exit(0)


if __name__ == "__main__":
    main()
