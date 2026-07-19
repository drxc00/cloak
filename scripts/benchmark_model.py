#!/usr/bin/env python3
"""
Benchmark the exported token-classification model against Cloak's test cases.

Replicates the *production* NER path (internal/stages/ner/stage.go):

  1. Pre-claiming — spans owned by the regex/secrets stages (EMAIL, IPv4,
     JWT, …) are carved out first, using the benchmark's expected output as
     ground truth for those stages. The model only sees the leftover gaps,
     exactly like `Stage.Detect` sends gap chunks to the sidecar.
  2. Inference — via the cloak-nerd sidecar over NDJSON (preferred, byte
     offsets) or a pure-Python ONNX decode fallback (char offsets).
  3. Post-processing — same-type spans separated by connector "glue" are
     merged, and ADDRESS spans are extended backwards over bare house
     numbers, mirroring mergeAdjacentSpans().

Every case is scored: cases with no expected NAME/ADDRESS/USERNAME spans
(hard negatives, regex-only cases) contribute false positives if the model
fires inside an unclaimed gap.

Reported metrics:
  - typed span P/R/F1 (overlap + type match)     ← primary, gates CI
  - type-agnostic P/R/F1 ("was it redacted at all?")
  - per-type breakdown

Usage:
  python scripts/benchmark_model.py --model-dir ./model-export-edge/ \
      --cases testdata/benchmark_cases.jsonl --nerd-bin ./bin/cloak-nerd
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cloak types that the NER model handles (Core-3).
# ---------------------------------------------------------------------------
NER_TYPES = {"NAME", "ADDRESS", "USERNAME"}

# Marker types are mixed-case ("IPv4", "GITHUB_PAT").
REDACTED_RE = re.compile(r"\[REDACTED - ([A-Za-z0-9_]+)\]")

# Port of internal/stages/ner/stage.go post-processing.
CONNECTOR_RE = re.compile(r"^,?\s*[A-Z0-9]{0,6}\s*,?\s*$")
HOUSE_NUMBER_RE = re.compile(r"(?:^|\s)(\d{1,6}[A-Za-z]?)\s*$")

Span = tuple[int, int, str]  # (char_start, char_end, type)


# ---------------------------------------------------------------------------
# Expected-output parsing: recover true entity spans in the input text
# ---------------------------------------------------------------------------

def parse_expected(input_text: str, expected: str) -> list[Span]:
    """
    Align `expected` (with [REDACTED - T] markers) against `input_text` and
    return the true (start, end, type) spans of the redacted entities.

    Falls back to marker-length spans if alignment fails.
    """
    markers = list(REDACTED_RE.finditer(expected))
    if not markers:
        return []

    # Literal segments between markers: seg0 [T1] seg1 [T2] seg2 ...
    segments: list[str] = []
    prev_end = 0
    for m in markers:
        segments.append(expected[prev_end:m.start()])
        prev_end = m.end()
    segments.append(expected[prev_end:])

    spans: list[Span] = []
    pos = 0
    ok = input_text.startswith(segments[0])
    if ok:
        pos = len(segments[0])
        for m, next_seg in zip(markers, segments[1:]):
            if next_seg:
                nxt = input_text.find(next_seg, pos)
                if nxt < pos:
                    ok = False
                    break
                spans.append((pos, nxt, m.group(1)))
                pos = nxt + len(next_seg)
            else:
                # Marker at end of text (or adjacent markers — rare).
                spans.append((pos, len(input_text), m.group(1)))
                pos = len(input_text)

    if ok:
        return spans

    # Fallback: prefix positions are right, entity lengths unknown.
    spans = []
    offset = 0
    for m in markers:
        start = m.start() - offset
        spans.append((start, start + (m.end() - m.start()), m.group(1)))
        offset += m.end() - m.start()
    return spans


# ---------------------------------------------------------------------------
# Production-path replication: gap chunking + span post-processing
# ---------------------------------------------------------------------------

def gap_chunks(text: str, claimed: list[Span]) -> list[tuple[str, int]]:
    """Trimmed unclaimed regions as (chunk_text, char_offset) — Stage.Detect."""
    chunks: list[tuple[str, int]] = []

    def add(raw: str, base: int) -> None:
        gap = raw.strip()
        if gap:
            chunks.append((gap, base + raw.index(gap)))

    pos = 0
    for start, end, _ in sorted(claimed):
        if start > pos:
            add(text[pos:start], pos)
        pos = max(pos, end)
    if pos < len(text):
        add(text[pos:], pos)
    return chunks


def merge_adjacent_spans(text: str, spans: list[Span]) -> list[Span]:
    """Port of mergeAdjacentSpans(): connector merge + house-number extend."""
    if not spans:
        return spans

    spans = sorted(spans)
    merged: list[Span] = []
    cur_s, cur_e, cur_t = spans[0]
    for s, e, t in spans[1:]:
        if t == cur_t and CONNECTOR_RE.match(text[cur_e:s]):
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e, cur_t))
        cur_s, cur_e, cur_t = s, e, t
    merged.append((cur_s, cur_e, cur_t))

    out: list[Span] = []
    for s, e, t in merged:
        if t == "ADDRESS":
            m = HOUSE_NUMBER_RE.search(text[:s])
            if m:
                s = m.start(1)
        out.append((s, e, t))
    return out


# ---------------------------------------------------------------------------
# Prediction backends
# ---------------------------------------------------------------------------

class SidecarBackend:
    """cloak-nerd over NDJSON. Returns byte offsets (Go slices bytes)."""

    def __init__(self, nerd_bin: Path, model_dir: Path, threshold: float):
        for f in ("model.onnx", "tokenizer.json", "model_config.json"):
            if not (model_dir / f).exists():
                sys.exit(f"ERROR: {model_dir / f} not found — run export_model.py first")
        self.proc = subprocess.Popen(
            [
                str(nerd_bin),
                "--model", str(model_dir / "model.onnx"),
                "--tokenizer", str(model_dir / "tokenizer.json"),
                "--config", str(model_dir / "model_config.json"),
                "--threshold", str(threshold),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def predict(self, chunk: str) -> list[Span]:
        req = json.dumps({"text": chunk, "labels": sorted(NER_TYPES)}) + "\n"
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(req.encode())
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())

        raw = chunk.encode("utf-8")
        spans: list[Span] = []
        for ent in resp.get("entities", []):
            if ent["type"] not in NER_TYPES:
                continue
            # Byte offsets → char offsets.
            cs = len(raw[: ent["start"]].decode("utf-8", errors="replace"))
            ce = len(raw[: ent["end"]].decode("utf-8", errors="replace"))
            spans.append((cs, ce, ent["type"]))
        return spans

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)


class OnnxBackend:
    """Pure-Python ONNX decode (§6.11). Offsets are already char-based."""

    def __init__(self, model_dir: Path, threshold: float):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.threshold = threshold

        config = json.loads((model_dir / "model_config.json").read_text())
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.max_len = config.get("max_len", 384)

        self.session = ort.InferenceSession(str(model_dir / "model.onnx"))
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_len)

    def predict(self, chunk: str) -> list[Span]:
        np = self.np
        enc = self.tokenizer.encode(chunk)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)

        inputs: dict[str, Any] = {}
        for name in self.input_names:
            if name == "input_ids":
                inputs[name] = ids
            elif name == "attention_mask":
                inputs[name] = mask
            else:
                inputs[name] = np.zeros_like(ids)

        logits = self.session.run([self.output_name], inputs)[0][0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        pred_ids = probs.argmax(axis=-1)

        spans: list[Span] = []
        open_span: list[Any] | None = None  # [start, end, type]

        for pos, (tok_start, tok_end) in enumerate(enc.offsets):
            if tok_start == tok_end:  # special token
                continue
            label = self.id2label.get(int(pred_ids[pos]), "O")
            if float(probs[pos, pred_ids[pos]]) < self.threshold:
                label = "O"

            if label == "O":
                if open_span:
                    spans.append(tuple(open_span))
                    open_span = None
            elif label.startswith("B-"):
                if open_span:
                    spans.append(tuple(open_span))
                open_span = [tok_start, tok_end, label[2:]]
            elif label.startswith("I-"):
                if open_span and open_span[2] == label[2:]:
                    open_span[1] = tok_end
                elif open_span:
                    spans.append(tuple(open_span))
                    open_span = None

        if open_span:
            spans.append(tuple(open_span))
        return [(s, e, t) for s, e, t in spans if t in NER_TYPES]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def match_spans(
    expected: list[Span], predicted: list[Span], typed: bool
) -> tuple[int, int, int, set[int]]:
    """Greedy overlap matching. Returns (tp, fp, fn, matched_pred_indices)."""
    matched: set[int] = set()
    tp = 0
    for es, ee, et in expected:
        for i, (ps, pe, pt) in enumerate(predicted):
            if i in matched:
                continue
            if (not typed or et == pt) and es < pe and ps < ee:
                tp += 1
                matched.add(i)
                break
    return tp, len(predicted) - tp, len(expected) - tp, matched


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark token-classification model against Cloak test cases"
    )
    parser.add_argument("--model-dir", required=True, help="Directory with model.onnx + tokenizer.json + model_config.json")
    parser.add_argument("--cases", required=True, help="Path to benchmark_cases.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entity score threshold (matches sidecar default)")
    parser.add_argument("--nerd-bin", default=None, help="Path to cloak-nerd binary (preferred); falls back to pure-Python ONNX decode")
    parser.add_argument("--min-f1", type=float, default=0.70, help="Quality gate: minimum typed span F1")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cases = [
        json.loads(line)
        for line in Path(args.cases).read_text().splitlines()
        if line.strip()
    ]

    if args.nerd_bin and Path(args.nerd_bin).exists():
        print(f"⟳  Backend: sidecar ({args.nerd_bin})")
        backend: Any = SidecarBackend(Path(args.nerd_bin), model_dir, args.threshold)
    else:
        if args.nerd_bin:
            print(f"⚠  {args.nerd_bin} not found — falling back to pure-Python ONNX decode")
        else:
            print("⟳  Backend: pure-Python ONNX decode")
        backend = OnnxBackend(model_dir, args.threshold)

    header = f"{'Case':<38} {'Exp':>3} {'Pred':>4} {'TP/FP/FN':>9}  Result"
    print(f"\n{header}\n{'=' * len(header)}")

    TP = FP = FN = 0
    ag_tp = ag_fp = ag_fn = 0
    per_type: dict[str, list[int]] = {t: [0, 0, 0] for t in sorted(NER_TYPES)}
    failures: list[tuple[Any, list[str], list[str]]] = []
    case1_ok = None

    for case in cases:
        text = case["input"]
        all_expected = parse_expected(text, case["expected_output"])
        expected = [s for s in all_expected if s[2] in NER_TYPES]
        claimed = [s for s in all_expected if s[2] not in NER_TYPES]

        # Production path: predict per unclaimed gap, then post-process globally.
        predicted: list[Span] = []
        for chunk, offset in gap_chunks(text, claimed):
            for s, e, t in backend.predict(chunk):
                predicted.append((offset + s, offset + e, t))
        predicted = merge_adjacent_spans(text, sorted(set(predicted)))

        tp, fp, fn, matched = match_spans(expected, predicted, typed=True)
        a_tp, a_fp, a_fn, _ = match_spans(expected, predicted, typed=False)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        ag_tp, ag_fp, ag_fn = ag_tp + a_tp, ag_fp + a_fp, ag_fn + a_fn

        for _, _, t in expected:
            per_type[t][2] += 1
        for i, (_, _, t) in enumerate(predicted):
            per_type[t][0 if i in matched else 1] += 1

        if case["id"] == 1:
            meta_pos = text.find("META")
            case1_ok = any(t == "NAME" for _, _, t in predicted) and not any(
                s <= meta_pos < e for s, e, _ in predicted
            )

        status = "ok" if not (fp or fn) else ("MISS" if fn and not fp else "FP" if fp and not fn else "MISS+FP")
        if not expected and not predicted:
            status = "ok (neg)"
        print(
            f"#{case['id']:<4}{case['category']:<33} {len(expected):>3} {len(predicted):>4} "
            f"{tp:>3}/{fp}/{fn}   {status}"
        )
        if fp or fn:
            failures.append((
                case,
                [f"{text[s:e]!r}:{t}" for s, e, t in predicted],
                [f"{text[s:e]!r}:{t}" for s, e, t in expected],
            ))

    backend.close()

    p, r, f1 = prf(TP, FP, FN)
    ap, ar, af1 = prf(ag_tp, ag_fp, ag_fn)

    print(f"\n{'=' * 62}")
    print(f"Typed span metrics (primary):   P={p:.4f}  R={r:.4f}  F1={f1:.4f}   TP={TP} FP={FP} FN={FN}")
    print(f"Type-agnostic ('redacted at all'): P={ap:.4f}  R={ar:.4f}  F1={af1:.4f}")
    print("\nPer-type breakdown:")
    print(f"  {'type':<10} {'recall':>14} {'false-pos':>10}")
    for t, (tp_, fp_, total) in per_type.items():
        rec = tp_ / total if total else 0.0
        print(f"  {t:<10} {rec:>8.2f} ({tp_}/{total}) {fp_:>10}")

    if failures:
        print(f"\nFailing cases ({len(failures)}):")
        for case, preds, exps in failures:
            print(f"  #{case['id']} {case['category']}")
            print(f"      expected:  {', '.join(exps) if exps else '(none)'}")
            print(f"      predicted: {', '.join(preds) if preds else '(none)'}")

    print(f"\n{'=' * 62}")
    gate_ok = True
    if f1 < args.min_f1:
        print(f"✗  Typed F1 {f1:.4f} < gate {args.min_f1:.2f}")
        gate_ok = False
    else:
        print(f"✓  Typed F1 {f1:.4f} ≥ gate {args.min_f1:.2f}")
    if case1_ok is None:
        print("⚠  Case #1 not found in suite — regression check skipped")
    elif case1_ok:
        print("✓  Case #1: NAME detected, META untouched")
    else:
        print("✗  Case #1: FAILED — the §1.1 regression is not fixed")
        gate_ok = False

    if not gate_ok:
        print("\n⚠️  Quality gate FAILED")
        sys.exit(1)
    print("\n✓  Model passes quality gate")


if __name__ == "__main__":
    main()
