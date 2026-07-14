#!/usr/bin/env python3
"""
Build one canonical combined dataset from all English PII sources.

Sources:
  - ai4privacy train  (datasets/train/1en.jsonl)
  - ai4privacy val    (datasets/validation/1en.jsonl)
  - Augment synthetic (datasets/augmented/augmented.jsonl)
  - Few-NERD          (datasets/augmented/fewnerd.jsonl)

Output: data/combined.jsonl — normalized, deduplicated, shuffle-seeded.

Usage:
  python scripts/training/build_combined.py --out data/combined.jsonl --seed 42
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
assert _HERE.name == "training", f"Expected scripts/training, got {_HERE}"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from label_map import AI4_TO_CLOAK, CLOAK_TYPES  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"  ⚠  skipping missing: {path}", file=sys.stderr)
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize(row: dict, uid: int) -> dict:
    """Drop extra fields (mbert_*, label_index) and assign a stable UID."""
    privacy_mask: list[dict] = []
    for m in row.get("privacy_mask", []):
        privacy_mask.append(
            {
                "start": m["start"],
                "end": m["end"],
                "label": m["label"],
                "value": m.get("value", ""),
            }
        )
    return {
        "source_text": row["source_text"],
        "language": row.get("language", "en"),
        "locale": row.get("locale", "US"),
        "split": "train",  # placeholder — will be overwritten by split_dataset.py
        "privacy_mask": privacy_mask,
        "uid": uid,
        "masked_text": "",
    }


def _deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by (source_text, frozenset of span tuples)."""
    seen: set[tuple[str, frozenset]] = set()
    unique: list[dict] = []
    dupes = 0
    for row in rows:
        spans = frozenset(
            (m["start"], m["end"], m["label"]) for m in row["privacy_mask"]
        )
        key = (row["source_text"], spans)
        if key not in seen:
            seen.add(key)
            unique.append(row)
        else:
            dupes += 1
    return unique, dupes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one canonical combined dataset from all English sources"
    )
    parser.add_argument(
        "--out",
        default="data/combined.jsonl",
        help="Output path for the combined jsonl file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    args = parser.parse_args()

    project_root = _HERE.parent.parent  # scripts/training → scripts → repo root
    datasets_root = _HERE / "datasets"

    # Gather all sources — glob all language files from ai4privacy train/val.
    sources: list[tuple[str, Path]] = []
    for lang_file in sorted((datasets_root / "train").glob("*.jsonl")):
        sources.append((f"ai4privacy train/{lang_file.stem}", lang_file))
    for lang_file in sorted((datasets_root / "validation").glob("*.jsonl")):
        sources.append((f"ai4privacy val/{lang_file.stem}", lang_file))
    sources += [
        ("augment synthetic", datasets_root / "augmented" / "augmented.jsonl"),
        ("Few-NERD", datasets_root / "augmented" / "fewnerd.jsonl"),
    ]

    all_rows: list[dict] = []
    uid = 0
    for name, path in sources:
        raw = _load_jsonl(path)
        for row in raw:
            all_rows.append(_normalize(row, uid))
            uid += 1
        print(f"  {name:25s} {len(raw):>7,} rows  →  {path}")

    print(f"  {'─' * 60}")
    print(f"  {'Total loaded':25s} {len(all_rows):>7,} rows")

    # Deduplicate.
    all_rows, dupes = _deduplicate(all_rows)
    if dupes:
        print(
            f"  {'Deduplicated':25s} {dupes:>7,} rows removed  ({len(all_rows):,} remaining)"
        )

    # Shuffle with fixed seed.
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    # Re-assign UIDs after shuffle for clean ordering.
    for i, row in enumerate(all_rows):
        row["uid"] = i

    # Write.
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Label summary.
    label_counts: Counter[str] = Counter()
    for row in all_rows:
        for m in row["privacy_mask"]:
            ct = AI4_TO_CLOAK.get(m["label"])
            if ct:
                label_counts[ct] += 1
    total_spans = sum(label_counts.values())
    breakdown = ", ".join(
        f"{t}={label_counts[t]:,} ({label_counts[t]/total_spans*100:.1f}%)"
        for t in CLOAK_TYPES
    )
    print(f"\n✓  {len(all_rows):,} rows  →  {out_path}")
    print(f"   Label spans: {breakdown}")


if __name__ == "__main__":
    main()
