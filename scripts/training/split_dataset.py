#!/usr/bin/env python3
"""
Stratified train / validation / test split from the combined dataset.

Splits rows so each Core-3 label type (NAME, ADDRESS, USERNAME) is
proportionally represented in every split.  Writes one .jsonl file per split
under data/{train,validation,test}/ matching the naming convention
train_model.py expects (e.g. 1en.jsonl).

Usage:
  python scripts/training/split_dataset.py --in data/combined.jsonl \
      --out-dir data/ --train 0.80 --val 0.10 --test 0.10 --seed 42
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
assert _HERE.name == "training", f"Expected scripts/training, got {_HERE}"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from label_map import AI4_TO_CLOAK, CLOAK_TYPES  # noqa: E402


def _core3_set(mask: list[dict]) -> frozenset[str]:
    types: set[str] = set()
    for m in mask:
        ct = AI4_TO_CLOAK.get(m["label"])
        if ct:
            types.add(ct)
    return frozenset(types)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified train/validation/test split"
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Path to the combined jsonl file",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory (will contain train/ validation/ test/ subdirs)",
    )
    parser.add_argument("--train", type=float, default=0.80, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.10, help="Validation fraction")
    parser.add_argument("--test", type=float, default=0.10, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.001:
        print(f"ERROR: split fractions must sum to 1.0 (got {total})", file=sys.stderr)
        sys.exit(1)

    project_root = _HERE.parent.parent
    out_dir = project_root / args.out_dir

    # Load.
    print(f"Loading {args.input_path} ...")
    rows: list[dict] = []
    with open(args.input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"  {len(rows):,} rows loaded")

    # Bucket by Core-3 label set.
    buckets: dict[frozenset[str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[_core3_set(row.get("privacy_mask", []))].append(row)

    print(f"\n  Buckets by Core-3 label set:")
    for key in sorted(buckets, key=lambda k: (len(k), sorted(k))):
        label_str = "+".join(sorted(key)) if key else "(no Core-3 label)"
        print(f"    {label_str:30s} {len(buckets[key]):>7,} rows")

    # Stratified split.
    rng = np.random.default_rng(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for _type_set, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        # Test gets the remainder to ensure exact split.
        n_test = n - n_train - n_val
        if n_test < 1:
            # Not enough rows — give everything to train.
            train_rows.extend(group)
            continue
        train_rows.extend(group[:n_train])
        val_rows.extend(group[n_train : n_train + n_val])
        test_rows.extend(group[n_train + n_val :])

    # Shuffle each split internally (so order isn't bucketed).
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    # Assign split field.
    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "validation"
    for row in test_rows:
        row["split"] = "test"

    # Write.
    splits = {
        "train": (out_dir / "train", train_rows),
        "validation": (out_dir / "validation", val_rows),
        "test": (out_dir / "test", test_rows),
    }

    print(f"\n  Split ({args.train:.0%}/{args.val:.0%}/{args.test:.0%}):")
    for name, (subdir, split_rows) in splits.items():
        subdir.mkdir(parents=True, exist_ok=True)
        out_file = subdir / "1en.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for row in split_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        pct = len(split_rows) / len(rows) * 100
        print(f"    {name:12s} {len(split_rows):>7,} rows ({pct:.1f}%)  →  {out_file}")

    # Per-split label distribution.
    print(f"\n  Label spans per split:")
    for name, (_, split_rows) in splits.items():
        from collections import Counter

        lc: Counter[str] = Counter()
        for row in split_rows:
            for m in row.get("privacy_mask", []):
                ct = AI4_TO_CLOAK.get(m["label"])
                if ct:
                    lc[ct] += 1
        total_s = sum(lc.values())
        parts = ", ".join(
            f"{t}={lc[t]:,} ({lc[t]/total_s*100:.1f}%)" if total_s else f"{t}=0"
            for t in CLOAK_TYPES
        )
        print(f"    {name:12s} {parts}")

    print(
        f"\n✓  {len(rows):,} rows split into {out_dir}/{{train,validation,test}}/1en.jsonl"
    )
    print(
        f"   Next: python scripts/training/train_model.py --data-dir {out_dir}/ --langs 1en"
    )


if __name__ == "__main__":
    main()
