#!/usr/bin/env python3
"""
Single-entry orchestration — unify augmentation + Few-NERD into one directory.

Runs augment.py and fewnerd_to_jsonl.py as subprocesses (shared seed / output dir),
then prints a combined per-label row count using the canonical label_map so a human
can sanity-check the mix before launching a full training run.

Usage:
  python scripts/training/prepare_dataset.py --out ./augmented/ --seed 42 \\
      --synthetic-rows 10000 --fewnerd-person-rows 8000 --fewnerd-negative-rows 2000
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
assert _HERE.name == "training", f"Expected scripts/training, got {_HERE}"

# Ensure scripts/training is on sys.path so we can import label_map.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from label_map import AI4_TO_CLOAK, CLOAK_TYPES  # noqa: E402


def _count_labels(jsonl_dir: Path) -> Counter[str]:
    """Count per-Cloak-type span occurrences across all *.jsonl in a directory."""
    counts: Counter[str] = Counter()
    for fp in sorted(jsonl_dir.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for m in row.get("privacy_mask", []):
                    ct = AI4_TO_CLOAK.get(m["label"])
                    if ct:
                        counts[ct] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unify augment.py + fewnerd_to_jsonl.py into one augmented directory"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for augmented .jsonl files"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (shared)")
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=10000,
        help="Rows for augment.py to generate",
    )
    parser.add_argument(
        "--fewnerd-person-rows",
        type=int,
        default=8000,
        help="Max person-bearing rows from Few-NERD",
    )
    parser.add_argument(
        "--fewnerd-negative-rows",
        type=int,
        default=2000,
        help="Max negative (zero-span) rows from Few-NERD",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run augment.py
    print("── Running augment.py ──", flush=True)
    aug_argv = [
        sys.executable,
        str(_HERE / "augment.py"),
        "--out",
        str(out_dir),
        "--rows",
        str(args.synthetic_rows),
        "--seed",
        str(args.seed),
    ]
    result = subprocess.run(aug_argv, capture_output=False, text=True)
    if result.returncode != 0:
        print(
            f"ERROR: augment.py failed with code {result.returncode}", file=sys.stderr
        )
        sys.exit(result.returncode)

    # 2. Run fewnerd_to_jsonl.py
    print("\n── Running fewnerd_to_jsonl.py ──", flush=True)
    fewnerd_argv = [
        sys.executable,
        str(_HERE / "fewnerd_to_jsonl.py"),
        "--out",
        str(out_dir),
        "--split",
        "supervised",
        "--part",
        "train",
        "--max-person-rows",
        str(args.fewnerd_person_rows),
        "--max-negative-rows",
        str(args.fewnerd_negative_rows),
        "--seed",
        str(args.seed),
    ]
    result = subprocess.run(fewnerd_argv, capture_output=False, text=True)
    if result.returncode != 0:
        print(
            f"ERROR: fewnerd_to_jsonl.py failed with code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    # 3. Count and summarize
    counts = _count_labels(out_dir)

    # Also count total rows across files.
    total_rows = 0
    for fp in sorted(out_dir.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            total_rows += sum(1 for line in fh if line.strip())

    print()
    print(f"✓  {total_rows} total augmentation rows in {out_dir}/")
    breakdown = ", ".join(f"{t}={counts.get(t, 0)}" for t in CLOAK_TYPES)
    print(f"   Label spans: {breakdown}")
    print()
    print(
        f"   Next: python scripts/training/train_model.py --augment-dir {out_dir}/ --langs 1en"
    )


if __name__ == "__main__":
    main()
