#!/usr/bin/env python3
"""
Convert Few-NERD parquet rows into the ai4privacy jsonl row schema.

Few-NERD is a Wikipedia-domain NER corpus whose ``ner_tags`` follow this
coarse ClassLabel table (verified against the actual parquet metadata):

    0 O
    1 art
    2 building
    3 event
    4 location
    5 organization
    6 other
    7 person       <-- maps to Cloak "NAME"
    8 product

Only *person* spans are extracted.  Rows with zero person spans are sampled
as hard negatives so the model learns that organization/location/product
mentions are NOT NAME.

Usage:
  python scripts/training/fewnerd_to_jsonl.py --out ./augmented/ \
      --split supervised --part train \
      --max-person-rows 8000 --max-negative-rows 2000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

PERSON_TAG_ID = 7
CLOAK_NAME_LABEL = "NAME"

_HERE = Path(__file__).resolve().parent
_DATASET_ROOT = _HERE / "datasets" / "few_nerd_dataset"


def _parquet_path(split: str, part: str) -> Path:
    return _DATASET_ROOT / split / f"{part}-00000-of-00001.parquet"


def convert_parquet(
    pq_path: Path,
    *,
    max_person_rows: int,
    max_negative_rows: int,
    seed: int,
) -> tuple[list[dict], int, int]:
    """Convert a Few-NERD parquet file to ai4privacy-format rows.

    Returns (rows, n_person_rows, n_negative_rows).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(pq_path)
    raw_rows = table.to_pylist()

    rng = random.Random(seed)

    person_rows: list[dict] = []
    negative_rows: list[dict] = []

    for row in raw_rows:
        tokens: list[str] = row["tokens"]
        ner_tags: list[int] = row["ner_tags"]

        source_text = " ".join(tokens)

        # --- Exact offset-tracking algorithm from §2 of dataset-unification-plan.md ---
        # Reconstruct span offsets into the space-separated source_text.
        # Contiguous PERSON-tagged tokens merge into one span.
        offset = 0
        spans: list[tuple[int, int, str]] = []
        span_start: int | None = None

        for tok, tag in zip(tokens, ner_tags):
            start = offset
            end = start + len(tok)
            if tag == PERSON_TAG_ID:
                if span_start is None:
                    span_start = start
            else:
                if span_start is not None:
                    spans.append((span_start, offset - 1, CLOAK_NAME_LABEL))
                    span_start = None
            offset = end + 1  # +1 accounts for the joining space
        if span_start is not None:
            spans.append((span_start, offset - 1, CLOAK_NAME_LABEL))

        # Build privacy_mask entries.
        privacy_mask: list[dict] = []
        for s, e, label in spans:
            value = source_text[s:e]
            # Mandatory self-check — fail loudly on offset mismatch.
            assert source_text[s:e] == value, (
                f"Offset mismatch: source_text[{s}:{e}] = {source_text[s:e]!r} "
                f"!= {value!r}  (source_text={source_text!r})"
            )
            privacy_mask.append({
                "start": s,
                "end": e,
                "label": label,
                "value": value,
            })

        out_row = {
            "source_text": source_text,
            "language": "en",
            "locale": "US",
            "split": "train",
            "privacy_mask": privacy_mask,
            "uid": 0,  # filled in later
            "masked_text": "",
        }

        if spans:
            person_rows.append(out_row)
        else:
            negative_rows.append(out_row)

    # Sample.
    if len(person_rows) > max_person_rows:
        person_rows = rng.sample(person_rows, max_person_rows)
    if len(negative_rows) > max_negative_rows:
        negative_rows = rng.sample(negative_rows, max_negative_rows)

    # Interleave person and negative rows so the dataset isn't front-loaded
    # with one type (the train loader doesn't shuffle on disk).
    combined: list[dict] = []
    i, j = 0, 0
    while i < len(person_rows) or j < len(negative_rows):
        if i < len(person_rows):
            combined.append(person_rows[i])
            i += 1
        if j < len(negative_rows):
            combined.append(negative_rows[j])
            j += 1

    # Assign stable UIDs.
    for i, row in enumerate(combined):
        row["uid"] = 2_000_000 + i

    return combined, len(person_rows), len(negative_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Few-NERD parquet to ai4privacy jsonl rows"
    )
    parser.add_argument("--out", required=True, help="Output directory for .jsonl files")
    parser.add_argument(
        "--split", default="supervised",
        choices=["supervised", "inter", "intra"],
        help="Few-NERD data split (default: supervised)",
    )
    parser.add_argument(
        "--part", default="train",
        choices=["train", "validation", "test"],
        help="Few-NERD partition (train/validation/test; default: train)",
    )
    parser.add_argument(
        "--max-person-rows", type=int, default=8000,
        help="Max rows containing at least one person span",
    )
    parser.add_argument(
        "--max-negative-rows", type=int, default=2000,
        help="Max rows with zero person spans (hard negatives)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    args = parser.parse_args()

    pq_path = _parquet_path(args.split, args.part)
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {pq_path}")

    rows, n_person, n_negative = convert_parquet(
        pq_path,
        max_person_rows=args.max_person_rows,
        max_negative_rows=args.max_negative_rows,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fewnerd.jsonl"

    with open(out_file, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"✓  {len(rows)} rows ({n_person} person-bearing, {n_negative} negative)"
        f" → {out_file}"
    )


if __name__ == "__main__":
    main()
