#!/usr/bin/env python3
"""
Train a fixed-label PII token classifier on the ai4privacy dataset.

Loads JSONL data from scripts/training/datasets/{train,validation}/,
maps fine-grained ai4privacy labels to Cloak Core-3 BIO tags (§6.2–§6.3),
fine-tunes a transformer encoder with a token-classification head, and
evaluates on a held-out test split via seqeval.

Usage:
  python scripts/train_model.py --backbone microsoft/deberta-v3-small --out ./trained/
  python scripts/train_model.py --backbone distilbert-base-multilingual-cased --out ./trained-distilbert/ --max-len 512
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Ensure scripts/ is on sys.path so we can import label_map.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from label_map import (  # noqa: E402
    AI4_TO_CLOAK,
    BIO_LABELS,
    CLOAK_TYPES,
    ID2LABEL,
    LABEL2ID,
)

logger = logging.getLogger(__name__)


def _load_jsonl_split(
    jsonl_dir: Path, max_rows: int | None = None, langs: list[str] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(jsonl_dir.glob("*.jsonl"))
    if langs is not None:
        files = [fp for fp in files if fp.stem in langs]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {jsonl_dir} (langs={langs})")
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def load_dataset(
    train_dir: Path,
    val_dir: Path,
    test_frac: float = 0.2,
    seed: int = 42,
    max_rows: int | None = None,
    augment_dir: Path | None = None,
    langs: list[str] | None = None,
    test_dir: Path | None = None,
) -> DatasetDict:
    """
    Load JSONL splits.  If *test_dir* is provided and contains .jsonl files
    it is used directly as the held-out test set; otherwise test is carved
    from the validation split (stratified by Core-3 label presence).

    Returns a DatasetDict with 'train', 'validation', 'test'.
    """
    train_rows = _load_jsonl_split(train_dir, max_rows=max_rows, langs=langs)
    val_rows = _load_jsonl_split(val_dir, max_rows=max_rows, langs=langs)

    # Pre-split test takes priority over carving from validation.
    pre_split_test: list[dict] | None = None
    if test_dir and test_dir.exists():
        try:
            pre_split_test = _load_jsonl_split(test_dir, langs=langs)
        except FileNotFoundError:
            pass

    if pre_split_test:
        test_rows = pre_split_test
        kept_val = val_rows
        logger.info("Loaded pre-split test: %d rows", len(test_rows))
    else:
        # Carve a held-out test from validation (stratify by presence of
        # Core-3 gold labels so each type is represented).
        rng = np.random.default_rng(seed)
        test_rows = []
        kept_val = []

        # Stratify: for each example compute the set of Core-3 types it contains.
        def _core3_set(mask: list[dict]) -> frozenset[str]:
            types: set[str] = set()
            for m in mask:
                t = AI4_TO_CLOAK.get(m["label"])
                if t:
                    types.add(t)
            return frozenset(types)

        # Group validation rows by their Core-3 type set.
        from collections import defaultdict

        buckets: dict[frozenset[str], list[dict]] = defaultdict(list)
        for row in val_rows:
            buckets[_core3_set(row.get("privacy_mask", []))].append(row)

        for _type_set, group in buckets.items():
            n_test = max(1, int(len(group) * test_frac))
            rng.shuffle(group)
            test_rows.extend(group[:n_test])
            kept_val.extend(group[n_test:])

    if augment_dir and augment_dir.exists():
        aug_rows = _load_jsonl_split(augment_dir)
        train_rows.extend(aug_rows)
        logger.info("Added %d augmentation rows to train", len(aug_rows))

    logger.info(
        "Train: %d  Val: %d  Test: %d", len(train_rows), len(kept_val), len(test_rows)
    )
    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(kept_val),
            "test": Dataset.from_list(test_rows),
        }
    )


def preprocess_dataset(
    dataset: DatasetDict, tokenizer: Any, max_len: int
) -> DatasetDict:
    """
    Tokenize raw rows and produce BIO-tagged label sequences.

    The tagging convention labels *every* subword of an entity:
      B-<TYPE> on the first subword, I-<TYPE> on subsequent subwords.
    Non-entity subwords = O.  Special tokens (CLS/SEP/pad) = -100.

    This lets the inference decoder merge by contiguous non-O tokens using
    byte offsets alone — no word-boundary reconstruction needed in Rust.
    """

    def _encode(examples: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(
            examples["source_text"],
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            # No padding here — the data collator handles dynamic padding.
        )

        all_labels: list[list[int]] = []
        for i in range(len(examples["source_text"])):
            mask = examples["privacy_mask"][i]

            # Collect gold spans, mapped to Core-3, sorted by start.
            gold_spans: list[tuple[int, int, str]] = []
            for m in mask:
                t = AI4_TO_CLOAK.get(m["label"])
                if t is not None:
                    gold_spans.append((m["start"], m["end"], t))
            gold_spans.sort()

            offsets = enc["offset_mapping"][i]
            labels: list[int] = []
            for tok_start, tok_end in offsets:
                if tok_start == tok_end:  # special token / empty
                    labels.append(-100)
                    continue
                tag = "O"
                for gs, ge, gtype in gold_spans:
                    # Token fully inside a gold span.
                    if tok_start >= gs and tok_end <= ge:
                        tag = f"B-{gtype}" if tok_start == gs else f"I-{gtype}"
                        break
                labels.append(LABEL2ID[tag])
            all_labels.append(labels)

        enc["labels"] = all_labels
        # Drop offset_mapping before collation — not needed after this step.
        enc.pop("offset_mapping", None)
        return enc

    encoded = dataset.map(
        _encode, batched=True, remove_columns=dataset["train"].column_names
    )
    return encoded  # type: ignore[return-value]


def _build_compute_metrics():
    """Lazy-load seqeval to avoid import cost when --help is used."""
    import evaluate

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []
        for p_row, l_row in zip(preds, labels):
            t = [ID2LABEL[int(l)] for p, l in zip(p_row, l_row) if l != -100]
            q = [ID2LABEL[int(p)] for p, l in zip(p_row, l_row) if l != -100]
            true_labels.append(t)
            pred_labels.append(q)

        results = seqeval.compute(
            predictions=pred_labels, references=true_labels, zero_division=0
        )

        # Flatten per-type metrics into the top-level dict.
        out: dict[str, float] = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        for t in CLOAK_TYPES:
            t_lower = t.lower()
            per_type = results.get(t, {})  # type: ignore[arg-type]
            out[f"{t_lower}_precision"] = per_type.get("precision", 0.0)
            out[f"{t_lower}_recall"] = per_type.get("recall", 0.0)
            out[f"{t_lower}_f1"] = per_type.get("f1", 0.0)
        return out

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a fixed-label PII token classifier"
    )
    parser.add_argument(
        "--backbone",
        default="distilbert-base-multilingual-cased",
        help="HuggingFace model ID (§4.1: distilbert-base-multilingual-cased, microsoft/deberta-v3-small, etc.)",
    )
    parser.add_argument(
        "--out", default="./trained", help="Output directory for the HF checkpoint"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=384,
        help="Max token length (must be ≤ model max)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per-device training batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows loaded across all splits (for fast smoke tests)",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Cap rows in training split only (subsample for faster iteration)",
    )
    parser.add_argument(
        "--augment-dir",
        type=str,
        default=None,
        help="Optional directory of augmented JSONL rows",
    )
    parser.add_argument(
        "--data-dir", default=None, help="Override the default data/ directory"
    )
    parser.add_argument(
        "--langs",
        default="1en",
        help=(
            "Comma-separated dataset language stems to load (matches "
            "data/{train,validation,test}/<lang>.jsonl), or 'all' for every "
            "language on disk. Cloak only redacts English text, so training "
            "on the other 5 shipped languages (de/es/fr/it/nl) is ~5x more "
            "rows for no product benefit — default is English-only."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Dataloader worker processes (parallel batch collation)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps (simulates larger batch)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (trades ~20%% speed for ~40%% less VRAM)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent.parent
    datasets_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    train_dir = datasets_dir / "train"
    val_dir = datasets_dir / "validation"
    test_dir = datasets_dir / "test"

    augment_dir = Path(args.augment_dir) if args.augment_dir else None
    langs = (
        None
        if args.langs.strip().lower() == "all"
        else [s.strip() for s in args.langs.split(",") if s.strip()]
    )

    raw_ds = load_dataset(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        seed=args.seed,
        max_rows=args.max_rows,
        augment_dir=augment_dir,
        langs=langs,
    )

    # Subsample train for faster iteration without touching val/test.
    if args.max_train_rows and len(raw_ds["train"]) > args.max_train_rows:
        import random
        rng_train = random.Random(args.seed)
        indices = rng_train.sample(range(len(raw_ds["train"])), args.max_train_rows)
        raw_ds["train"] = raw_ds["train"].select(indices)
        logger.info("Subsampled train to %d rows", args.max_train_rows)

    # Try fast first; fall back to slow if it fails.
    logger.info("Loading tokenizer: %s", args.backbone)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        tokenizer_is_fast = getattr(tokenizer, "is_fast", False)
        logger.info("  fast tokenizer loaded ✓")
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(
            "  fast tokenizer failed (%s) — falling back to slow tokenizer", e
        )
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=False)
        tokenizer_is_fast = False
        logger.info("  slow tokenizer loaded")

    logger.info("Loading model: %s  num_labels=%d", args.backbone, len(BIO_LABELS))
    model = AutoModelForTokenClassification.from_pretrained(
        args.backbone,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("  gradient checkpointing enabled (less VRAM, slightly slower)")
    # Print class distribution for sanity check.
    all_labels: Counter[str] = Counter()
    for row in raw_ds["train"]:
        for m in row.get("privacy_mask", []):
            ct = AI4_TO_CLOAK.get(m["label"])
            if ct:
                all_labels[ct] += 1
    total = sum(all_labels.values())
    logger.info(
        "Training label distribution: %s",
        {k: f"{v} ({v/total*100:.1f}%)" for k, v in all_labels.most_common()},
    )

    # Preprocess
    logger.info("Tokenizing & BIO-aligning (max_len=%d) ...", args.max_len)
    encoded_ds = preprocess_dataset(raw_ds, tokenizer, args.max_len)

    # Training
    training_args = TrainingArguments(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        save_total_limit=2,
        dataloader_drop_last=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        data_collator=data_collator,
        compute_metrics=_build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training ...")
    trainer.train()

    # Final eval on held-out test
    logger.info("Evaluating on held-out test set ...")
    test_metrics = trainer.evaluate(encoded_ds["test"], metric_key_prefix="test")
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # Save checkpoint
    logger.info("Saving checkpoint to %s ...", args.out)
    trainer.save_model(args.out)

    # Ensure a fast tokenizer (tokenizer.json) is saved for the Rust sidecar.
    # If we fell back to a slow tokenizer during training, reload as fast
    # now to produce the tokenizer.json the sidecar needs.
    if not tokenizer_is_fast:
        logger.info("  reloading tokenizer as fast for export ...")
        try:
            fast_tok = AutoTokenizer.from_pretrained(args.backbone)
            fast_tok.save_pretrained(args.out)
            logger.info("  fast tokenizer saved ✓")
        except Exception as e:
            logger.warning(
                "  could not save fast tokenizer (%s) — slow tokenizer will be used", e
            )
            tokenizer.save_pretrained(args.out)
    else:
        tokenizer.save_pretrained(args.out)

    # Write training manifest.
    manifest = {
        "backbone": args.backbone,
        "seed": args.seed,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "bio_labels": BIO_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "train_rows": len(raw_ds["train"]),
        "val_rows": len(raw_ds["validation"]),
        "test_rows": len(raw_ds["test"]),
        "test_metrics": test_metrics,
    }
    manifest_path = Path(args.out) / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("✓  Saved: %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    main()
