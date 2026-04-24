#!/usr/bin/env python3
"""
build_dataset.py
────────────────
Reads labeled JSON files (output of ls_to_token_labels.py),
builds a HuggingFace Dataset ready for LayoutLMv3 fine-tuning,
and saves it to disk in Arrow format.

Expected input JSON schema (one file per page):
  {
    "image_path": "dataset/.../page_004.png",
    "image_size": {"width": 2481, "height": 3509},
    "tokens":  ["word", ...],
    "bboxes":  [[x1,y1,x2,y2], ...],   # normalized [0,1000]
    "labels":  ["INSTRUCTION", "CONTENT", "OTHER", ...]
  }

Output:
  dataset/hf_dataset/
    train/
    val/
    test/
    dataset_info.json
    label2id.json
    class_weights.json

Usage:
    python build_dataset.py
    python build_dataset.py --input dataset/labeled --output dataset/hf_dataset
    python build_dataset.py --input dataset/labeled --val-split 0.15 --test-split 0.10 --seed 42
    python build_dataset.py --input dataset/labeled --drop-other   # exclude OTHER tokens
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel
    from PIL import Image
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install datasets pillow")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Label schema
# ──────────────────────────────────────────────────────────────────────────────

LABELS = [
    "INSTRUCTION",
    "CONTENT",
    "OTHER",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ──────────────────────────────────────────────────────────────────────────────
# LayoutLMv3 limits
# ──────────────────────────────────────────────────────────────────────────────

MAX_SEQ_LEN = 512   # LayoutLMv3 hard limit

BBOX_MIN = 0
BBOX_MAX = 1000


def chunk_page(tokens, bboxes, label_ids, max_len=MAX_SEQ_LEN):
    """
    Split a page into non-overlapping chunks of max_len tokens.
    Most pages fit in one chunk; long pages get split.
    """
    chunks = []
    for start in range(0, len(tokens), max_len):
        end = start + max_len
        chunks.append({
            "tokens":    tokens[start:end],
            "bboxes":    bboxes[start:end],
            "label_ids": label_ids[start:end],
        })
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def validate_sample(sample: dict, source_file: str) -> list[str]:
    """
    Run integrity checks on a single sample. Returns a list of error strings
    (empty list means the sample is clean).

    Checks:
      1. Every token has exactly one label.
      2. Every token has exactly one bbox of length 4.
      3. All bbox coordinates are integers in [0, 1000].
      4. No empty token strings.
    """
    errors = []
    sid = sample["id"]
    tokens    = sample["tokens"]
    bboxes    = sample["bboxes"]
    label_ids = sample["label_ids"]

    n = len(tokens)

    # ── 1. Parallel length check ──────────────────────────────────────────────
    if len(label_ids) != n:
        errors.append(
            f"[{sid}] token/label length mismatch: "
            f"{n} tokens vs {len(label_ids)} labels  (file: {source_file})"
        )
    if len(bboxes) != n:
        errors.append(
            f"[{sid}] token/bbox length mismatch: "
            f"{n} tokens vs {len(bboxes)} bboxes  (file: {source_file})"
        )

    # ── 2. bbox normalization check ───────────────────────────────────────────
    bad_bboxes = []
    for ti, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            bad_bboxes.append((ti, bbox, "wrong length"))
            continue
        for coord in bbox:
            if not isinstance(coord, (int, float)):
                bad_bboxes.append((ti, bbox, f"non-numeric coord {coord!r}"))
                break
            if not (BBOX_MIN <= coord <= BBOX_MAX):
                bad_bboxes.append((ti, bbox, f"coord {coord} out of [0,1000]"))
                break

    if bad_bboxes:
        # Report first 5 offenders to keep noise low
        for ti, bbox, reason in bad_bboxes[:5]:
            errors.append(
                f"[{sid}] bbox[{ti}]={bbox} invalid ({reason})  (file: {source_file})"
            )
        if len(bad_bboxes) > 5:
            errors.append(
                f"[{sid}] … and {len(bad_bboxes) - 5} more bbox error(s)"
            )

    # ── 3. Empty token check ──────────────────────────────────────────────────
    empty_tok = [i for i, t in enumerate(tokens) if not isinstance(t, str) or t.strip() == ""]
    if empty_tok:
        errors.append(
            f"[{sid}] {len(empty_tok)} empty/non-string token(s) at indices "
            f"{empty_tok[:5]}{'...' if len(empty_tok) > 5 else ''}  (file: {source_file})"
        )

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Class-weight computation  (inverse-frequency, normalized to mean=1)
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(samples: list[dict]) -> dict[str, float]:
    """
    Compute inverse-frequency class weights over all tokens in *samples*.

    Strategy: w_i = total_tokens / (n_classes * count_i)
    Weights are then normalized so their mean equals 1.0, which keeps
    the effective learning rate stable regardless of class imbalance.

    Returns a dict  {label_name: weight, ...}  and also includes
    the raw counts for transparency.
    """
    counts: Counter = Counter()
    for s in samples:
        for lid in s["label_ids"]:
            counts[lid] += 1

    total   = sum(counts.values())
    n_cls   = len(LABELS)
    weights_raw = {}

    for lid, label in ID2LABEL.items():
        cnt = counts.get(lid, 0)
        if cnt == 0:
            # Assign a high weight so unseen class is at least represented
            weights_raw[label] = float(n_cls)
        else:
            weights_raw[label] = total / (n_cls * cnt)

    # Normalize: divide by mean so weights average to 1.0
    mean_w = sum(weights_raw.values()) / len(weights_raw)
    weights = {label: w / mean_w for label, w in weights_raw.items()}

    return weights, dict(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Load one JSON file → list of samples
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path, drop_other: bool = False) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))

    tokens     = data["tokens"]
    bboxes     = data["bboxes"]
    raw_labels = data["labels"]
    image_path = data.get("image_path", "")

    # Optionally remove OTHER tokens entirely
    if drop_other:
        filtered = [
            (t, b, l)
            for t, b, l in zip(tokens, bboxes, raw_labels)
            if l != "OTHER"
        ]
        if not filtered:
            return []
        tokens, bboxes, raw_labels = zip(*filtered)
        tokens, bboxes, raw_labels = list(tokens), list(bboxes), list(raw_labels)

    label_ids = [LABEL2ID[l] for l in raw_labels]

    chunks = chunk_page(tokens, bboxes, label_ids)

    samples = []
    for i, chunk in enumerate(chunks):
        samples.append({
            "id":          f"{path.stem}_chunk{i}",
            "image_path":  image_path,
            "tokens":      chunk["tokens"],
            "bboxes":      chunk["bboxes"],
            "label_ids":   chunk["label_ids"],
        })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Build HuggingFace Dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_dataset(samples: list[dict]) -> Dataset:
    """Convert list of sample dicts to a HuggingFace Dataset."""
    features = Features({
        "id":          Value("string"),
        "image_path":  Value("string"),
        "tokens":      Sequence(Value("string")),
        "bboxes":      Sequence(Sequence(Value("int32"))),
        "label_ids":   Sequence(Value("int32")),
    })
    print(samples[1]["id"])
    print(samples[0]["image_path"])
    print(samples[0]["tokens"][:10])
    print("bboxes:", samples[0]["bboxes"][:10])
    print("label_ids:", samples[0]["label_ids"][:10])
    flat = {key: [s[key] for s in samples] for key in samples[0]}
    return Dataset.from_dict(flat, features=features)


# ──────────────────────────────────────────────────────────────────────────────
# Split helper  (train / val / test)
# ──────────────────────────────────────────────────────────────────────────────

def three_way_split(
    samples: list[dict],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Shuffle and split *samples* into train / val / test.

    Both val_frac and test_frac are fractions of the *total* dataset.
    train_frac is implicitly  1 - val_frac - test_frac.

    Raises ValueError if fractions are invalid.
    """
    if val_frac < 0 or test_frac < 0:
        raise ValueError("Split fractions must be non-negative.")
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_split ({val_frac}) + test_split ({test_frac}) must be < 1.0"
        )

    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_val  = max(1, math.floor(n * val_frac))  if val_frac  > 0 else 0
    n_test = max(1, math.floor(n * test_frac)) if test_frac > 0 else 0
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough samples ({n}) for the requested splits "
            f"(val={n_val}, test={n_test})."
        )

    train = samples[:n_train]
    val   = samples[n_train : n_train + n_val]
    test  = samples[n_train + n_val :]

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace Dataset from labeled JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",       default="dataset/labeled",    help="Folder with *_labeled.json files")
    parser.add_argument("--output",      default="dataset/hf_dataset", help="Output folder")
    parser.add_argument("--val-split",   type=float, default=0.15,     help="Fraction of total data for validation set")
    parser.add_argument("--test-split",  type=float, default=0.10,     help="Fraction of total data for test set (0 = no test split)")
    parser.add_argument("--seed",        type=int,   default=42,       help="Random seed for reproducible splits")
    parser.add_argument("--drop-other",  action="store_true",          help="Remove OTHER tokens from dataset")
    parser.add_argument("--strict",      action="store_true",          help="Abort on any validation error instead of just warning")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    json_files = sorted(input_dir.glob("*_labeled.json"))

    if not json_files:
        print(f"Error: no *_labeled.json files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} labeled JSON file(s)\n")

    # ── Load all samples ──────────────────────────────────────────────────────
    all_samples   = []
    all_errors    = []
    file_map      = {}   # sample id → source filename (for error messages)

    for jf in json_files:
        samples = load_json(jf, drop_other=args.drop_other)
        print(f"  {jf.name:45s}  →  {len(samples)} chunk(s)")
        for s in samples:
            file_map[s["id"]] = jf.name
        all_samples.extend(samples)

    if not all_samples:
        print("\nError: no samples loaded.")
        sys.exit(1)

    # ── Validation pass ───────────────────────────────────────────────────────
    print(f"\nValidating {len(all_samples)} sample(s)…")
    for s in all_samples:
        errs = validate_sample(s, file_map[s["id"]])
        all_errors.extend(errs)

    if all_errors:
        print(f"\n{'!'*60}")
        print(f"  {len(all_errors)} VALIDATION ERROR(S) FOUND:")
        print(f"{'!'*60}")
        for e in all_errors:
            print(f"  ✗  {e}")
        print()
        if args.strict:
            print("Aborting due to --strict flag.")
            sys.exit(1)
        else:
            print("Continuing anyway (pass --strict to abort on errors).\n")
    else:
        print("  ✓  All tokens labeled, bboxes normalized. No issues found.\n")

    # ── Train / val / test split ──────────────────────────────────────────────
    try:
        train_samples, val_samples, test_samples = three_way_split(
            all_samples,
            val_frac=args.val_split,
            test_frac=args.test_split,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    split_msg = f"train={len(train_samples)}  val={len(val_samples)}"
    if test_samples:
        split_msg += f"  test={len(test_samples)}"
    print(f"Split:  {split_msg}")

    # ── Class weights (computed on TRAIN split only to avoid leakage) ─────────
    weights, counts = compute_class_weights(train_samples)

    print(f"\nClass distribution (train tokens):")
    total_train_tokens = sum(counts.values())
    for label in LABELS:
        lid   = LABEL2ID[label]
        cnt   = counts.get(lid, 0)
        pct   = 100.0 * cnt / total_train_tokens if total_train_tokens else 0
        w     = weights[label]
        print(f"  {label:15s}  count={cnt:>8,}  ({pct:5.1f}%)  weight={w:.4f}")

    # ── Build HuggingFace datasets ────────────────────────────────────────────
    splits: dict = {"train": build_hf_dataset(train_samples)}
    if val_samples:
        splits["val"] = build_hf_dataset(val_samples)
    if test_samples:
        splits["test"] = build_hf_dataset(test_samples)

    dataset_dict = DatasetDict(splits)

    # ── Save to disk ──────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(out_dir))

    # Label mappings
    (out_dir / "label2id.json").write_text(
        json.dumps(LABEL2ID, indent=2), encoding="utf-8"
    )
    (out_dir / "id2label.json").write_text(
        json.dumps(ID2LABEL, indent=2), encoding="utf-8"
    )

    # Class weights  — ready to pass directly to nn.CrossEntropyLoss(weight=…)
    weights_payload = {
        "class_weights": weights,          # {label: float}  mean-normalized
        "class_weights_by_id": {           # {str(id): float} for torch.tensor()
            str(LABEL2ID[lbl]): w
            for lbl, w in weights.items()
        },
        "token_counts_train": {
            lbl: counts.get(LABEL2ID[lbl], 0) for lbl in LABELS
        },
        "note": (
            "Weights are inverse-frequency, normalized so mean(weights)=1. "
            "Usage:  import torch; w = torch.tensor([weights_by_id[str(i)] "
            "for i in range(num_labels)]);  loss_fn = nn.CrossEntropyLoss(weight=w)"
        ),
    }
    (out_dir / "class_weights.json").write_text(
        json.dumps(weights_payload, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Label schema  : {LABELS}")
    print(f"Train samples : {len(splits['train'])}")
    if "val" in splits:
        print(f"Val   samples : {len(splits['val'])}")
    if "test" in splits:
        print(f"Test  samples : {len(splits['test'])}")
    print(f"Saved to      : {out_dir.resolve()}")
    print(f"\nClass weights saved to: {out_dir / 'class_weights.json'}")
    print(f"\nLoad later with:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{out_dir}')")
    print(f"  ds['train'][0]  # first sample")
    print(f"\nUsing weights in PyTorch:")
    print(f"  import json, torch, torch.nn as nn")
    print(f"  cw = json.load(open('{out_dir}/class_weights.json'))")
    print(f"  n  = {len(LABELS)}  # num_labels")
    print(f"  w  = torch.tensor([cw['class_weights_by_id'][str(i)] for i in range(n)])")
    print(f"  loss_fn = nn.CrossEntropyLoss(weight=w)")


if __name__ == "__main__":
    main()