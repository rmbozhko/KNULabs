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
    dataset_info.json
    label2id.json

Usage:
    python build_dataset.py
    python build_dataset.py --input dataset/labeled --output dataset/hf_dataset
    python build_dataset.py --input dataset/labeled --val-split 0.2 --seed 42
    python build_dataset.py --input dataset/labeled --drop-other   # exclude OTHER tokens
"""

import argparse
import json
import random
import sys
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
    "INSTRUCTION",            # maps from OTHER / unlabeled
    "CONTENT",
    "OTHER"
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ──────────────────────────────────────────────────────────────────────────────
# LayoutLMv3 limits
# ──────────────────────────────────────────────────────────────────────────────

MAX_SEQ_LEN = 512   # LayoutLMv3 hard limit


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
# Load one JSON file → list of samples
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: Path, drop_other: bool = False) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))

    tokens    = data["tokens"]
    bboxes    = data["bboxes"]
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

    label_ids  = [LABEL2ID[l] for l in raw_labels]

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
        "bboxes":      Sequence(Sequence(Value("int32"), length=4)),
        "label_ids":   Sequence(
                           ClassLabel(num_classes=len(LABELS), names=LABELS)
                       ),
    })

    flat = {key: [s[key] for s in samples] for key in samples[0]}
    return Dataset.from_dict(flat, features=features)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build HuggingFace Dataset from labeled JSON files.")
    parser.add_argument("--input",      default="dataset/labeled",    help="Folder with *_labeled.json files")
    parser.add_argument("--output",     default="dataset/hf_dataset", help="Output folder")
    parser.add_argument("--val-split",  type=float, default=0.15,     help="Fraction for validation set (default 0.15)")
    parser.add_argument("--seed",       type=int,   default=42,       help="Random seed for train/val split")
    parser.add_argument("--drop-other", action="store_true",          help="Remove OTHER tokens from dataset")
    args = parser.parse_args()

    input_dir = Path(args.input)
    json_files = sorted(input_dir.glob("*_labeled.json"))

    if not json_files:
        print(f"Error: no *_labeled.json files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} labeled JSON file(s)\n")

    # ── Load all samples ──
    all_samples = []
    for jf in json_files:
        samples = load_json(jf, drop_other=args.drop_other)
        print(f"  {jf.name:45s}  →  {len(samples)} chunk(s)")
        all_samples.extend(samples)

    if not all_samples:
        print("\nError: no samples loaded.")
        sys.exit(1)

    # ── Train / val split (by page, not by chunk) ──
    random.seed(args.seed)
    random.shuffle(all_samples)

    split_idx   = max(1, int(len(all_samples) * (1 - args.val_split)))
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]

    print(f"\nSplit:  train={len(train_samples)}  val={len(val_samples)}")

    # ── Build datasets ──
    train_ds = build_hf_dataset(train_samples)
    val_ds   = build_hf_dataset(val_samples)

    dataset_dict = DatasetDict({"train": train_ds, "val": val_ds})

    # ── Save to disk ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(out_dir))

    # Save label mapping separately (handy for training script)
    (out_dir / "label2id.json").write_text(
        json.dumps(LABEL2ID, indent=2), encoding="utf-8"
    )
    (out_dir / "id2label.json").write_text(
        json.dumps(ID2LABEL, indent=2), encoding="utf-8"
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Label schema  : {LABELS}")
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Saved to      : {out_dir.resolve()}")
    print(f"\nLoad later with:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{out_dir}')")
    print(f"  ds['train'][0]  # first sample")


if __name__ == "__main__":
    main()