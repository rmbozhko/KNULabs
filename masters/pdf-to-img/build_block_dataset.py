#!/usr/bin/env python3
"""
build_block_dataset.py
──────────────────────
Converts labeled JSON files (ls_to_token_labels.py output) into a
block-level HuggingFace Dataset for LayoutLMv3 sequence classification.

Token-level approach:  every word → label  (needs BIO, 200+ pages)
Block-level approach:  every annotated region → label  (works with 30+ pages) ✅

Each sample = one annotated block from Label Studio:
  - text:  all tokens inside the block, joined as a string
  - bbox:  block bbox normalized to [0, 1000]
  - image: cropped region of the page image  (optional, --no-images to skip)
  - label: INSTRUCTION | CONTENT

Input:  dataset/labeled/*_labeled.json   (contains _blocks with label+bbox)
Output: dataset/hf_block_dataset/
          train/ val/
          label2id.json  id2label.json

Usage:
    python build_block_dataset.py
    python build_block_dataset.py --input dataset/labeled --output dataset/hf_block_dataset
    python build_block_dataset.py --no-images   # skip image crops (faster, text+layout only)
    python build_block_dataset.py --val-split 0.2 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Image as HFImage, Sequence
    from PIL import Image
except ImportError:
    print("Missing dependencies:  pip install datasets pillow")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Label schema  (block-level = flat, no BIO needed)
# ──────────────────────────────────────────────────────────────────────────────

LABELS    = ["INSTRUCTION", "CONTENT"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def tokens_in_block(
    tokens: list[str],
    token_bboxes: list[list[int]],
    block_bbox: list[int],
    threshold: float = 0.5,
) -> str:
    """
    Collect all tokens whose bbox overlaps block_bbox by >= threshold,
    and join them into a single string (the text of the block).
    """
    def overlap(tb, bb):
        ix1, iy1 = max(tb[0], bb[0]), max(tb[1], bb[1])
        ix2, iy2 = min(tb[2], bb[2]), min(tb[3], bb[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        t_area = max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        return inter / t_area

    words = [t for t, b in zip(tokens, token_bboxes) if overlap(b, block_bbox) >= threshold]
    return " ".join(words)


def norm_bbox_to_pixel(bbox: list[int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert [0,1000] bbox back to pixel coordinates for cropping."""
    x1 = int(bbox[0] / 1000 * img_w)
    y1 = int(bbox[1] / 1000 * img_h)
    x2 = int(bbox[2] / 1000 * img_w)
    y2 = int(bbox[3] / 1000 * img_h)
    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────────────────────────
# Load one labeled JSON → list of block samples
# ──────────────────────────────────────────────────────────────────────────────

def load_blocks_from_json(
    path: Path,
    use_images: bool = True,
    image_size: tuple[int, int] = (224, 224),
) -> list[dict]:
    data   = json.loads(path.read_text(encoding="utf-8"))
    blocks = data.get("_blocks", [])

    if not blocks:
        return []

    tokens       = data.get("tokens", [])
    token_bboxes = data.get("bboxes", [])       # normalized [0,1000]
    img_path_str = data.get("image_path", "")
    img_size_raw = data.get("image_size", {})
    img_w        = img_size_raw.get("width",  2481)
    img_h        = img_size_raw.get("height", 3509)

    # Try to load the page image (needed for crops)
    page_image = None
    if use_images:
        for candidate in [
            Path(img_path_str),
            Path(img_path_str.replace("\\", "/")),
        ]:
            if candidate.exists():
                try:
                    page_image = Image.open(candidate).convert("RGB")
                except Exception:
                    pass
                break
        if page_image is None:
            print(f"  ⚠  Image not found: {img_path_str} — skipping image crops")

    samples = []
    for block in blocks:
        label_str = block.get("label", "")
        if label_str not in LABEL2ID:
            continue                         # skip OTHER / unknown labels

        bbox = block["bbox"]                 # already [0,1000]

        # ── text of this block ──
        text = tokens_in_block(tokens, token_bboxes, bbox)
        if not text.strip():
            continue                         # empty block — skip

        sample = {
            "id":         f"{path.stem}__{len(samples)}",
            "image_path": img_path_str,
            "text":       text,
            "bbox":       bbox,              # [x1,y1,x2,y2] in [0,1000]
            "label":      LABEL2ID[label_str],
        }

        # ── optional image crop ──
        if use_images and page_image is not None:
            x1, y1, x2, y2 = norm_bbox_to_pixel(bbox, img_w, img_h)
            # Add small padding
            pad = 4
            crop = page_image.crop((
                max(0, x1 - pad), max(0, y1 - pad),
                min(img_w, x2 + pad), min(img_h, y2 + pad),
            ))
            crop = crop.resize(image_size, Image.LANCZOS)
            sample["image"] = crop           # PIL Image — HF will encode it
        else:
            sample["image"] = None

        samples.append(sample)

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Build HuggingFace DatasetDict
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_dataset(samples: list[dict], use_images: bool) -> Dataset:
    flat = {key: [s[key] for s in samples] for key in samples[0]}

    features = Features({
        "id":         Value("string"),
        "image_path": Value("string"),
        "text":       Value("string"),
        "bbox":       Sequence(Value("int32"), length=4),
        "label":      ClassLabel(num_classes=len(LABELS), names=LABELS),
    })
    if use_images:
        features["image"] = HFImage()

    return Dataset.from_dict(flat, features=features)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build block-level HF Dataset for LayoutLMv3.")
    parser.add_argument("--input",      default="dataset/labeled",          help="Folder with *_labeled.json")
    parser.add_argument("--output",     default="dataset/hf_block_dataset", help="Output folder")
    parser.add_argument("--val-split",  type=float, default=0.15,           help="Val fraction (default 0.15)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-images",  action="store_true",                help="Skip image crops")
    parser.add_argument("--image-size", type=int,   default=224,            help="Crop resize (default 224)")
    args = parser.parse_args()

    use_images = not args.no_images
    json_files = sorted(Path(args.input).glob("*_labeled.json"))

    if not json_files:
        print(f"No *_labeled.json files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(json_files)} labeled JSON files  |  images={'yes' if use_images else 'no'}\n")

    all_samples = []
    for jf in json_files:
        samples = load_blocks_from_json(
            jf,
            use_images=use_images,
            image_size=(args.image_size, args.image_size),
        )
        label_counts = {}
        for s in samples:
            l = LABELS[s["label"]]
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"  {jf.name:45s}  blocks={len(samples):3d}  {label_counts}")
        all_samples.extend(samples)

    if not all_samples:
        print("\nNo blocks loaded. Check that _blocks field exists in your JSON files.")
        sys.exit(1)

    # ── Split ──
    random.seed(args.seed)
    random.shuffle(all_samples)
    split_idx     = max(1, int(len(all_samples) * (1 - args.val_split)))
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]

    print(f"\nSplit:  train={len(train_samples)}  val={len(val_samples)}")

    train_ds = build_hf_dataset(train_samples, use_images)
    val_ds   = build_hf_dataset(val_samples,   use_images)
    ds_dict  = DatasetDict({"train": train_ds, "val": val_ds})

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(out_dir))

    (out_dir / "label2id.json").write_text(json.dumps(LABEL2ID, indent=2))
    (out_dir / "id2label.json").write_text(json.dumps(ID2LABEL, indent=2))

    # Label distribution
    all_labels = [LABELS[s["label"]] for s in all_samples]
    dist = {l: all_labels.count(l) for l in LABELS}

    print(f"\n{'='*60}")
    print(f"Total blocks   : {len(all_samples)}")
    print(f"Label dist     : {dist}")
    print(f"Train / Val    : {len(train_ds)} / {len(val_ds)}")
    print(f"Saved to       : {out_dir.resolve()}")
    print(f"\nLoad later:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{out_dir}')")
    print(f"  print(ds['train'][0])")


if __name__ == "__main__":
    main()