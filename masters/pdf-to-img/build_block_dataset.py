#!/usr/bin/env python3
"""
build_block_dataset.py  (v3 — automatic 3-way page-level split)
───────────────────────────────────────────────────────────────
Splits pages into train / val / test automatically based on dataset size.

Split strategy (by number of pages):
  ≤ 15 pages  →  70 / 15 / 15 %
  16–30 pages →  75 / 12.5 / 12.5 %   (your case: 26 pages)
  31–60 pages →  80 / 10 / 10 %
  > 60 pages  →  85 / 7.5 / 7.5 %

Guarantees:
  - val  ≥ 2 pages
  - test ≥ 2 pages
  - No page appears in more than one split

Usage:
    python build_block_dataset.py
    python build_block_dataset.py --input dataset/labeled --output dataset/hf_block_dataset
    python build_block_dataset.py --no-images
    python build_block_dataset.py --train 0.8 --val 0.1 --test 0.1  # manual override
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

try:
    from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Image as HFImage, Sequence
    from PIL import Image
except ImportError:
    print("pip install datasets pillow")
    sys.exit(1)


LABELS   = ["INSTRUCTION", "CONTENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Auto split ratio based on dataset size
# ──────────────────────────────────────────────────────────────────────────────

def auto_split_ratio(n_pages: int) -> tuple[float, float, float]:
    """Return (train, val, test) fractions based on number of pages."""
    if n_pages <= 15:
        return 0.70, 0.15, 0.15
    elif n_pages <= 30:
        return 0.75, 0.125, 0.125
    elif n_pages <= 60:
        return 0.80, 0.10, 0.10
    else:
        return 0.85, 0.075, 0.075


def compute_page_counts(
    n_pages: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[int, int, int]:
    """
    Convert fractions to page counts, ensuring val and test each get ≥ 2 pages.
    Train gets whatever remains.
    """
    n_val  = max(2, round(n_pages * val_frac))
    n_test = max(2, round(n_pages * test_frac))

    # Make sure we don't over-allocate
    if n_val + n_test >= n_pages:
        n_val  = max(2, (n_pages - 2) // 2)
        n_test = max(2, n_pages - n_val - 1)

    n_train = n_pages - n_val - n_test
    return n_train, n_val, n_test


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def tokens_in_block(tokens, token_bboxes, block_bbox, threshold=0.5):
    def overlap(tb, bb):
        ix1, iy1 = max(tb[0], bb[0]), max(tb[1], bb[1])
        ix2, iy2 = min(tb[2], bb[2]), min(tb[3], bb[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        t_area = max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        return inter / t_area
    return " ".join(t for t, b in zip(tokens, token_bboxes) if overlap(b, block_bbox) >= threshold)


def norm_to_pixel(bbox, img_w, img_h):
    return (
        int(bbox[0] / 1000 * img_w),
        int(bbox[1] / 1000 * img_h),
        int(bbox[2] / 1000 * img_w),
        int(bbox[3] / 1000 * img_h),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Load one page
# ──────────────────────────────────────────────────────────────────────────────

def load_blocks(path: Path, use_images: bool, image_size: int) -> list[dict]:
    data         = json.loads(path.read_text(encoding="utf-8"))
    blocks       = data.get("_blocks", [])
    if not blocks:
        return []

    tokens       = data.get("tokens", [])
    token_bboxes = data.get("bboxes", [])
    img_path_str = data.get("image_path", "")
    img_size_raw = data.get("image_size", {})
    img_w        = img_size_raw.get("width",  2481)
    img_h        = img_size_raw.get("height", 3509)
    page_name    = path.stem

    page_image = None
    if use_images:
        for candidate in [Path(img_path_str), Path(img_path_str.replace("\\", "/"))]:
            if candidate.exists():
                try:
                    page_image = Image.open(candidate).convert("RGB")
                except Exception:
                    pass
                break
        if page_image is None:
            print(f"  ⚠  Image not found: {img_path_str}")

    samples = []
    for idx, block in enumerate(blocks):
        label_str = block.get("label", "")
        if label_str not in LABEL2ID:
            continue
        bbox = block["bbox"]
        text = tokens_in_block(tokens, token_bboxes, bbox)
        if not text.strip():
            continue

        sample = {
            "id":         f"{page_name}__{idx}",
            "page":       page_name,
            "image_path": img_path_str,
            "text":       text,
            "bbox":       bbox,
            "label":      LABEL2ID[label_str],
        }

        if use_images and page_image is not None:
            x1, y1, x2, y2 = norm_to_pixel(bbox, img_w, img_h)
            pad  = 4
            crop = page_image.crop((
                max(0, x1 - pad), max(0, y1 - pad),
                min(img_w, x2 + pad), min(img_h, y2 + pad),
            ))
            sample["image"] = crop.resize((image_size, image_size), Image.LANCZOS)
        else:
            sample["image"] = None

        samples.append(sample)

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Page-level 3-way split
# ──────────────────────────────────────────────────────────────────────────────

def three_way_split(
    samples_by_page: dict,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[list, list, list, dict]:
    pages = list(samples_by_page.keys())
    random.seed(seed)
    random.shuffle(pages)

    test_pages  = pages[:n_test]
    val_pages   = pages[n_test : n_test + n_val]
    train_pages = pages[n_test + n_val :]

    train = [s for p in train_pages for s in samples_by_page[p]]
    val   = [s for p in val_pages   for s in samples_by_page[p]]
    test  = [s for p in test_pages  for s in samples_by_page[p]]

    page_map = {
        "train": sorted(train_pages),
        "val":   sorted(val_pages),
        "test":  sorted(test_pages),
    }
    return train, val, test, page_map


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Dataset
# ──────────────────────────────────────────────────────────────────────────────

def to_hf_dataset(samples: list[dict], use_images: bool) -> Dataset:
    keys = ["id", "page", "image_path", "text", "bbox", "label"]
    if use_images:
        keys.append("image")
    flat = {k: [s[k] for s in samples] for k in keys}

    features = Features({
        "id":         Value("string"),
        "page":       Value("string"),
        "image_path": Value("string"),
        "text":       Value("string"),
        "bbox":       Sequence(Value("int32"), length=4),
        "label":      ClassLabel(num_classes=len(LABELS), names=LABELS),
    })
    if use_images:
        features["image"] = HFImage()

    return Dataset.from_dict(flat, features=features)


def label_dist(samples: list[dict]) -> dict:
    d = {}
    for s in samples:
        l = LABELS[s["label"]]
        d[l] = d.get(l, 0) + 1
    return d


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build 3-way block-level HF Dataset with automatic page-level split."
    )
    parser.add_argument("--input",      default="dataset/labeled")
    parser.add_argument("--output",     default="dataset/hf_block_dataset")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-images",  action="store_true")
    parser.add_argument("--image-size", type=int,   default=224)
    # Optional manual ratio override
    parser.add_argument("--train",  type=float, default=None, help="Train fraction override, e.g. 0.8")
    parser.add_argument("--val",    type=float, default=None, help="Val fraction override,   e.g. 0.1")
    parser.add_argument("--test",   type=float, default=None, help="Test fraction override,  e.g. 0.1")
    args = parser.parse_args()

    use_images = not args.no_images
    json_files = sorted(Path(args.input).glob("*_labeled.json"))

    if not json_files:
        print(f"No *_labeled.json files in {args.input}")
        sys.exit(1)

    # ── Load all pages ──
    print(f"Loading {len(json_files)} pages  |  images={use_images}\n")
    samples_by_page = {}
    for jf in json_files:
        blocks = load_blocks(jf, use_images, args.image_size)
        if blocks:
            samples_by_page[jf.stem] = blocks
            d = label_dist(blocks)
            print(f"  {jf.name:45s}  blocks={len(blocks):3d}  {d}")

    n_pages = len(samples_by_page)
    if n_pages < 5:
        print(f"\nError: need at least 5 pages to make a 3-way split, got {n_pages}")
        sys.exit(1)

    # ── Determine split fractions ──
    if args.train and args.val and args.test:
        total = args.train + args.val + args.test
        train_frac = args.train / total
        val_frac   = args.val   / total
        test_frac  = args.test  / total
        source = "manual"
    else:
        train_frac, val_frac, test_frac = auto_split_ratio(n_pages)
        source = "auto"

    n_train, n_val, n_test = compute_page_counts(n_pages, train_frac, val_frac, test_frac)

    print(f"\n{'─'*60}")
    print(f"Pages total : {n_pages}  →  split ratio ({source}): "
          f"train={train_frac:.0%} / val={val_frac:.0%} / test={test_frac:.0%}")
    print(f"Pages       : train={n_train}  val={n_val}  test={n_test}")

    # ── Split ──
    train, val, test, page_map = three_way_split(
        samples_by_page, n_val=n_val, n_test=n_test, seed=args.seed
    )

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        pages_in_split = page_map[split_name]
        print(f"  {split_name:5s}  {len(split):3d} blocks  {label_dist(split)}")
        print(f"         pages: {[p.replace('_labeled','') for p in pages_in_split]}")

    # ── Build & save ──
    out_dir = Path(args.output)
    ds_dict = DatasetDict({
        "train": to_hf_dataset(train, use_images),
        "val":   to_hf_dataset(val,   use_images),
        "test":  to_hf_dataset(test,  use_images),
    })
    ds_dict.save_to_disk(str(out_dir))

    (out_dir / "label2id.json").write_text(json.dumps(LABEL2ID, indent=2))
    (out_dir / "id2label.json").write_text(json.dumps(ID2LABEL, indent=2))
    (out_dir / "split_info.json").write_text(json.dumps({
        "n_pages":      n_pages,
        "split_ratio":  {"train": train_frac, "val": val_frac, "test": test_frac},
        "split_source": source,
        "pages":        page_map,
    }, indent=2))

    total_blocks = len(train) + len(val) + len(test)
    print(f"\n{'='*60}")
    print(f"Total blocks : {total_blocks}  {label_dist(train + val + test)}")
    print(f"Saved to     : {out_dir.resolve()}")
    print(f"\nLoad with:")
    print(f"  from datasets import load_from_disk")
    print(f"  ds = load_from_disk('{out_dir}')")
    print(f"  ds['train'], ds['val'], ds['test']")


if __name__ == "__main__":
    main()