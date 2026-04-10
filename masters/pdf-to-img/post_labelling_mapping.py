#!/usr/bin/env python3
"""
ls_to_token_labels.py
─────────────────────
Reads a Label Studio export JSON (single file with all tasks),
maps each OCR token to the block label it belongs to,
and writes one output JSON per page.

Input format (Label Studio export):
  annotations[0].result[*].value  →  block rectangles in % coords
  data.tokens                     →  list of word strings
  data.bboxes                     →  normalized [0,1000] coords  [x1,y1,x2,y2]
  data.image_size                 →  {width, height}

Output (one file per task):
  {
    "image_path": "...",
    "image_size": {...},
    "tokens":  ["word", ...],
    "bboxes":  [[x1,y1,x2,y2], ...],   # normalized [0,1000]
    "labels":  ["INSTRUCTION", "CONTENT", "OTHER", ...]
  }

Usage:
    python ls_to_token_labels.py --input export.json
    python ls_to_token_labels.py --input export.json --iou 0.3 --output dataset/labeled
    python ls_to_token_labels.py --input export.json --report   # stats only
"""

import argparse
import json
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers  (all coordinates normalised to [0, 1000])
# ──────────────────────────────────────────────────────────────────────────────

def pct_to_norm(value: dict, img_w: int, img_h: int) -> list[int]:
    """
    Convert a Label Studio percentage rectangle to [0,1000] normalised coords.
    LS stores x,y as % of image dimensions, top-left corner.
    """
    x1 = int(value["x"] / 100 * 1000)
    y1 = int(value["y"] / 100 * 1000)
    x2 = int((value["x"] + value["width"])  / 100 * 1000)
    y2 = int((value["y"] + value["height"]) / 100 * 1000)
    return [x1, y1, x2, y2]


def intersection_area(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def box_area(b: list[int]) -> float:
    return float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))


def overlap_ratio(token_box: list[int], block_box: list[int]) -> float:
    """
    Fraction of the token box covered by the block box.
    Returns 0.0–1.0.  We use token-coverage (not IoU) so that small tokens
    inside large blocks are correctly assigned even if the block is huge.
    """
    inter = intersection_area(token_box, block_box)
    t_area = box_area(token_box)
    if t_area == 0:
        return 0.0
    return inter / t_area


# ──────────────────────────────────────────────────────────────────────────────
# Core mapping logic
# ──────────────────────────────────────────────────────────────────────────────

def extract_blocks(annotation_result: list[dict], img_w: int, img_h: int) -> list[dict]:
    """
    Pull all rectanglelabels regions out of one annotation result list.
    Returns list of {"label": str, "bbox": [x1,y1,x2,y2] in [0,1000]}.
    """
    blocks = []
    for region in annotation_result:
        if region.get("type") != "rectanglelabels":
            continue
        v = region["value"]
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        bbox = pct_to_norm(v, img_w, img_h)
        blocks.append({"label": labels[0], "bbox": bbox})
    return blocks


def assign_labels(
    tokens: list[str],
    bboxes: list[list[int]],
    blocks: list[dict],
    threshold: float = 0.5,
    default_label: str = "OTHER",
) -> list[str]:
    """
    For every token find the block with the highest overlap ratio.
    If best overlap >= threshold → assign that block's label.
    Otherwise → default_label.
    """
    labels = []
    for bbox in bboxes:
        best_ratio = 0.0
        best_label = default_label
        for block in blocks:
            ratio = overlap_ratio(bbox, block["bbox"])
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = block["label"]
        labels.append(best_label if best_ratio >= threshold else default_label)
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Per-task processing
# ──────────────────────────────────────────────────────────────────────────────

def process_task(task: dict, threshold: float) -> dict | None:
    """Process a single Label Studio task. Returns None if no annotation."""
    annotations = task.get("annotations", [])
    if not annotations:
        return None

    # Use first non-cancelled annotation
    annotation = None
    for ann in annotations:
        if not ann.get("was_cancelled", False):
            annotation = ann
            break
    if annotation is None:
        return None

    data = task.get("data", {})
    tokens  = data.get("tokens", [])
    bboxes  = data.get("bboxes", [])
    img_size = data.get("image_size", {})
    img_w   = img_size.get("width",  1)
    img_h   = img_size.get("height", 1)

    if not tokens or not bboxes:
        return None

    blocks = extract_blocks(annotation["result"], img_w, img_h)
    if not blocks:
        return None

    labels = assign_labels(tokens, bboxes, blocks, threshold=threshold)

    return {
        "task_id":    task.get("id"),
        "image_path": data.get("image_path", ""),
        "image_size": img_size,
        "tokens":     tokens,
        "bboxes":     bboxes,
        "labels":     labels,
        # ── extra: keep block-level info for debugging ──
        "_blocks": blocks,
        "_stats": {
            "total_tokens": len(tokens),
            "labeled":  sum(1 for l in labels if l != "OTHER"),
            "OTHER":    sum(1 for l in labels if l == "OTHER"),
            "label_counts": {
                lbl: labels.count(lbl) for lbl in sorted(set(labels))
            },
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Map Label Studio block annotations → per-token labels."
    )
    parser.add_argument("--input",   required=True,              help="Label Studio export JSON file")
    parser.add_argument("--output",  default="mapped",  help="Output folder")
    parser.add_argument("--iou",     type=float, default=0.5,    help="Min token-coverage threshold (default 0.5)")
    parser.add_argument("--report",  action="store_true",        help="Print stats only, skip writing files")
    args = parser.parse_args()

    export_path = Path(args.input)
    if not export_path.exists():
        print(f"Error: file not found: {export_path}")
        sys.exit(1)

    raw = json.loads(export_path.read_text(encoding="utf-8"))

    # Label Studio export can be a list of tasks or a single task dict
    tasks = raw if isinstance(raw, list) else [raw]
    print(f"Loaded {len(tasks)} task(s) from {export_path.name}\n")

    out_dir = Path(args.output)
    if not args.report:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = total_labeled = total_other = 0
    processed = 0

    for task in tasks:
        result = process_task(task, threshold=args.iou)
        if result is None:
            task_id = task.get("id", "?")
            print(f"  [task {task_id}] skipped (no annotation or no tokens)")
            continue

        s = result["_stats"]
        total_tokens  += s["total_tokens"]
        total_labeled += s["labeled"]
        total_other   += s["OTHER"]
        processed     += 1

        stem = Path(result["image_path"]).stem or f"task_{result['task_id']}"
        print(
            f"  {stem:40s}  tokens={s['total_tokens']:4d}  "
            f"labeled={s['labeled']:4d}  OTHER={s['OTHER']:3d}  "
            f"counts={s['label_counts']}"
        )

        if not args.report:
            out_path = out_dir / f"{stem}_labeled.json"
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\n{'='*70}")
    print(f"Processed : {processed}/{len(tasks)} tasks")
    print(f"Tokens    : total={total_tokens}  labeled={total_labeled}  OTHER={total_other}")
    if not args.report:
        print(f"Output    : {out_dir.resolve()}")


if __name__ == "__main__":
    main()