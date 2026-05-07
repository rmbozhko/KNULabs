"""
visualize_predictions.py
------------------------
Draws bounding boxes and predicted labels from a CSV onto images.

Usage:
    python visualize_predictions.py --images /path/to/images --csv filtered_blocks.csv
    python visualize_predictions.py --images /path/to/images --csv filtered_blocks.csv --output ./results
    python visualize_predictions.py --images /path/to/images --csv filtered_blocks.csv --field Pred

The CSV must contain columns:
    ImagePath, x_min, y_min, x_max, y_max, Final_Pred (or another label column),
    Block_ID, Confidence_Mean

Coordinate system
-----------------
Before training, bounding boxes were normalized to a 0–1000 scale and then
each block was cropped to 224×224 px.  The values stored in the CSV are in
the 0–1000 normalized space, so this script reverses the transform:

    pixel_x = int(norm_x / 1000 * img_w)
    pixel_y = int(norm_y / 1000 * img_h)

This maps coordinates back to the original image dimensions correctly.

The script matches images by filename only (ignoring the directory stored in
ImagePath), so images can live anywhere on your machine — just point
--images at the folder.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── Colour palette ──────────────────────────────────────────────────────────
# Maps label → (box_colour, text_bg_colour).  Unknown labels get a grey fallback.
PALETTE: dict[str, tuple[str, str]] = {
    "INSTRUCTION": ("#E63946", "#E63946"),   # red
    "CONTENT":     ("#2A9D8F", "#2A9D8F"),   # teal
    "HEADER":      ("#E9C46A", "#E9C46A"),   # amber
    "FOOTER":      ("#F4A261", "#F4A261"),   # orange
    "TABLE":       ("#457B9D", "#457B9D"),   # steel-blue
    "FIGURE":      ("#A8DADC", "#A8DADC"),   # light-blue
}
FALLBACK_COLOUR = ("#6C757D", "#6C757D")

BOX_ALPHA      = 40    # 0-255: fill transparency inside the box
BOX_THICKNESS  = 3     # border thickness in pixels
FONT_SIZE      = 14    # label font size (scaled later for large images)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.ImageFont:
    """Try to load a truetype font; fall back to PIL's built-in bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def colour_with_alpha(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    """Convert '#RRGGBB' + alpha int → RGBA tuple."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


def denorm(coord: float, dim: int) -> int:
    """Reverse the pre-training normalisation: 0-1000 scale → pixel coordinate.

    Mirrors the original transform used before training:
        norm_to_pixel(bbox, img_w, img_h) → int(bbox[i] / 1000 * dim)
    """
    return int(coord / 1000 * dim)


def draw_blocks(image_path: Path, blocks: pd.DataFrame, label_col: str) -> Image.Image:
    """Draw all predicted blocks onto a copy of the image and return it."""
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Scale font relative to image size so labels are legible on any resolution
    scale    = max(1.0, min(width, height) / 800)
    font_px  = max(12, int(FONT_SIZE * scale))
    font     = get_font(font_px)
    box_w    = max(2, int(BOX_THICKNESS * scale))

    overlay  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_ov  = ImageDraw.Draw(overlay)   # semi-transparent fills
    draw_img = ImageDraw.Draw(img)       # opaque borders and text

    for _, row in blocks.iterrows():
        label      = str(row[label_col]).upper()
        confidence = row.get("Confidence_Mean", None)
        block_id   = row.get("Block_ID", "")
        box_colour, txt_bg = PALETTE.get(label, FALLBACK_COLOUR)

        # Coords are stored in 0-1000 normalised space → convert back to pixels
        x0 = denorm(row["x_min"], width)
        y0 = denorm(row["y_min"], height)
        x1 = denorm(row["x_max"], width)
        y1 = denorm(row["y_max"], height)
        # Guard against swapped coords
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        # Semi-transparent fill
        draw_ov.rectangle(
            [x0, y0, x1, y1],
            fill=colour_with_alpha(box_colour, BOX_ALPHA),
        )
        # Solid border
        draw_img.rectangle(
            [x0, y0, x1, y1],
            outline=box_colour,
            width=box_w,
        )

        # Label tag (top-left corner of the box)
        conf_str = f" {confidence:.2f}" if pd.notna(confidence) else ""
        tag      = f" #{block_id} {label}{conf_str} "

        # Measure text so we can draw a background pill
        bbox_text = draw_img.textbbox((0, 0), tag, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        tag_x = x0
        tag_y = max(0, y0 - th - 4)          # sit just above the box

        # Background rectangle for the label
        draw_img.rectangle(
            [tag_x, tag_y, tag_x + tw, tag_y + th + 4],
            fill=txt_bg,
        )
        draw_img.text((tag_x, tag_y + 2), tag, fill="white", font=font)

    # Composite overlay onto original
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize predicted bounding-box labels on images."
    )
    parser.add_argument(
        "--images", "-i", required=True,
        help="Directory that contains the image files.",
    )
    parser.add_argument(
        "--csv", "-c", required=True,
        help="Path to the CSV file (e.g. filtered_blocks.csv).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Directory to save annotated images.  "
             "Defaults to '<images_dir>/annotated/'.",
    )
    parser.add_argument(
        "--field", "-f", default="Final_Pred",
        help="CSV column to use as the displayed label.  "
             "Defaults to 'Final_Pred'.  Other useful values: 'Pred'.",
    )
    parser.add_argument(
        "--ext", default=None,
        help="Force output file extension (e.g. '.jpg').  "
             "Defaults to same as input.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    csv_path   = Path(args.csv).resolve()
    out_dir    = Path(args.output).resolve() if args.output else images_dir / "annotated"

    # ── Validate inputs ──
    if not images_dir.is_dir():
        sys.exit(f"[ERROR] Images directory not found: {images_dir}")
    if not csv_path.is_file():
        sys.exit(f"[ERROR] CSV file not found: {csv_path}")

    # ── Load CSV ──
    df = pd.read_csv(csv_path)
    required_cols = {"ImagePath", "x_min", "y_min", "x_max", "y_max", args.field}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV is missing columns: {missing}")

    # Extract just the filename from the stored path so we can match it
    # to whatever directory the user points us at.
    df["_filename"] = df["ImagePath"].apply(lambda p: Path(p).name)

    # Group blocks by filename
    groups: dict[str, pd.DataFrame] = {
        fname: subset for fname, subset in df.groupby("_filename")
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build a filename → Path map from the images directory ──
    image_map: dict[str, Path] = {}
    for f in images_dir.iterdir():
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            image_map[f.name] = f

    matched   = 0
    unmatched = []

    for csv_filename, blocks in groups.items():
        image_path = image_map.get(csv_filename)

        if image_path is None:
            unmatched.append(csv_filename)
            continue

        print(f"  Processing  {csv_filename}  ({len(blocks)} blocks) …", end=" ")

        try:
            annotated = draw_blocks(image_path, blocks, label_col=args.field)
        except Exception as exc:
            print(f"[SKIP] Could not process: {exc}")
            continue

        # Keep original filename
        suffix   = Path(args.ext).suffix if args.ext else image_path.suffix
        out_path = out_dir / (image_path.stem + "_annotated" + suffix)
        annotated.save(out_path)
        print(f"→ saved to {out_path}")
        matched += 1

    # ── Summary ──
    print()
    print(f"Done.  {matched} image(s) annotated → {out_dir}")
    if unmatched:
        print(f"[WARN] {len(unmatched)} filename(s) from CSV had no match in {images_dir}:")
        for name in unmatched:
            print(f"       • {name}")


if __name__ == "__main__":
    main()