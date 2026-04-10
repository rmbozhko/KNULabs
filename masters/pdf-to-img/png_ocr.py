#!/usr/bin/env python3
"""
OCR pipeline: PNG → tokens + normalized bboxes → JSON

Steps:
  1. Run Tesseract OCR on each PNG
  2. Draw bounding boxes for visual verification
  3. Normalize bboxes to [0, 1000] for LayoutLMv3
  4. Save per-page JSON results

Usage:
    python ocr_pipeline.py                        # process dataset/images/*.png
    python ocr_pipeline.py --images path/to/imgs  # custom image folder
    python ocr_pipeline.py --verify               # save bbox debug images
    python ocr_pipeline.py --lang ukr+eng         # set Tesseract language

Output:
    dataset/
      ocr/
        page_001.json
        page_002.json
        ...
      debug/           (only with --verify)
        page_001_boxes.png
        ...
"""

import argparse
import json
from pathlib import Path

try:
    import pytesseract
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: missing dependencies.")
    print("Install with: pip install pytesseract pillow")
    print("Also install Tesseract: https://github.com/tesseract-ocr/tesseract")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

# TODO: switch from text level to block level (alias as paragraph)
# кластеризація, x/y cut, 
# merging nearby boxes 
def run_ocr(image: Image.Image, lang: str = "eng") -> tuple[list[str], list[list[int]]]:
    """Return raw (tokens, bboxes) from Tesseract. Bboxes are in pixel coords."""
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    tokens, bboxes = [], []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append(word)
        bboxes.append([x, y, x + w, y + h])

    return tokens, bboxes


def normalize_bbox(box: list[int], width: int, height: int) -> list[int]:
    """Scale pixel bbox to [0, 1000] as expected by LayoutLMv3."""
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def draw_boxes(image: Image.Image, bboxes: list[list[int]], color: str = "red") -> Image.Image:
    """Return a copy of the image with bounding boxes drawn."""
    debug = image.copy().convert("RGB")
    draw = ImageDraw.Draw(debug)
    for box in bboxes:
        draw.rectangle(box, outline=color, width=1)
    return debug


def process_image(
    image_path: Path,
    lang: str = "eng",
    verify: bool = False,
    ocr_dir: Path = Path("dataset/ocr"),
    debug_dir: Path = Path("dataset/debug"),
) -> dict:
    image = Image.open(image_path)
    width, height = image.size

    tokens, pixel_bboxes = run_ocr(image, lang=lang)
    norm_bboxes = [normalize_bbox(b, width, height) for b in pixel_bboxes]

    result = {
        "image_path": str(image_path),
        "image_size": {"width": width, "height": height},
        "tokens": tokens,
        "bboxes": norm_bboxes,          # normalized [0,1000]
        "bboxes_pixel": pixel_bboxes,   # raw pixels (useful for debugging)
    }

    # Save JSON
    ocr_dir.mkdir(parents=True, exist_ok=True)
    out_json = ocr_dir / (image_path.stem + ".json")
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optionally save debug image
    if verify:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_img = draw_boxes(image, pixel_bboxes)
        debug_img.save(debug_dir / (image_path.stem + "_boxes.png"))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OCR pipeline for LayoutLMv3 dataset prep.")
    parser.add_argument("--images",   default="dataset/images", help="Folder with PNG files")
    parser.add_argument("--output",   default="dataset/ocr",    help="Output folder for JSON")
    parser.add_argument("--debug",    default="dataset/debug",  help="Output folder for debug images")
    parser.add_argument("--lang",     default="eng",            help="Tesseract language code(s), e.g. ukr+eng")
    parser.add_argument("--verify",   action="store_true",      help="Save bbox debug images")
    parser.add_argument("--limit",    type=int, default=0,      help="Process only first N images (0 = all)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Error: images folder not found: {images_dir}")
        raise SystemExit(1)

    png_files = sorted(images_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {images_dir}")
        raise SystemExit(1)

    if args.limit:
        png_files = png_files[: args.limit]

    print(f"Found {len(png_files)} image(s)  |  lang={args.lang}  |  verify={args.verify}\n")

    for i, img_path in enumerate(png_files, 1):
        print(f"[{i}/{len(png_files)}] {img_path.name} ... ", end="", flush=True)
        result = process_image(
            img_path,
            lang=args.lang,
            verify=args.verify,
            ocr_dir=Path(args.output),
            debug_dir=Path(args.debug),
        )
        print(f"{len(result['tokens'])} tokens")

    print(f"\n✅ Done — JSON saved to: {Path(args.output).resolve()}")
    if args.verify:
        print(f"🖼  Debug images saved to: {Path(args.debug).resolve()}")

    # Quick sample preview
    if png_files:
        sample_json = Path(args.output) / (png_files[0].stem + ".json")
        sample = json.loads(sample_json.read_text(encoding="utf-8"))
        print("\n--- Sample (first 5 tokens) ---")
        for token, bbox in zip(sample["tokens"][:5], sample["bboxes"][:5]):
            print(f"  {token!r:20s}  bbox={bbox}")


if __name__ == "__main__":
    main()