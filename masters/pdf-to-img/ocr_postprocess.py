"""
ocr_postprocess.py — clean up Tesseract OCR output before annotation.

Fixes:
  - Remove garbage tokens (single chars, symbols, checkbox artifacts)
  - Flag low-confidence tokens (if conf data available)
  - Basic stats report per page

Usage:
    python ocr_postprocess.py                  # process all dataset/ocr/*.json
    python ocr_postprocess.py --min-len 2      # drop tokens shorter than 2 chars
    python ocr_postprocess.py --report         # print per-page stats only
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Garbage patterns — tokens to drop
# ---------------------------------------------------------------------------

# Single non-alphanumeric characters
_SINGLE_GARBAGE = re.compile(r"^[^a-zA-Z0-9]+$")

# Tesseract checkbox / bullet artifacts
_CHECKBOX_ARTIFACTS = {"O", "o", "oO", "oe", "OC", "CO", "C)", "C0", "'@)", "(@)"}

# Strings that are purely punctuation/symbols after stripping
_PUNCT_ONLY = re.compile(r"^[\W_]+$")


def is_garbage(token: str, min_len: int = 1) -> bool:
    if len(token) < min_len:
        return True
    if token in _CHECKBOX_ARTIFACTS:
        return True
    if _SINGLE_GARBAGE.match(token):
        return True
    # Pure punctuation strings longer than 2 chars are almost always OCR noise
    if len(token) > 2 and _PUNCT_ONLY.match(token):
        return True
    return False


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def clean_page(data: dict, min_len: int = 1) -> dict:
    tokens_in  = data["tokens"]
    bboxes_in  = data["bboxes"]
    pixels_in  = data.get("bboxes_pixel", [None] * len(tokens_in))

    tokens_out, bboxes_out, pixels_out = [], [], []
    dropped = []

    for token, bbox, pixel in zip(tokens_in, bboxes_in, pixels_in):
        if is_garbage(token, min_len):
            dropped.append(token)
        else:
            tokens_out.append(token)
            bboxes_out.append(bbox)
            if pixel is not None:
                pixels_out.append(pixel)

    result = dict(data)
    # relative_path = Path(data["image_path"])
    # absolute_path = relative_path.resolve()

    result["image"] = f"http://localhost:8081/{data["image_path"].replace('\\', '/')}"
    result["tokens"]       = tokens_out
    result["bboxes"]       = bboxes_out
    result["bboxes_pixel"] = pixels_out
    result["_dropped"]     = dropped          # keep for audit
    result["_stats"] = {
        "tokens_before": len(tokens_in),
        "tokens_after":  len(tokens_out),
        "dropped":       len(dropped),
    }
    return result


def process_all(ocr_dir: Path, out_dir: Path, min_len: int, report_only: bool):
    json_files = sorted(ocr_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {ocr_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    total_before = total_after = total_dropped = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        cleaned = clean_page(data, min_len)

        s = cleaned["_stats"]
        total_before  += s["tokens_before"]
        total_after   += s["tokens_after"]
        total_dropped += s["dropped"]

        print(
            f"  {jf.name:30s}  "
            f"before={s['tokens_before']:4d}  "
            f"after={s['tokens_after']:4d}  "
            f"dropped={s['dropped']:3d}"
        )
        if cleaned["_dropped"]:
            print(f"    dropped: {cleaned['_dropped']}")

        if not report_only:
            out_path = out_dir / jf.name
            out_path.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(f"\n{'='*60}")
    print(f"Total  before={total_before}  after={total_after}  dropped={total_dropped}")
    if not report_only:
        print(f"Cleaned JSON saved to: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-process Tesseract OCR JSON output.")
    parser.add_argument("--input",    default="dataset/ocr",         help="Folder with raw OCR JSON files")
    parser.add_argument("--output",   default="dataset/ocr_clean",   help="Output folder for cleaned JSON")
    parser.add_argument("--min-len",  type=int, default=1,           help="Minimum token length to keep (default 1)")
    parser.add_argument("--report",   action="store_true",           help="Print stats only, don't write files")
    args = parser.parse_args()

    process_all(
        ocr_dir     = Path(args.input),
        out_dir     = Path(args.output),
        min_len     = args.min_len,
        report_only = args.report,
    )


if __name__ == "__main__":
    main()