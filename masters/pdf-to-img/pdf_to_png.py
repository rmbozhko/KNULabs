#!/usr/bin/env python3
"""
Convert a PDF file to images at 300 DPI.

Usage:
    python pdf_to_images.py <path_to_pdf>

Output:
    dataset/
      images/
        page_001.png
        page_002.png
        ...
"""

import sys
import argparse
from pathlib import Path

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Error: pdf2image is not installed.")
    print("Install it with: pip install pdf2image")
    print("You may also need poppler: https://poppler.freedesktop.org/")
    sys.exit(1)


def pdf_to_images(pdf_path: str, output_dir: str = "dataset/images", dpi: int = 300) -> None:
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    if pdf_file.suffix.lower() != ".pdf":
        print(f"Error: Expected a .pdf file, got: {pdf_file.suffix}")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {pdf_file.name}")
    print(f"Output dir: {out_path.resolve()}")
    print(f"DPI: {dpi}")
    print()

    pages = convert_from_path(str(pdf_file), dpi=dpi)
    total = len(pages)

    for i, page in enumerate(pages, start=1):
        filename = out_path / f"page_{i:03d}.png"
        page.save(str(filename), "PNG")
        print(f"  [{i}/{total}] Saved {filename.name}")

    print(f"\nDone — {total} page(s) saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a PDF into PNG images at 300 DPI."
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "--output-dir",
        default="dataset/images",
        help="Output directory (default: dataset/images)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI (default: 300)",
    )

    args = parser.parse_args()
    pdf_to_images(args.pdf, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()