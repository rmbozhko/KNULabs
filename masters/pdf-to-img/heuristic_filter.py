"""
heuristic_filter.py
--------------------
Production-grade "Domain Knowledge Layer" for ESL textbook token blocks.

Takes the output of merge_token_blocks.py (merged_blocks.csv) and applies
two complementary heuristic override strategies:

  1. TEXT HEURISTICS — whitelist of ESL instructional keywords/phrases.
     If a block is classified as CONTENT but its text contains any of these,
     it is overridden to INSTRUCTION.

  2. VISUAL HEURISTICS (OpenCV) — if the source images are available, the
     region-of-interest (ROI) cropped from the block's bounding box is
     analysed for:
       a. Background brightness — shaded/highlighted blocks are instructions.
       b. Background colour saturation — coloured tint bands are instructions.
     The CV path is skipped gracefully when images are not found on disk.

Output adds three columns to the CSV:
  - Text_Override   : True if text heuristic triggered
  - Visual_Override : True if CV heuristic triggered  (False when images absent)
  - Final_Pred      : The final label after all overrides

Usage
-----
    python heuristic_filter.py merged_blocks.csv -o filtered_blocks.csv

    # With images available:
    python heuristic_filter.py merged_blocks.csv \\
        --image-root /content/dataset/images \\
        -o filtered_blocks.csv

    # Tune thresholds:
    python heuristic_filter.py merged_blocks.csv \\
        --brightness-thresh 210 --saturation-thresh 30
"""

import re
import os
import argparse
import ast
from pathlib import Path

import pandas as pd

# ── optional CV import ─────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("[WARN] OpenCV not available – visual heuristics disabled.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INSTRUCTIONAL KEYWORD WHITELIST
# ═══════════════════════════════════════════════════════════════════════════════
#
# Organised by pedagogical category so the list is easy to audit/extend.
# All matching is case-insensitive.  Phrases take priority over single words.

INSTRUCTION_PHRASES: list[str] = [
    # ── Activity directives ────────────────────────────────────────────────
    "listen to",
    "listen again",
    "listen and",
    "read the",
    "read and",
    "look at",
    "look and",
    "write the",
    "write a",
    "write your",
    "write down",
    "tick the",
    "tick items",
    "check the",
    "check your",
    "check and",
    "match the",
    "match each",
    "choose the",
    "choose the correct",
    "complete the",
    "complete with",
    "fill in",
    "fill the",
    "answer the",
    "answer these",
    "discuss the",
    "discuss whether",
    "discuss how",
    "compare the",
    "compare your",
    "find the",
    "find a",
    "underline the",
    "circle the",
    "label the",
    "order the",
    "rank the",
    "sort the",
    "put the",
    "use the words",
    "use the phrases",
    "use the expressions",
    "use the correct",
    "use the",
    "replace the",
    "correct the",
    "translate the",
    "say the",
    "tell the",
    "describe the",
    "explain the",
    "note the",
    "note down",
    "work in",
    "work with",
    "work in pairs",
    "work in groups",
    "work together",
    "ask and answer",
    "ask your partner",
    "discuss with",
    "talk about",
    "talk to",
    "make a list",
    "make notes",
    "make sure",
    "think about",
    "think of",
    "decide whether",
    "decide which",
    "number the",
    "number these",
    "name the",
    "name a",
    "cross out",
    "cross off",
    "do the quiz",
    "do the exercise",
    "do the task",
    "do the activity",
    "answer questions",
    "answer question",
    "questions below",
    "then discuss",
    "then listen",
    "then read",
    "then check",
    "then answer",
    "then write",
    "then compare",
    # ── Section/exercise headers ───────────────────────────────────────────
    "exercise",
    "task",
    "activity",
    "practice",
    "exam focus",
    "language focus",
    "word store",
    "speaking",
    "listening",
    "reading",
    "writing",
    "grammar",
    "vocabulary",
    "pronunciation",
    "warm up",
    "warm-up",
    "lead-in",
    "follow up",
    "follow-up",
    "extension",
    "homework",
    "project",
    "review",
    "test yourself",
    "self-check",
    "challenge",
    # ── Instruction-only verbs (sentence-initial position) ─────────────────
    # detected via leading-word regex separately (see TEXT_TRIGGER_VERBS)
]

# Verbs that are instructional only when they START a sentence/block
# (avoids false positives like "She read the letter" in passage text)
TEXT_TRIGGER_VERBS: list[str] = [
    "read", "listen", "watch", "look", "write", "speak", "say",
    "discuss", "answer", "complete", "fill", "match", "choose",
    "check", "tick", "circle", "underline", "order", "rank", "sort",
    "label", "find", "identify", "describe", "explain", "translate",
    "correct", "replace", "rewrite", "compare", "number", "name",
    "work", "ask", "tell", "make", "think", "decide", "cross",
    "note", "put", "use",
]

# Regex: verb appears as the first real word in the block (ignoring leading
# digits, punctuation like "1.", "2)" etc.)
_LEADING_VERB_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)*\s*(" + "|".join(TEXT_TRIGGER_VERBS) + r")\b",
    re.IGNORECASE,
)

# Precompile phrase patterns (longest-first for efficiency)
_PHRASE_PATTERNS = [
    re.compile(r"\b" + re.escape(ph) + r"\b", re.IGNORECASE)
    for ph in sorted(INSTRUCTION_PHRASES, key=len, reverse=True)
]


def text_is_instruction(text: str) -> tuple[bool, str]:
    """
    Returns (override: bool, reason: str).
    Checks phrase whitelist first, then leading-verb rule.
    """
    # 1. Phrase whitelist
    for pat in _PHRASE_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"phrase:{m.group(0).lower()}"

    # 2. Leading-verb rule
    m = _LEADING_VERB_RE.match(text)
    if m:
        return True, f"leading_verb:{m.group(1).lower()}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VISUAL HEURISTICS (OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

# Threshold tunables (overridable via CLI)
BRIGHTNESS_THRESH  = 220   # mean pixel brightness below this → shaded block
SATURATION_THRESH  = 25    # mean saturation above this → coloured tint detected
DARK_BG_THRESH     = 80    # mean brightness below this → dark/inverted block


def _load_image(image_path: str, image_root: str | None) -> "np.ndarray | None":
    """
    Try to load image from:
      1. The path as-is (absolute or relative to CWD)
      2. image_root / basename
    Returns BGR numpy array or None.
    """
    if not CV_AVAILABLE:
        return None

    candidates = [image_path]
    if image_root:
        candidates.append(os.path.join(image_root, Path(image_path).name))

    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c)
            if img is not None:
                return img
    return None


_image_cache: dict[str, "np.ndarray | None"] = {}


def get_image(image_path: str, image_root: str | None) -> "np.ndarray | None":
    if image_path not in _image_cache:
        _image_cache[image_path] = _load_image(image_path, image_root)
    return _image_cache[image_path]


def visual_is_instruction(
    image_path: str,
    x_min: int, y_min: int, x_max: int, y_max: int,
    image_root: str | None = None,
    brightness_thresh: int = BRIGHTNESS_THRESH,
    saturation_thresh: int = SATURATION_THRESH,
    dark_bg_thresh: int = DARK_BG_THRESH,
) -> tuple[bool, str]:
    """
    Returns (override: bool, reason: str).

    Visual signals for INSTRUCTION blocks in ESL textbooks:
      - Shaded/grey background band        → mean brightness < brightness_thresh
      - Coloured tint (yellow, blue, etc.) → mean HSV saturation > saturation_thresh
      - Dark/inverted background           → mean brightness < dark_bg_thresh
    """
    img = get_image(image_path, image_root)
    if img is None:
        return False, "no_image"

    h, w = img.shape[:2]
    # Clamp bbox to image bounds
    x1 = max(0, x_min)
    y1 = max(0, y_min)
    x2 = min(w, x_max)
    y2 = min(h, y_max)

    if x2 <= x1 or y2 <= y1:
        return False, "invalid_roi"

    roi_bgr = img[y1:y2, x1:x2]

    # ── Brightness check (convert to greyscale) ────────────────────────────
    grey = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(grey.mean())

    if mean_brightness < dark_bg_thresh:
        return True, f"dark_bg:brightness={mean_brightness:.1f}"

    if mean_brightness < brightness_thresh:
        # ── Saturation check to distinguish grey shade from colour tint ──
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        if mean_sat > saturation_thresh:
            return True, f"coloured_tint:sat={mean_sat:.1f},brightness={mean_brightness:.1f}"
        return True, f"shaded_bg:brightness={mean_brightness:.1f}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def apply_heuristics(
    df: pd.DataFrame,
    image_root: str | None = None,
    brightness_thresh: int = BRIGHTNESS_THRESH,
    saturation_thresh: int = SATURATION_THRESH,
) -> pd.DataFrame:
    df = df.copy()

    text_overrides   = []
    text_reasons     = []
    visual_overrides = []
    visual_reasons   = []
    final_preds      = []

    for _, row in df.iterrows():
        model_pred = row["Pred"]
        text       = str(row["Tokens"])

        # ── Text heuristic ────────────────────────────────────────────────
        t_override, t_reason = False, ""
        if model_pred == "CONTENT":
            t_override, t_reason = text_is_instruction(text)

        # ── Visual heuristic ──────────────────────────────────────────────
        v_override, v_reason = False, ""
        if model_pred == "CONTENT" and not t_override:
            # Only run CV if text heuristic didn't already trigger
            v_override, v_reason = visual_is_instruction(
                image_path        = row["ImagePath"],
                x_min             = int(row["x_min"]),
                y_min             = int(row["y_min"]),
                x_max             = int(row["x_max"]),
                y_max             = int(row["y_max"]),
                image_root        = image_root,
                brightness_thresh = brightness_thresh,
                saturation_thresh = saturation_thresh,
            )

        text_overrides.append(t_override)
        text_reasons.append(t_reason)
        visual_overrides.append(v_override)
        visual_reasons.append(v_reason)

        if t_override or v_override:
            final_preds.append("INSTRUCTION")
        else:
            final_preds.append(model_pred)

    df["Text_Override"]    = text_overrides
    df["Text_Reason"]      = text_reasons
    df["Visual_Override"]  = visual_overrides
    df["Visual_Reason"]    = visual_reasons
    df["Final_Pred"]       = final_preds

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Heuristic filtering layer for ESL token block predictions."
    )
    parser.add_argument(
        "input", nargs="?",
        default="merged_blocks.csv",
        help="Path to merged_blocks.csv (default: merged_blocks.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="filtered_blocks.csv",
        help="Output CSV path (default: filtered_blocks.csv)"
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory where page images live (enables CV heuristics)"
    )
    parser.add_argument(
        "--brightness-thresh", type=int, default=BRIGHTNESS_THRESH,
        help=f"CV: mean brightness below this flags a shaded block (default: {BRIGHTNESS_THRESH})"
    )
    parser.add_argument(
        "--saturation-thresh", type=int, default=SATURATION_THRESH,
        help=f"CV: mean HSV saturation above this flags a coloured tint (default: {SATURATION_THRESH})"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} blocks from {args.input}")
    print(f"Model label distribution:\n{df['Pred'].value_counts().to_string()}\n")

    result = apply_heuristics(
        df,
        image_root        = args.image_root,
        brightness_thresh = args.brightness_thresh,
        saturation_thresh = args.saturation_thresh,
    )

    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} filtered blocks to: {args.output}")

    # ── Report ─────────────────────────────────────────────────────────────
    n_text_overrides   = result["Text_Override"].sum()
    n_visual_overrides = result["Visual_Override"].sum()
    print(f"\n── Override summary ──────────────────────────────────")
    print(f"  Text heuristic overrides   : {n_text_overrides}")
    print(f"  Visual heuristic overrides : {n_visual_overrides}")
    print(f"  Total overrides            : {n_text_overrides + n_visual_overrides}")
    print(f"\n── Final label distribution ──────────────────────────")
    print(result["Final_Pred"].value_counts().to_string())

    print("\n── Overridden blocks ─────────────────────────────────")
    overridden = result[result["Text_Override"] | result["Visual_Override"]]
    for _, row in overridden.iterrows():
        reason = row["Text_Reason"] or row["Visual_Reason"]
        print(f"  [{row['Block_ID']:>3}] [{reason:<35}] {row['Tokens'][:70]!r}")


if __name__ == "__main__":
    main()