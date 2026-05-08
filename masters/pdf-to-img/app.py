"""
ESL Textbook → Excalidraw
Streamlit app — single PNG upload, LayoutLMv3 inference, Excalidraw JSON download.

Required local files (set MODEL_DIR below or via the sidebar):
  config.json, model.safetensors, processor_config.json,
  tokenizer_config.json, tokenizer.json, training_args.bin
"""

import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESL Textbook → Excalidraw",
    page_icon="📖",
    layout="centered",
)
st.markdown(
    """
    <style>
        .block-container { max-width: 720px; padding-top: 2rem; }
        .stAlert { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration — edit MODEL_DIR or set it via the sidebar at runtime
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL_DIR       = "./model"    # ← point this at your saved model folder
DEFAULT_TESSERACT_LANG  = "eng"        # e.g. "ukr+eng" for mixed pages
CONFIDENCE_THRESHOLD    = 0.70         # softmax threshold (from your notebook)
MV_LINE_Y_TOLERANCE     = 10.0        # majority-voting y-grouping tolerance (0-1000 scale)
MV_MAJORITY_THRESHOLD   = 0.80        # majority-voting dominance threshold


# u2500u2500 Stage 4 u2014 block consolidation (toggle here, no UI exposure) u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500u2500
ENABLE_BLOCK_CONSOLIDATION = True   # set False to skip stage 4 entirely
INSTRUCTION_MERGE_Y_GAP    = 30     # max vertical gap (0-1000 units) between two
                                    # adjacent INSTRUCTION blocks to merge them;
                                    # CONTENT blocks are always merged when not
                                    # separated by an INSTRUCTION block
# ── Excalidraw visual config ───────────────────────────────────────────────────
LABEL_COLOR = {
    "INSTRUCTION": "#ffc9c9",   # pink  (matches your sample .excalidraw)
    "CONTENT":     "#b2f2bb",   # green (matches your sample .excalidraw)
}
DEFAULT_COLOR = "#eeeeee"


# ══════════════════════════════════════════════════════════════════════════════
#  Lazy model loader — cached so the model is loaded only once per session
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading LayoutLMv3 model…")
def load_model(model_dir: str):
    """Load processor + model from a local directory. Cached across reruns."""
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    import torch

    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path.resolve()}")

    processor = LayoutLMv3Processor.from_pretrained(str(path), apply_ocr=False)
    model     = LayoutLMv3ForTokenClassification.from_pretrained(str(path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # id2label: prefer model config, fall back to id2label.json next to model
    if model.config.id2label:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    else:
        id2label_path = path / "id2label.json"
        if not id2label_path.exists():
            raise FileNotFoundError(
                "id2label not found in model config or as id2label.json "
                f"in {path}"
            )
        raw      = json.loads(id2label_path.read_text())
        id2label = {int(k): v for k, v in raw.items()}

    label2id = {v: k for k, v in id2label.items()}
    return processor, model, id2label, label2id, device


# ══════════════════════════════════════════════════════════════════════════════
#  OCR garbage filter  (logic from ocr_postprocess.py)
# ══════════════════════════════════════════════════════════════════════════════
_SINGLE_GARBAGE     = re.compile(r"^[^a-zA-Z0-9]+$")
_CHECKBOX_ARTIFACTS = {"O", "o", "oO", "oe", "OC", "CO", "C)", "C0", "'@)", "(@)"}
_PUNCT_ONLY         = re.compile(r"^[\W_]+$")


def _is_garbage(token: str) -> bool:
    if not token:
        return True
    if token in _CHECKBOX_ARTIFACTS:
        return True
    if _SINGLE_GARBAGE.match(token):
        return True
    if len(token) > 2 and _PUNCT_ONLY.match(token):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING  (png_ocr.py + ocr_postprocess.py, adapted for in-memory PIL)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(image: Image.Image, lang: str = DEFAULT_TESSERACT_LANG) -> dict:
    """
    Run Tesseract OCR on `image`, strip garbage tokens, normalise bounding
    boxes to [0, 1000] (as LayoutLMv3 expects), and return:

        {
            "image":        PIL.Image (RGB),
            "tokens":       list[str],
            "bboxes":       list[[x1,y1,x2,y2]],  # normalised 0-1000
            "bboxes_pixel": list[[x1,y1,x2,y2]],  # raw pixels
            "image_size":   {"width": int, "height": int},
        }
    """
    import pytesseract

    image_rgb       = image.convert("RGB")
    width, height   = image_rgb.size

    # ── Tesseract OCR (mirrors run_ocr in png_ocr.py) ─────────────────────────
    data = pytesseract.image_to_data(
        image_rgb, lang=lang, output_type=pytesseract.Output.DICT
    )

    raw_tokens, pixel_bboxes = [], []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        raw_tokens.append(word)
        pixel_bboxes.append([x, y, x + w, y + h])

    # ── Clean (mirrors clean_page in ocr_postprocess.py) ──────────────────────
    tokens, clean_pixel, norm_bboxes = [], [], []
    for token, pbox in zip(raw_tokens, pixel_bboxes):
        if _is_garbage(token):
            continue
        tokens.append(token)
        clean_pixel.append(pbox)
        norm_bboxes.append([
            int(1000 * pbox[0] / width),
            int(1000 * pbox[1] / height),
            int(1000 * pbox[2] / width),
            int(1000 * pbox[3] / height),
        ])

    return {
        "image":        image_rgb,
        "tokens":       tokens,
        "bboxes":       norm_bboxes,
        "bboxes_pixel": clean_pixel,
        "image_size":   {"width": width, "height": height},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL INFERENCE  (mirrors preprocess_with_images + run_evaluation notebook)
# ══════════════════════════════════════════════════════════════════════════════
def run_model(preprocessed: dict, model_dir: str = DEFAULT_MODEL_DIR) -> dict:
    """
    Run LayoutLMv3 token-classification inference on the OCR output.

    Returns:
        {
            "tokens":       list[str],
            "bboxes":       list[[x1,y1,x2,y2]],   # normalised 0-1000
            "label_ids":    list[int],               # per-token predicted label id
            "label_names":  list[str],               # human-readable label per token
            "id2label":     dict[int, str],
        }
    """
    import torch
    import torch.nn.functional as F

    processor, model, id2label, label2id, device = load_model(model_dir)

    tokens = preprocessed["tokens"]
    bboxes = preprocessed["bboxes"]
    image  = preprocessed["image"]

    if not tokens:
        return {
            "tokens": [], "bboxes": [], "label_ids": [],
            "label_names": [], "id2label": id2label,
        }

    # ── Encode (mirrors preprocess_with_images in your fine-tuned notebook) ───
    encoding = processor(
        image,
        text=tokens,
        boxes=bboxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits  # shape (1, seq_len, num_labels)

    # Softmax + confidence threshold (from run_evaluation in your baseline notebook)
    probs              = F.softmax(logits, dim=-1)
    max_probs, preds   = torch.max(probs, dim=-1)
    content_id         = label2id.get("CONTENT", 0)
    preds              = torch.where(
        max_probs >= CONFIDENCE_THRESHOLD,
        preds,
        torch.tensor(content_id, device=device),
    )

    # ── Align predictions back to words via word_ids() ────────────────────────
    # The processor adds special tokens ([CLS], [SEP]) and may split words into
    # sub-tokens.  We keep the prediction for the first sub-token of each word.
    try:
        word_id_list = processor.tokenizer(
            text=tokens,
            boxes=bboxes,
            padding="max_length",
            truncation=True,
            max_length=512,
        ).word_ids()
    except Exception:
        word_id_list = None

    preds_seq = preds[0].cpu().tolist()   # (seq_len,)

    if word_id_list is not None:
        seen       = set()
        word_preds = {}
        for pos, wid in enumerate(word_id_list):
            if wid is None:
                continue
            if wid not in seen:
                seen.add(wid)
                word_preds[wid] = preds_seq[pos]
        label_ids = [word_preds.get(i, content_id) for i in range(len(tokens))]
    else:
        # Fallback: strip the leading [CLS] token
        label_ids = preds_seq[1 : len(tokens) + 1]

    label_names = [id2label.get(lid, "CONTENT") for lid in label_ids]

    return {
        "tokens":      tokens,
        "bboxes":      bboxes,
        "label_ids":   label_ids,
        "label_names": label_names,
        "id2label":    id2label,
        "image":       preprocessed["image"],   # PIL Image — needed by visual heuristics
    }


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING
#  Three stages, each a faithful port of the corresponding script:
#    1. Majority voting      — baseline notebook (apply_block_majority_voting)
#    2. Block merging        — merge_token_blocks.py
#    3. Heuristic filtering  — heuristic_filter.py
#  Followed by the Excalidraw element builder.
# ══════════════════════════════════════════════════════════════════════════════

# ── Merge thresholds (from merge_token_blocks.py) ─────────────────────────────
MERGE_Y_THRESH    = 10   # 0-1000 normalised units — same as pixel scale used in scripts
MERGE_X_GAP_THRESH = 30  # horizontal gap that starts a new block on the same line

# ── Heuristic filter keyword lists (from heuristic_filter.py) ─────────────────
_INSTRUCTION_PHRASES: list[str] = [
    "listen to", "listen again", "listen and",
    "read the", "read and",
    "look at", "look and",
    "write the", "write a", "write your", "write down",
    "tick the", "tick items",
    "check the", "check your", "check and",
    "match the", "match each",
    "choose the", "choose the correct",
    "complete the", "complete with",
    "fill in", "fill the",
    "answer the", "answer these", "answer questions", "answer question",
    "discuss the", "discuss whether", "discuss how", "discuss with",
    "compare the", "compare your",
    "find the", "find a",
    "underline the", "circle the", "label the",
    "order the", "rank the", "sort the", "put the",
    "use the words", "use the phrases", "use the expressions",
    "use the correct", "use the",
    "replace the", "correct the", "translate the",
    "say the", "tell the", "describe the", "explain the",
    "note the", "note down",
    "work in", "work with", "work in pairs", "work in groups", "work together",
    "ask and answer", "ask your partner",
    "talk about", "talk to",
    "make a list", "make notes", "make sure",
    "think about", "think of",
    "decide whether", "decide which",
    "number the", "number these",
    "name the", "name a",
    "cross out", "cross off",
    "do the quiz", "do the exercise", "do the task", "do the activity",
    "questions below",
    "then discuss", "then listen", "then read", "then check",
    "then answer", "then write", "then compare",
    "exercise", "task", "activity", "practice",
    "exam focus", "language focus", "word store",
    "speaking", "listening", "reading", "writing",
    "grammar", "vocabulary", "pronunciation",
    "warm up", "warm-up", "lead-in",
    "follow up", "follow-up", "extension",
    "homework", "project", "review",
    "test yourself", "self-check", "challenge",
]

_TEXT_TRIGGER_VERBS: list[str] = [
    "read", "listen", "watch", "look", "write", "speak", "say",
    "discuss", "answer", "complete", "fill", "match", "choose",
    "check", "tick", "circle", "underline", "order", "rank", "sort",
    "label", "find", "identify", "describe", "explain", "translate",
    "correct", "replace", "rewrite", "compare", "number", "name",
    "work", "ask", "tell", "make", "think", "decide", "cross",
    "note", "put", "use",
]

_LEADING_VERB_RE = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)*\s*(" + "|".join(_TEXT_TRIGGER_VERBS) + r")\b",
    re.IGNORECASE,
)
_PHRASE_PATTERNS = [
    re.compile(r"\b" + re.escape(ph) + r"\b", re.IGNORECASE)
    for ph in sorted(_INSTRUCTION_PHRASES, key=len, reverse=True)
]

# ── OpenCV optional import ─────────────────────────────────────────────────────
try:
    import cv2 as _cv2
    import numpy as _np
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

# CV thresholds (from heuristic_filter.py)
_BRIGHTNESS_THRESH = 220
_SATURATION_THRESH = 25
_DARK_BG_THRESH    = 80


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — majority voting  (apply_block_majority_voting, baseline notebook)
# ─────────────────────────────────────────────────────────────────────────────
def _y_centre(bbox: list) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _apply_majority_voting(
    label_ids: list,
    bboxes: list,
    line_y_tolerance: float = MV_LINE_Y_TOLERANCE,
    majority_threshold: float = MV_MAJORITY_THRESHOLD,
) -> list:
    if not label_ids:
        return label_ids

    groups: list[list[tuple]] = []
    current: list[tuple]      = []
    prev_yc                   = None

    for idx, (lid, bbox) in enumerate(zip(label_ids, bboxes)):
        yc = _y_centre(bbox)
        if prev_yc is None or abs(yc - prev_yc) <= line_y_tolerance:
            current.append((idx, lid))
        else:
            groups.append(current)
            current = [(idx, lid)]
        prev_yc = yc
    if current:
        groups.append(current)

    corrected = list(label_ids)
    for group in groups:
        indices = [i for i, _ in group]
        labels  = [lbl for _, lbl in group]
        dominant, count = Counter(labels).most_common(1)[0]
        if count / len(labels) >= majority_threshold:
            for i in indices:
                corrected[i] = dominant
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — block merging  (merge_token_blocks.py)
# ─────────────────────────────────────────────────────────────────────────────
def _majority_label(preds: list, confidences: list) -> str:
    """Majority vote; tie-break by confidence sum (from merge_token_blocks.py)."""
    counts    = Counter(preds)
    max_count = max(counts.values())
    candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    conf_sums = {
        lbl: sum(c for p, c in zip(preds, confidences) if p == lbl)
        for lbl in candidates
    }
    return max(conf_sums, key=conf_sums.get)


def _merge_token_blocks(
    tokens: list[str],
    bboxes: list[list[int]],      # normalised 0-1000
    label_names: list[str],
    y_thresh: float = MERGE_Y_THRESH,
    x_gap_thresh: float = MERGE_X_GAP_THRESH,
) -> list[dict]:
    """
    Port of merge_tokens() from merge_token_blocks.py.

    Pass 1 — line grouping by Y proximity.
    Pass 2 — column splitting by X gap within each line.
    Block label = majority vote (uniform confidence = 1.0 since we have no
    per-token confidence at this stage; voting is still meaningful).
    """
    if not tokens:
        return []

    # Build a lightweight list of token dicts sorted by reading order
    rows = sorted(
        [
            {
                "token": tokens[i],
                "label": label_names[i],
                "x":  (bboxes[i][0] + bboxes[i][2]) / 2,   # x-centre
                "y":  (bboxes[i][1] + bboxes[i][3]) / 2,   # y-centre
                "x1": bboxes[i][0], "y1": bboxes[i][1],
                "x2": bboxes[i][2], "y2": bboxes[i][3],
            }
            for i in range(len(tokens))
        ],
        key=lambda r: (r["y"], r["x"]),
    )

    # ── Pass 1: group into lines by Y proximity ────────────────────────────
    line_groups: list[list[dict]] = []
    current_line: list[dict]      = []
    current_y: float | None       = None

    for row in rows:
        if current_y is None or abs(row["y"] - current_y) <= y_thresh:
            current_line.append(row)
            current_y = sum(r["y"] for r in current_line) / len(current_line)
        else:
            line_groups.append(current_line)
            current_line = [row]
            current_y = row["y"]
    if current_line:
        line_groups.append(current_line)

    # ── Pass 2: split lines by X gap ──────────────────────────────────────
    raw_blocks: list[list[dict]] = []
    for line in line_groups:
        line_sorted  = sorted(line, key=lambda r: r["x"])
        current_block: list[dict] = []
        for row in line_sorted:
            if not current_block:
                current_block.append(row)
            else:
                gap = row["x1"] - current_block[-1]["x2"]
                if gap <= x_gap_thresh:
                    current_block.append(row)
                else:
                    raw_blocks.append(current_block)
                    current_block = [row]
        if current_block:
            raw_blocks.append(current_block)

    # ── Build block records ────────────────────────────────────────────────
    blocks = []
    for block_rows in raw_blocks:
        preds  = [r["label"] for r in block_rows]
        confs  = [1.0] * len(preds)        # uniform; sufficient for majority vote
        label  = _majority_label(preds, confs)
        x_min  = min(r["x1"] for r in block_rows)
        y_min  = min(r["y1"] for r in block_rows)
        x_max  = max(r["x2"] for r in block_rows)
        y_max  = max(r["y2"] for r in block_rows)
        blocks.append({
            "label":  label,
            "tokens": " ".join(r["token"] for r in block_rows),
            "x_min": x_min, "y_min": y_min,
            "x_max": x_max, "y_max": y_max,
            # Excalidraw geometry
            "x":      float(x_min),
            "y":      float(y_min),
            "width":  float(max(x_max - x_min, 40)),
            "height": float(max(y_max - y_min + 10, 20)),
        })
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — heuristic filtering  (heuristic_filter.py)
# ─────────────────────────────────────────────────────────────────────────────
def _text_is_instruction(text: str) -> tuple[bool, str]:
    """Port of text_is_instruction() from heuristic_filter.py."""
    for pat in _PHRASE_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"phrase:{m.group(0).lower()}"
    m = _LEADING_VERB_RE.match(text)
    if m:
        return True, f"leading_verb:{m.group(1).lower()}"
    return False, ""


def _visual_is_instruction(
    image_rgb,      # PIL Image (RGB) — passed directly, no disk I/O needed
    x_min: int, y_min: int, x_max: int, y_max: int,
) -> tuple[bool, str]:
    """
    Port of visual_is_instruction() from heuristic_filter.py.
    Works on the PIL Image already in memory instead of loading from disk.
    """
    if not _CV_AVAILABLE or image_rgb is None:
        return False, "no_cv"

    img_bgr = _np.array(image_rgb)[:, :, ::-1]   # RGB → BGR
    h, w    = img_bgr.shape[:2]
    x1 = max(0, x_min);  y1 = max(0, y_min)
    x2 = min(w, x_max);  y2 = min(h, y_max)
    if x2 <= x1 or y2 <= y1:
        return False, "invalid_roi"

    roi = img_bgr[y1:y2, x1:x2]
    grey = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY)
    mean_brightness = float(grey.mean())

    if mean_brightness < _DARK_BG_THRESH:
        return True, f"dark_bg:brightness={mean_brightness:.1f}"

    if mean_brightness < _BRIGHTNESS_THRESH:
        hsv      = _cv2.cvtColor(roi, _cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean())
        if mean_sat > _SATURATION_THRESH:
            return True, f"coloured_tint:sat={mean_sat:.1f}"
        return True, f"shaded_bg:brightness={mean_brightness:.1f}"

    return False, ""


def _apply_heuristics(blocks: list[dict], image_rgb) -> list[dict]:
    """
    Port of apply_heuristics() from heuristic_filter.py.
    Mutates each block dict in-place, adding Final_Pred.
    Only overrides CONTENT → INSTRUCTION, never the reverse.
    """
    for block in blocks:
        model_pred = block["label"]
        t_override, t_reason = False, ""
        v_override, v_reason = False, ""

        if model_pred == "CONTENT":
            t_override, t_reason = _text_is_instruction(block["tokens"])
            if not t_override:
                # bboxes are in 0-1000 space; image_rgb coordinates are pixels.
                # Scale back to pixel coords for the CV crop.
                img_w, img_h = image_rgb.size
                x_min_px = int(block["x_min"] * img_w / 1000)
                y_min_px = int(block["y_min"] * img_h / 1000)
                x_max_px = int(block["x_max"] * img_w / 1000)
                y_max_px = int(block["y_max"] * img_h / 1000)
                v_override, v_reason = _visual_is_instruction(
                    image_rgb, x_min_px, y_min_px, x_max_px, y_max_px
                )

        block["text_override"]   = t_override
        block["text_reason"]     = t_reason
        block["visual_override"] = v_override
        block["visual_reason"]   = v_reason
        block["final_label"]     = "INSTRUCTION" if (t_override or v_override) else model_pred

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Excalidraw element builder
# ─────────────────────────────────────────────────────────────────────────────
def _make_excalidraw_elements(block: dict, index: int) -> list[dict]:
    """Convert one filtered block into an Excalidraw rectangle + text element pair.

    The rectangle is coloured by label; the text element shows the merged token
    string so the actual textbook content is visible inside each block.
    The text element width matches the rectangle so Excalidraw wraps the text
    naturally, and autoResize is disabled to preserve the box dimensions.
    """
    rect_id     = str(uuid.uuid4())[:21]
    text_id     = str(uuid.uuid4())[:21]
    final_label = block["final_label"]
    color       = LABEL_COLOR.get(final_label, DEFAULT_COLOR)
    ts          = int(time.time() * 1000)
    content     = block["tokens"]
    padding     = 8          # horizontal inset so text does not touch the border
    font_size   = 14
    line_height = 1.25
    text_w      = max(block["width"] - padding * 2, 40)

    def seed(): return int(uuid.uuid4().int % 2**31)

    rect = {
        "id": rect_id, "type": "rectangle",
        "x": block["x"], "y": block["y"],
        "width": block["width"], "height": block["height"],
        "angle": 0,
        "strokeColor": "#1e1e1e", "backgroundColor": color,
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None,
        "index": f"a{index}",
        "roundness": {"type": 3},
        "seed": seed(), "version": 1, "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": [{"type": "text", "id": text_id}],
        "updated": ts, "link": None, "locked": False,
    }

    text = {
        "id": text_id, "type": "text",
        "x": block["x"] + padding,
        "y": block["y"] + padding,
        "width": text_w, "height": block["height"] - padding * 2,
        "angle": 0,
        "strokeColor": "#1e1e1e", "backgroundColor": color,
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None,
        "index": f"a{index}V",
        "roundness": None,
        "seed": seed(), "version": 1, "versionNonce": seed(),
        "isDeleted": False, "boundElements": None,
        "updated": ts, "link": None, "locked": False,
        "text": content, "fontSize": font_size, "fontFamily": 5,
        "textAlign": "left", "verticalAlign": "top",
        "containerId": rect_id,
        "originalText": content, "autoResize": False, "lineHeight": line_height,
    }
    return [rect, text]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — block consolidation  (controlled by ENABLE_BLOCK_CONSOLIDATION)
# ─────────────────────────────────────────────────────────────────────────────
def _consolidate_blocks(
    blocks: list[dict],
    instruction_y_gap: float = INSTRUCTION_MERGE_Y_GAP,
) -> list[dict]:
    """
    Reduces the number of Excalidraw blocks by applying two rules in a single
    pass over the list (which is already in top-to-bottom reading order):

    CONTENT rule — unconditional greedy merge:
        Consecutive CONTENT blocks are always merged into one, regardless of
        the vertical gap between them.  The run is broken only when an
        INSTRUCTION block appears between them.

    INSTRUCTION rule — proximity-gated merge:
        Consecutive INSTRUCTION blocks are merged only when the vertical gap
        between the bottom edge of the previous block and the top edge of the
        next block is <= instruction_y_gap (in 0-1000 normalised units).
        A larger gap means a visible separator on the page (e.g. a content
        paragraph in between that the model labelled INSTRUCTION), so they
        are kept separate.

    Merged blocks get a bounding box that is the union of their members and
    their token strings are joined with a space.
    """

    def _merge_run(run: list[dict]) -> dict:
        """Union-bbox merge of a non-empty list of blocks."""
        merged = dict(run[0])
        merged["tokens"]  = " ".join(b["tokens"] for b in run)
        merged["x_min"]   = min(b["x_min"]  for b in run)
        merged["y_min"]   = min(b["y_min"]  for b in run)
        merged["x_max"]   = max(b["x_max"]  for b in run)
        merged["y_max"]   = max(b["y_max"]  for b in run)
        merged["x"]       = float(merged["x_min"])
        merged["y"]       = float(merged["y_min"])
        merged["width"]   = float(max(merged["x_max"] - merged["x_min"], 40))
        merged["height"]  = float(max(merged["y_max"] - merged["y_min"] + 10, 20))
        return merged

    result: list[dict] = []
    run:    list[dict] = []
    run_label: str | None = None

    for block in blocks:
        label = block["final_label"]

        if run_label is None:
            run       = [block]
            run_label = label

        elif label == run_label == "CONTENT":
            # CONTENT -> CONTENT: always extend the run
            run.append(block)

        elif label == run_label == "INSTRUCTION":
            # INSTRUCTION -> INSTRUCTION: extend only if close enough
            gap = block["y_min"] - run[-1]["y_max"]
            if gap <= instruction_y_gap:
                run.append(block)
            else:
                result.append(_merge_run(run))
                run = [block]

        else:
            # Label changed — flush current run, start a new one
            result.append(_merge_run(run))
            run       = [block]
            run_label = label

    if run:
        result.append(_merge_run(run))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# postprocess() — orchestrates all four stages
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(model_output: dict) -> list[dict]:
    """
    Stage 1 — majority voting      (baseline notebook)
    Stage 2 — block merging        (merge_token_blocks.py)
    Stage 3 — heuristic filtering  (heuristic_filter.py)
    Stage 4 — block consolidation  (ENABLE_BLOCK_CONSOLIDATION flag)
    -> Excalidraw element pairs
    """
    tokens    = model_output["tokens"]
    bboxes    = model_output["bboxes"]
    label_ids = model_output["label_ids"]
    id2label  = model_output["id2label"]
    image_rgb = model_output.get("image")

    if not tokens:
        return []

    # Stage 1 — majority voting
    corrected_ids   = _apply_majority_voting(label_ids, bboxes)
    corrected_names = [id2label.get(lid, "CONTENT") for lid in corrected_ids]

    # Stage 2 — merge into spatial blocks
    blocks = _merge_token_blocks(tokens, bboxes, corrected_names)

    # Stage 3 — heuristic filtering (text + visual)
    if image_rgb is not None:
        blocks = _apply_heuristics(blocks, image_rgb)
    else:
        for b in blocks:
            b["final_label"] = b["label"]

    # Stage 4 — block consolidation (optional)
    if ENABLE_BLOCK_CONSOLIDATION:
        blocks = _consolidate_blocks(blocks)

    # Build Excalidraw elements
    elements = []
    for i, block in enumerate(blocks):
        elements.extend(_make_excalidraw_elements(block, i))
    return elements


# ══════════════════════════════════════════════════════════════════════════════
#  Excalidraw JSON assembler
# ══════════════════════════════════════════════════════════════════════════════
def build_excalidraw_json(elements: list[dict]) -> str:
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": 20,
            "gridStep": 5,
            "gridModeEnabled": False,
            "viewBackgroundColor": "#ffffff",
            "lockedMultiSelections": {},
        },
        "files": {},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("📖 ESL Textbook → Excalidraw")
st.caption("Upload a textbook page (PNG) and download a ready-to-import Excalidraw diagram.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_dir = st.text_input(
        "Model directory",
        value=DEFAULT_MODEL_DIR,
        help="Path to the folder containing config.json, model.safetensors, etc.",
    )
    ocr_lang = st.text_input(
        "Tesseract language",
        value=DEFAULT_TESSERACT_LANG,
        help="e.g. 'eng', 'ukr+eng'",
    )
    st.divider()
    st.markdown("**Label colours**\n- 🟥 `INSTRUCTION`\n- 🟩 `CONTENT`")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a textbook page",
    type=["png"],
    help="Only PNG files are accepted.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.expander("Preview", expanded=True):
        st.image(image, width='stretch')

    st.divider()

    if st.button("▶ Generate Excalidraw diagram", type="primary", width='stretch'):
        try:
            with st.status("Processing…", expanded=True) as status:
                st.write("🔍 Running OCR and cleaning tokens…")
                preprocessed = preprocess(image, lang=ocr_lang)
                st.write(f"   → {len(preprocessed['tokens'])} tokens extracted")

                st.write("🤖 Running LayoutLMv3 inference…")
                model_output = run_model(preprocessed, model_dir=model_dir)
                label_counts = Counter(model_output["label_names"])
                st.write(f"   → predictions: {dict(label_counts)}")

                st.write("🔧 Applying majority voting and merging blocks…")
                elements = postprocess(model_output)
                st.write(f"   → {len(elements) // 2} Excalidraw blocks created")

                st.write("📦 Assembling Excalidraw JSON…")
                excalidraw_json = build_excalidraw_json(elements)

                status.update(label="Done!", state="complete", expanded=False)

            st.success("Diagram ready — click below to download.")

            filename = uploaded_file.name.rsplit(".", 1)[0] + ".excalidraw"
            st.download_button(
                label="⬇ Download .excalidraw file",
                data=excalidraw_json,
                file_name=filename,
                mime="application/json",
                type="primary",
                width='stretch',
            )

        except FileNotFoundError as e:
            st.error(f"**Model not found.**\n\n{e}\n\nCheck the model directory in the sidebar.")
        except Exception as e:
            st.error(f"**Error during processing:** {e}")
            raise