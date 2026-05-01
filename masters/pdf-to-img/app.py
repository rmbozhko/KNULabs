import streamlit as st
import json
import uuid
import time
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESL Textbook → Excalidraw",
    page_icon="📖",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { max-width: 700px; padding-top: 2.5rem; }
        .stAlert { border-radius: 8px; }
        div[data-testid="stFileUploader"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER — Pre-processing
#  Replace this function with your OCR + data-formatting logic.
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(image: Image.Image) -> dict:
    """
    TODO: Implement pre-processing pipeline.

    Steps to implement:
      1. OCR  – extract raw text / layout from `image`
      2. Parse – convert OCR output into the structured format the model expects
                 (e.g. list of exercise blocks, instruction strings, content items)

    Args:
        image: PIL Image object of the uploaded textbook page.

    Returns:
        A dict in whatever intermediate format your model expects.
        The stub returns a minimal example so the rest of the pipeline
        can run end-to-end during development.
    """
    # ── STUB ──────────────────────────────────────────────────────────────────
    return {
        "blocks": [
            {"type": "instruction", "text": "Instruction (OCR placeholder)"},
            {"type": "content",     "text": "Content (OCR placeholder)"},
        ]
    }
    # ─────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER — Model call
#  Replace with your actual model inference.
# ══════════════════════════════════════════════════════════════════════════════
def run_model(preprocessed: dict) -> dict:
    """
    TODO: Call your model here.

    Args:
        preprocessed: Output of preprocess().

    Returns:
        Raw model output (structure depends on your model).
        The stub echoes the input so post-processing has something to work with.
    """
    # ── STUB ──────────────────────────────────────────────────────────────────
    return preprocessed
    # ─────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER — Post-processing
#  Replace this function with your heuristics / cleanup logic.
# ══════════════════════════════════════════════════════════════════════════════
def postprocess(model_output: dict) -> list[dict]:
    """
    TODO: Implement post-processing heuristics.

    Steps to implement:
      1. Validate / clean model output
      2. Apply domain heuristics (e.g. merge split boxes, infer missing labels)
      3. Return a list of Excalidraw element dicts ready for the final JSON

    Args:
        model_output: Raw output returned by run_model().

    Returns:
        List of Excalidraw element dicts.
        The stub converts the placeholder blocks into two coloured rectangles
        matching the sample .excalidraw file so the download works out-of-the-box.
    """
    # ── STUB ──────────────────────────────────────────────────────────────────
    COLOR_MAP = {
        "instruction": "#ffc9c9",
        "content":     "#b2f2bb",
    }
    elements = []
    y_cursor = 100

    for block in model_output.get("blocks", []):
        rect_id = str(uuid.uuid4())[:21]
        text_id = str(uuid.uuid4())[:21]
        color   = COLOR_MAP.get(block["type"], "#ffffff")
        height  = 132

        rect = {
            "id": rect_id, "type": "rectangle",
            "x": 496, "y": y_cursor,
            "width": 396, "height": height, "angle": 0,
            "strokeColor": "#1e1e1e", "backgroundColor": color,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100,
            "groupIds": [], "frameId": None,
            "index": f"a{len(elements)}",
            "roundness": {"type": 3},
            "seed": int(uuid.uuid4().int % 2**31),
            "version": 1, "versionNonce": int(uuid.uuid4().int % 2**31),
            "isDeleted": False,
            "boundElements": [{"type": "text", "id": text_id}],
            "updated": int(time.time() * 1000),
            "link": None, "locked": False,
        }

        label = block.get("text", block["type"].capitalize())
        text = {
            "id": text_id, "type": "text",
            "x": 496 + 396 / 2 - 70, "y": y_cursor + height / 2 - 12.5,
            "width": 140, "height": 25, "angle": 0,
            "strokeColor": "#1e1e1e", "backgroundColor": color,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 1, "opacity": 100,
            "groupIds": [], "frameId": None,
            "index": f"a{len(elements)}V",
            "roundness": None,
            "seed": int(uuid.uuid4().int % 2**31),
            "version": 1, "versionNonce": int(uuid.uuid4().int % 2**31),
            "isDeleted": False, "boundElements": None,
            "updated": int(time.time() * 1000),
            "link": None, "locked": False,
            "text": label, "fontSize": 20, "fontFamily": 5,
            "textAlign": "center", "verticalAlign": "middle",
            "containerId": rect_id,
            "originalText": label, "autoResize": True, "lineHeight": 1.25,
        }

        elements.extend([rect, text])
        y_cursor += height + 40

    return elements
    # ─────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Excalidraw JSON assembler  (no changes needed here)
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
st.caption("Upload a textbook page and download a ready-to-import Excalidraw diagram.")

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
        st.image(image, width=True)

    st.divider()

    if st.button("▶ Generate Excalidraw diagram", type="primary", width=True):
        with st.status("Processing…", expanded=True) as status:
            st.write("🔍 Pre-processing (OCR + formatting)…")
            preprocessed = preprocess(image)

            st.write("🤖 Running model…")
            model_output = run_model(preprocessed)

            st.write("🔧 Post-processing (heuristics)…")
            elements = postprocess(model_output)

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
            width=True,
        )