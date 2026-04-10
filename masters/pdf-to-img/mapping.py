import json
from typing import List, Dict
import argparse


def load_labelstudio_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- utils ---
def convert_block_to_1000(block: Dict) -> List[int]:
    """
    Convert Label Studio % bbox → [0,1000]
    """
    x = block["x"]
    y = block["y"]
    w = block["width"]
    h = block["height"]

    x1 = int(x * 10)
    y1 = int(y * 10)
    x2 = int((x + w) * 10)
    y2 = int((y + h) * 10)

    return [x1, y1, x2, y2]


def compute_iou(boxA, boxB):
    """
    IoU between two boxes in [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(boxA_area + boxB_area - inter_area)


# --- main logic ---
def extract_blocks(task_json):
    annotations = task_json["annotations"]
    if not annotations:
        return []

    results = annotations[0]["result"]

    blocks = []
    for r in results:
        val = r["value"]
        label = val["rectanglelabels"][0]

        bbox = convert_block_to_1000(
            val
        )

        blocks.append({
            "bbox": bbox,
            "label": label
        })

    return blocks


def map_tokens_to_blocks(token_bboxes, blocks, iou_threshold=0.3):
    labels = []

    for t_bbox in token_bboxes:
        best_label = "O"
        best_iou = 0.0

        for block in blocks:
            iou = compute_iou(t_bbox, block["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_label = block["label"]

        if best_iou < iou_threshold:
            best_label = "O"

        labels.append(best_label)

    return labels


def process_task(task_json, iou_threshold=0.3):
    data = task_json["data"]

    bboxes = data["bboxes"]

    blocks = extract_blocks(task_json)

    labels = map_tokens_to_blocks(bboxes, blocks, iou_threshold)

    return {
        "tokens": data["tokens"],
        "bboxes": bboxes,
        "labels": labels
    }


# --- run ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to Label Studio JSON file")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for token-block mapping")
    parser.add_argument("--output", type=str, default="tokens_mapped_blocks.json", help="Output JSON file")
    args = parser.parse_args()
    path = args.path

    if not path.endswith(".json"):
        print("Please provide a valid Label Studio JSON file")
        exit(1) 

    task = load_labelstudio_json(path)

    for t in task:
        result = process_task(t, args.iou_threshold)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)