"""
merge_token_blocks.py
---------------------
Merges individual token predictions into coherent text blocks.

Merging strategy:
1. Tokens are grouped per image page first.
2. Within each page, tokens that are close in Y (same line) and X (horizontally
   adjacent) are clustered into blocks using a two-pass approach:
     - Pass 1: line grouping — tokens whose Y-centres are within `y_thresh`
               of each other are considered on the same line.
     - Pass 2: column grouping — within a line, tokens are split into separate
               blocks when the horizontal gap between consecutive tokens exceeds
               `x_gap_thresh`.
3. The final label of each block is determined by majority vote across all
   token `Pred` values in that block. Ties are broken by the label with the
   higher total confidence sum.

Output CSV columns:
  Block_ID, ImagePath, Tokens, Pred, Token_Count,
  Confidence_Mean, x_min, y_min, x_max, y_max, Bbox
"""

import ast
import pandas as pd
from collections import Counter

# ── tuneable thresholds ────────────────────────────────────────────────────────
Y_THRESH    = 10   # pixels — tokens within this vertical distance share a line
X_GAP_THRESH = 30  # pixels — horizontal gap that starts a new block on the same line
# ──────────────────────────────────────────────────────────────────────────────


def parse_bbox(bbox_str: str) -> list[int]:
    """Convert '[x1, y1, x2, y2]' string to a list of ints."""
    return ast.literal_eval(bbox_str)


def majority_label(preds: list[str], confidences: list[float]) -> str:
    """Return the label that appears most often; break ties by confidence sum."""
    counts = Counter(preds)
    max_count = max(counts.values())
    candidates = [label for label, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    # tie-break: highest total confidence
    conf_sums = {
        label: sum(c for p, c in zip(preds, confidences) if p == label)
        for label in candidates
    }
    return max(conf_sums, key=conf_sums.get)


def merge_tokens(df: pd.DataFrame,
                 y_thresh: float = Y_THRESH,
                 x_gap_thresh: float = X_GAP_THRESH) -> pd.DataFrame:
    """
    Core merging logic. Works on one page at a time (caller filters by image).

    Returns a DataFrame where every row is one merged block.
    """
    # Parse bboxes into separate columns for convenience
    bboxes = df["Bbox"].apply(parse_bbox)
    df = df.copy()
    df["x1"] = bboxes.apply(lambda b: b[0])
    df["y1"] = bboxes.apply(lambda b: b[1])
    df["x2"] = bboxes.apply(lambda b: b[2])
    df["y2"] = bboxes.apply(lambda b: b[3])

    # Sort by top-left reading order: top-to-bottom, left-to-right
    df = df.sort_values(["y", "x"]).reset_index(drop=True)

    blocks = []

    # ── Pass 1: group into lines by Y proximity ────────────────────────────
    line_groups: list[list[int]] = []   # list of row-index lists
    current_line: list[int] = []
    current_y: float | None = None

    for idx, row in df.iterrows():
        if current_y is None or abs(row["y"] - current_y) <= y_thresh:
            current_line.append(idx)
            # update running mean Y for the line
            current_y = df.loc[current_line, "y"].mean()
        else:
            line_groups.append(current_line)
            current_line = [idx]
            current_y = row["y"]

    if current_line:
        line_groups.append(current_line)

    # ── Pass 2: split lines into blocks by X gap ───────────────────────────
    for line_idxs in line_groups:
        line = df.loc[line_idxs].sort_values("x")
        current_block: list[int] = []

        for i, (idx, row) in enumerate(line.iterrows()):
            if not current_block:
                current_block.append(idx)
            else:
                prev = line.loc[current_block[-1]]
                gap = row["x1"] - prev["x2"]   # gap between token bboxes
                if gap <= x_gap_thresh:
                    current_block.append(idx)
                else:
                    blocks.append(current_block)
                    current_block = [idx]

        if current_block:
            blocks.append(current_block)

    # ── Build output rows ──────────────────────────────────────────────────
    records = []
    for block_idxs in blocks:
        sub = df.loc[block_idxs]
        preds       = sub["Pred"].tolist()
        confidences = sub["Confidence"].tolist()
        label       = majority_label(preds, confidences)

        x_min = int(sub["x1"].min())
        y_min = int(sub["y1"].min())
        x_max = int(sub["x2"].max())
        y_max = int(sub["y2"].max())

        records.append({
            "ImagePath":       sub["ImagePath"].iloc[0],
            "Tokens":          " ".join(sub["Token"].tolist()),
            "Pred":            label,
            "Token_Count":     len(sub),
            "Confidence_Mean": round(sub["Confidence"].mean(), 6),
            "x_min": x_min, "y_min": y_min,
            "x_max": x_max, "y_max": y_max,
            "Bbox":            f"[{x_min}, {y_min}, {x_max}, {y_max}]",
            # label breakdown for transparency
            "Label_Breakdown": dict(Counter(preds)),
        })

    return pd.DataFrame(records)


def main(input_path: str = "token_predictions.csv",
         output_path: str = "merged_blocks.csv"):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} tokens from {input_path}")
    print(f"Pages  : {sorted(df['ImagePath'].unique())}")
    print(f"Labels : {sorted(df['Pred'].unique())}\n")

    all_blocks = []
    for image_path, page_df in df.groupby("ImagePath"):
        page_blocks = merge_tokens(page_df)
        all_blocks.append(page_blocks)
        print(f"  {image_path.split('/')[-1]}: "
              f"{len(page_df)} tokens → {len(page_blocks)} blocks")

    result = pd.concat(all_blocks, ignore_index=True)
    result.insert(0, "Block_ID", range(1, len(result) + 1))

    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} blocks to: {output_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Block label distribution ──")
    print(result["Pred"].value_counts().to_string())

    print("\n── Sample output (first 5 rows) ──")
    print(result[["Block_ID", "Tokens", "Pred", "Token_Count",
                  "Confidence_Mean", "Bbox"]].head(5).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge token-level predictions into text blocks."
    )
    parser.add_argument(
        "input", nargs="?",
        default="token_predictions.csv",
        help="Path to the input CSV (default: token_predictions.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged_blocks.csv",
        help="Path for the output CSV (default: merged_blocks.csv)"
    )
    parser.add_argument(
        "--y-thresh", type=float, default=Y_THRESH,
        help=f"Max Y-distance to group tokens on the same line (default: {Y_THRESH})"
    )
    parser.add_argument(
        "--x-gap", type=float, default=X_GAP_THRESH,
        help=f"Max X-gap to keep tokens in the same block (default: {X_GAP_THRESH})"
    )
    args = parser.parse_args()

    # Allow threshold overrides from CLI
    Y_THRESH     = args.y_thresh
    X_GAP_THRESH = args.x_gap

    main(args.input, args.output)