#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess VPCT no-grid outputs from collab_sketch_with_label.py

Goal:
- Read each item_XXXXX.json in a results folder (results/mix_eval/...)
- Parse the model's raw text field (prefers "model_raw_text", falls back to "model_output_full")
  to extract:
    1) a polyline path near the end as a SEQUENCE OF POINTS:
         (x,y) -> (x,y) -> (x,y) ...
       (commas/newlines/arrows all allowed between points)
    2) the final answer formatted as \boxed{1|2|3}
- Render the extracted path as colored line segments on top of the ORIGINAL image.
  (Color cycle: red -> orange -> yellow -> green -> blue -> purple -> repeat)
- Write updated JSONs with a new "answer" field (string "1"/"2"/"3" or null)
  and overwrite or write new annotated images.

Coordinate system:
- Input x,y are in [0,1000], origin at top-left.
- Mapped to pixel coords: px = x/1000*(W-1), py = y/1000*(H-1)

Typical use:
  python postprocess_no_grid_paths.py --results-dir results/mix_eval/20251221_131636 --inplace
or:
  python postprocess_no_grid_paths.py --results-dir results/mix_eval/20251221_132640 --outdir results/mix_eval/20251221_132640_no_grid_post
  python postprocess_no_grid_paths.py --results-dir results/mix_eval/20251221_134211 --outdir results/mix_eval/20251221_134211_no_grid_post_6
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw


COLORS = [
    (255, 0, 0),       # red
    (255, 165, 0),     # orange
    (255, 255, 0),     # yellow
    (0, 128, 0),       # green
    (0, 0, 255),       # blue
    (128, 0, 128),     # purple
]

# Explicit segment pattern: (x1,y1)->(x2,y2)
SEG_RE = re.compile(
    r"\$?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\$?\s*->\s*\$?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\$?",
    re.IGNORECASE
)

# Coordinate pair, optional $...$ wrapper
COORD_RE = re.compile(
    r"\$?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\$?",
    re.IGNORECASE
)

BOX_RE = re.compile(r"\\boxed\s*\{\s*([123])\s*\}", re.IGNORECASE)

# Allowed "glue" between consecutive points in a path sequence
# (must contain ONLY delimiters/whitespace)
BETWEEN_OK_RE = re.compile(
    r"^\s*(?:,|\s|->|→|\\rightarrow|\r|\n)+\s*$",
    re.IGNORECASE
)


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def parse_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    matches = BOX_RE.findall(text)
    return matches[-1] if matches else None


def parse_segments(text: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse explicit segments like:
      (a,b)->(c,d), (c,d)->(e,f) ...
    Not the main path method anymore, but kept as a fallback.
    """
    if not text:
        return []
    segs = []
    for m in SEG_RE.finditer(text):
        x1, y1, x2, y2 = map(float, m.groups())
        segs.append((x1, y1, x2, y2))
    return segs


def points_to_segments(pts: List[Tuple[float, float]]) -> List[Tuple[float, float, float, float]]:
    segs: List[Tuple[float, float, float, float]] = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        segs.append((x1, y1, x2, y2))
    return segs


def extract_final_point_sequence(text: str) -> List[Tuple[float, float, float, float]]:
    """
    Robustly extract the FINAL point-sequence near the end of the output.

    Strategy:
    - Find the last \boxed{...} (if present). Use a window before it; else use end of text.
    - Find all coordinate pairs in that window.
    - Build runs where the text between consecutive pairs contains only delimiters:
        commas, arrows, whitespace/newlines, \\rightarrow, unicode arrows
    - Pick the LAST run with the MOST points (ties broken by later run).
    - Convert points to segments.

    This correctly handles:
      **Path:**
      (500, 80) -> (500, 260) -> (620, 280) -> ...
    and also cases with commas/newlines.
    """
    if not text:
        return []

    # Normalize arrow tokens for the "between" test
    t = text.replace("→", "->").replace("\\rightarrow", "->")

    # Choose an end position near the answer
    box_iter = list(BOX_RE.finditer(t))
    end_pos = box_iter[-1].start() if box_iter else len(t)

    # Window before the answer (tuneable)
    window_start = max(0, end_pos - 1500)
    window = t[window_start:end_pos]

    # Collect coordinate matches with spans
    matches = list(COORD_RE.finditer(window))
    if len(matches) < 2:
        return []

    # Convert match -> point + span
    pts = []
    for m in matches:
        x, y = m.group(1), m.group(2)
        pts.append((float(x), float(y), m.start(), m.end()))

    # Build runs of consecutive points where the "between text" is only delimiters
    best_run_pts: List[Tuple[float, float]] = []
    curr_run_pts: List[Tuple[float, float]] = [(pts[0][0], pts[0][1])]

    for i in range(1, len(pts)):
        prev_end = pts[i - 1][3]
        curr_start = pts[i][2]
        between = window[prev_end:curr_start]

        if BETWEEN_OK_RE.match(between):
            curr_run_pts.append((pts[i][0], pts[i][1]))
        else:
            # finalize current run
            if len(curr_run_pts) >= len(best_run_pts):
                best_run_pts = curr_run_pts
            curr_run_pts = [(pts[i][0], pts[i][1])]

    # finalize last run
    if len(curr_run_pts) >= len(best_run_pts):
        best_run_pts = curr_run_pts

    if len(best_run_pts) < 2:
        return []

    return points_to_segments(best_run_pts)


def map_to_px(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    x = clamp(x, 0.0, 1000.0)
    y = clamp(y, 0.0, 1000.0)
    px = (x / 1000.0) * (w - 1)
    py = (y / 1000.0) * (h - 1)
    return px, py


def draw_segments_on_image(
    img: Image.Image,
    segs: List[Tuple[float, float, float, float]],
    width: int = 4
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for i, (x1, y1, x2, y2) in enumerate(segs):
        c = COLORS[i % len(COLORS)]
        p1 = map_to_px(x1, y1, w, h)
        p2 = map_to_px(x2, y2, w, h)
        draw.line([p1, p2], fill=c, width=width)

    return out


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Folder containing item_XXXXX.json and images.")
    ap.add_argument("--outdir", default=None, help="If set, write outputs to a new folder instead of in-place.")
    ap.add_argument("--inplace", action="store_true", help="Overwrite json + annotated images in the same folder.")
    ap.add_argument("--line-width", type=int, default=4)
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    if not res_dir.exists():
        raise SystemExit(f"results-dir not found: {res_dir}")

    if not args.inplace and not args.outdir:
        raise SystemExit("Choose one: --inplace or --outdir ...")

    out_dir = res_dir if args.inplace else Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(res_dir.glob("item_*.json"))
    if not json_files:
        raise SystemExit(f"No item_*.json found in {res_dir}")

    updated = 0
    for jp in json_files:
        obj = load_json(jp)

        raw_text = obj.get("model_raw_text")
        if raw_text is None:
            raw_text = obj.get("model_output_full") or ""

        # 1) Primary: extract the final point-sequence near the end
        segs = extract_final_point_sequence(raw_text)

        # 2) Fallback: explicit (x,y)->(x,y) segments anywhere
        if not segs:
            segs = parse_segments(raw_text)

        ans = parse_boxed_answer(raw_text)

        # Load original image to draw on:
        src_img_path = obj.get("source_image")
        if src_img_path and Path(src_img_path).exists():
            img_path = Path(src_img_path)
        else:
            img_path = Path(obj.get("raw_image", ""))

        if not img_path.exists():
            obj["answer"] = ans
            obj["num_segments"] = len(segs)
            save_json(out_dir / jp.name, obj)
            continue

        img = Image.open(img_path).convert("RGB")
        ann = draw_segments_on_image(img, segs, width=args.line_width)

        ann_name = Path(obj.get("annotated_image", "")).name
        if not ann_name:
            ann_name = jp.stem + "_annotated.png"
        ann_out = out_dir / ann_name
        ann.save(ann_out)

        obj["answer"] = ans
        obj["num_segments"] = len(segs)
        obj["annotated_image"] = str(ann_out).replace("\\", "/")

        save_json(out_dir / jp.name, obj)
        updated += 1

    print(f"Done. Updated {updated} items -> {out_dir}")


if __name__ == "__main__":
    main()
