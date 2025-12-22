#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess VPCT no-grid outputs from collab_sketch_with_label.py

Goal:
- Read each item_XXXXX.json in a results folder (results/mix_eval/...)
- Parse the model's raw text field (prefers "model_raw_text", falls back to "model_output_full")
  to extract:
    1) a polyline path expressed as "(x,y)->(x,y), (x,y)->(x,y) ..."
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
  python postprocess_no_grid_paths.py --results-dir results/mix_eval/20251221_134211 --outdir results/mix_eval/20251221_134211_no_grid_post

"""

import argparse
import json
import os
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


SEG_RE = re.compile(
    r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*->\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE
)

BOX_RE = re.compile(r"\\boxed\s*\{\s*([123])\s*\}", re.IGNORECASE)


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def parse_segments(text: str) -> List[Tuple[float, float, float, float]]:
    if not text:
        return []
    segs = []
    for m in SEG_RE.finditer(text):
        x1, y1, x2, y2 = map(float, m.groups())
        segs.append((x1, y1, x2, y2))
    return segs


def parse_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = BOX_RE.search(text)
    if not m:
        return None
    return m.group(1)


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

        # Prefer model_raw_text, but keep compatibility with your existing schema
        raw_text = obj.get("model_raw_text")
        if raw_text is None:
            raw_text = obj.get("model_output_full") or ""

        segs = parse_segments(raw_text)
        ans = parse_boxed_answer(raw_text)

        # Load original image to draw on:
        # Prefer source_image path if it exists locally; otherwise fall back to saved raw_image.
        src_img_path = obj.get("source_image")
        if src_img_path and Path(src_img_path).exists():
            img_path = Path(src_img_path)
        else:
            img_path = Path(obj.get("raw_image", ""))

        if not img_path.exists():
            # Can't render, but still store answer/segment count
            obj["answer"] = ans
            obj["num_segments"] = len(segs)
            save_json(out_dir / jp.name, obj)
            continue

        img = Image.open(img_path).convert("RGB")
        ann = draw_segments_on_image(img, segs, width=args.line_width)

        # Determine annotated output filename
        ann_name = Path(obj.get("annotated_image", "")).name
        if not ann_name:
            ann_name = jp.stem + "_annotated.png"
        ann_out = out_dir / ann_name
        ann.save(ann_out)

        # Copy raw/orig/grid filenames forward as-is; just update annotated_image path to the new location
        # (keep Windows-style slashes out of the new path; use forward slashes for portability)
        obj["answer"] = ans
        obj["num_segments"] = len(segs)
        obj["annotated_image"] = str(ann_out).replace("\\", "/")

        save_json(out_dir / jp.name, obj)
        updated += 1

    print(f"Done. Updated {updated} items -> {out_dir}")


if __name__ == "__main__":
    main()
