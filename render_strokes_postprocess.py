#!/usr/bin/env python3
"""
Post-process stroke overlay renderer for SketchVLM outputs.

Given a results folder containing:
  - item_00001.json
  - item_00001_orig.jpg/png (optional)
  - item_00001_grid.png/jpg (optional)

This script extracts <strokes> blocks from model_output_full (or model_output)
and renders them onto either the orig or grid image.

Supports:
  - Grid token points: 'x500y95'
  - Numeric points: (500,95) or 500,95
  - bottom-left or top-left origin interpretation
  - mapping from grid-image coordinates to orig-image coordinates using grid_config + cell_pixel_map

Example:
  python render_strokes_postprocess.py --results-dir results/mix_eval/20251221_210419 \
      --base grid --origin top-left --only "1,2,3"

"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import io

import cairosvg
from PIL import Image

# Use the exact SVG + Bezier utilities that collab_sketch_with_label.py uses,
# so postprocessed renders match your in-run renders.
import utils


STROKE_BLOCK_RE = re.compile(r"<s(\d+)>(.*?)</s\1>", re.S | re.I)
POINTS_BLOCK_RE = re.compile(r"<points>(.*?)</points>", re.S | re.I)
TEXT_BLOCK_RE = re.compile(r"<text([^>]*)>\s*(?:'([^']+)'|\"([^\"]+)\"|([^<]+))\s*</text>", re.S | re.I)

# token style: 'x12y34' (quotes optional)
TOKEN_RE = re.compile(r"x(\d+)\s*y(\d+)", re.I)

# numeric pairs: (12,34) or 12,34
NUMPAIR_RE = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?")


def _parse_only_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def _find_image(results_dir: Path, stem: str, preferred_exts=("png", "jpg", "jpeg", "webp")) -> Optional[Path]:
    for ext in preferred_exts:
        p = results_dir / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def _extract_strokes_xml(d: Dict) -> str:
    # Prefer the cleaned field first when available.
    for k in ("model_output", "model_output_full"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _svg_root_open(w: int, h: int) -> str:
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )


def _composite_svg_on_image(base_rgb: Image.Image, svg_text: str, out_png: Path):
    """Match collab_sketch_with_label.py compositing semantics."""
    over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode("utf-8")))).convert("RGBA")
    base = base_rgb.convert("RGBA")
    if over.size != base.size:
        inner = re.sub(r'^.*?<svg[^>]*>|</svg>\s*$', '', svg_text, flags=re.S)
        svg_text = (
            f'<svg width="{base.size[0]}" height="{base.size[1]}" '
            f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
        )
        over = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode("utf-8")))).convert("RGBA")
    Image.alpha_composite(base, over).convert("RGB").save(out_png)


def _get_drawable_bbox_from_cell_map(cell_pixel_map: Dict[str, List[float]], cell_size: float) -> Tuple[float, float, float, float]:
    """
    Estimate the drawable (image) rectangle within a grid image from cell centers.
    Returns (left, top, right, bottom) in grid-image pixel coordinates.
    """
    xs = []
    ys = []
    for _, (cx, cy) in cell_pixel_map.items():
        xs.append(float(cx))
        ys.append(float(cy))
    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # centers -> boundaries
    half = float(cell_size) / 2.0
    left = min_x - half
    right = max_x + half
    top = min_y - half
    bottom = max_y + half
    return (left, top, right, bottom)


def _token_to_cell_key(x: int, y: int) -> str:
    return f"x{x}y{y}"


def _flip_y_if_needed(y: int, res_y: int, origin: str) -> int:
    # token coordinates are 1..res_y in "grid token space"
    if origin == "top-left":
        # top-left means y=1 at top. If tokens were produced with bottom-left,
        # convert by flipping. (If tokens were already top-left, user should
        # pass origin=top-left and the json's cell_pixel_map would have y=1 at top,
        # in which case flipping would be wrong. But user explicitly asked for a switch,
        # so we implement flipping as the conversion.)
        return res_y - y + 1
    return y  # bottom-left


def _map_points_to_pixels(
    pts_text: str,
    base_mode: str,
    origin: str,
    res_x: int,
    res_y: int,
    base_size: Tuple[int, int],
    grid_size_px: Optional[Tuple[int, int]] = None,
    cell_pixel_map: Optional[Dict[str, List[float]]] = None,
    cell_size: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], str]:
    """
    Map points in pts_text to pixel coordinates on the chosen base image.

    Returns (points_px, kind) where kind is 'token' or 'numeric'.
    """
    Wb, Hb = base_size

    # 1) token points 'xNyM'
    tokens = [(int(a), int(b)) for a, b in TOKEN_RE.findall(pts_text)]
    if tokens:
        # If base is grid image: use cell_pixel_map directly when available.
        if base_mode == "grid" and cell_pixel_map:
            out = []
            for x, y in tokens:
                y2 = _flip_y_if_needed(y, res_y, origin)
                key = _token_to_cell_key(x, y2)
                if key not in cell_pixel_map:
                    raise KeyError(key)
                cx, cy = cell_pixel_map[key]
                out.append((float(cx), float(cy)))
            return out, "token"

        # If base is orig image (no grid), mimic collab_sketch_with_label.py's
        # "annotated_nogrid" behavior: the raw/orig image was placed onto the grid
        # canvas with a fixed offset (no scaling). So we do a pure translation from
        # grid-canvas pixel space -> orig pixel space.
        if base_mode == "orig" and cell_pixel_map and cell_size and grid_size_px:
            grid_w, grid_h = grid_size_px
            # In the main renderer, the raw image is placed at:
            #   x_offset = cell_size
            #   y_offset = grid_h - cell_size - raw_h
            x_off = float(cell_size)
            y_off = float(grid_h) - float(cell_size) - float(Hb)

            out = []
            for x, y in tokens:
                y2 = _flip_y_if_needed(y, res_y, origin)
                key = _token_to_cell_key(x, y2)
                if key not in cell_pixel_map:
                    raise KeyError(key)
                gx, gy = cell_pixel_map[key]  # grid-canvas pixel coords (center of cell)
                ox = float(gx) - x_off
                oy = float(gy) - y_off
                out.append((ox, oy))
            return out, "token"

        # If no mapping info exists, interpret tokens as normalized to res_x/res_y and scale to base image
        out = []
        for x, y in tokens:
            # tokens assumed 1..res
            xf = (x / float(res_x)) * (Wb - 1)
            if origin == "bottom-left":
                yf = ((res_y - y) / float(res_y)) * (Hb - 1)
            else:
                yf = (y / float(res_y)) * (Hb - 1)
            out.append((xf, yf))
        return out, "token"

    # 2) numeric pairs
    nums = [(float(a), float(b)) for a, b in NUMPAIR_RE.findall(pts_text)]
    if nums:
        out = []
        for x, y in nums:
            xf = (x / float(res_x)) * (Wb - 1)
            if origin == "bottom-left":
                yf = ((res_y - y) / float(res_y)) * (Hb - 1)
            else:
                yf = (y / float(res_y)) * (Hb - 1)
            out.append((xf, yf))
        return out, "numeric"

    return [], "none"


def _iter_stroke_blocks(strokes_xml: str) -> List[Tuple[int, str]]:
    blocks = []
    for m in STROKE_BLOCK_RE.finditer(strokes_xml):
        n = int(m.group(1))
        body = m.group(2)
        blocks.append((n, body))
    return blocks


def _extract_strokes_section(xml: str) -> str:
    m = re.search(r"<strokes>(.*?)</strokes>", xml, re.S | re.I)
    return m.group(1) if m else xml


def _parse_text_style(attrs: str) -> Tuple[Optional[float], Optional[str]]:
    # attrs like: size="4.0" color="black"
    size = None
    color = None
    m_size = re.search(r'size\s*=\s*"([^"]+)"', attrs)
    if m_size:
        try:
            size = float(m_size.group(1))
        except Exception:
            size = None
    m_color = re.search(r'color\s*=\s*"([^"]+)"', attrs)
    if m_color:
        color = m_color.group(1).strip()
    return size, color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Folder containing item_*.json and images")
    ap.add_argument("--base", choices=["orig", "grid"], default="grid", help="Render strokes onto which base image")
    ap.add_argument("--origin", choices=["bottom-left", "top-left"], default="bottom-left",
                    help="How to interpret model coordinates. Use the same origin the prompt used.")
    ap.add_argument("--res-x", type=int, default=None, help="Model coordinate max X (defaults to json grid_config.res_x if present)")
    ap.add_argument("--res-y", type=int, default=None, help="Model coordinate max Y (defaults to json grid_config.res_y if present)")
    ap.add_argument("--only", type=str, default=None, help="Comma-separated item indices to process (e.g., 1,2,3)")
    ap.add_argument("--out-dir", type=str, default=None, help="Where to write outputs (default: results-dir)")
    ap.add_argument("--stroke-width", type=float, default=0.0, help="Override stroke width in px (0 = infer from cell_size)")
    ap.add_argument("--cell-size", type=float, default=15.0,
                    help="(Mimic collab_sketch bug) Base cell_size used for stroke/text sizing when rendering grid-token outputs. Default 15.")
    ap.add_argument("--use-json-cell-size", action="store_true",
                    help="Use grid_config.cell_size from each JSON for sizing (disables mimic-bug sizing).")
    ap.add_argument("--alt-colors", action="store_true", help="Alternate green/pink like colab mode (default: all green)")
    ap.add_argument("--save-svg", action="store_true", help="Also save an SVG alongside the PNG")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    only = _parse_only_list(args.only)

    json_paths = sorted(results_dir.glob("item_*.json"))
    if only is not None:
        keep = set(only)
        filtered = []
        for p in json_paths:
            m = re.search(r"item_(\d+)\.json$", p.name)
            if m and int(m.group(1)) in keep:
                filtered.append(p)
        json_paths = filtered

    if not json_paths:
        raise SystemExit("No item_*.json found (or none matched --only).")

    # Match collab_sketch_with_label.py defaults
    default_stroke_color = "green"
    alt_colors = ["green", "pink"]
    text_font_family = "Arial"
    text_font_scale = 3.2

    for jp in json_paths:
        d = json.loads(jp.read_text(encoding="utf-8"))

        item_id_match = re.search(r"item_(\d+)\.json$", jp.name)
        item_id = int(item_id_match.group(1)) if item_id_match else int(d.get("index", 0))

        grid_cfg = d.get("grid_config") or {}
        cell_pixel_map = d.get("cell_pixel_map") or None

        # coordinate system size
        res_x = args.res_x if args.res_x is not None else int(grid_cfg.get("res_x") or 1000)
        res_y = args.res_y if args.res_y is not None else int(grid_cfg.get("res_y") or 1000)

        # base image
        base_img_path = _find_image(results_dir, f"item_{item_id:05d}_orig") if args.base == "orig" else _find_image(results_dir, f"item_{item_id:05d}_grid")
        if base_img_path is None:
            # fallback to paths stored in json (may be relative)
            key = "raw_image" if args.base == "orig" else "grid_image"
            maybe = d.get(key)
            if maybe:
                base_img_path = (results_dir / Path(maybe).name) if not Path(maybe).is_absolute() else Path(maybe)
        if base_img_path is None or not base_img_path.exists():
            print(f"[WARN] Missing base image for item {item_id:05d} ({args.base}). Skipping.")
            continue

        base_img = Image.open(base_img_path).convert("RGB")
        Wb, Hb = base_img.size

        strokes_xml_full = _extract_strokes_xml(d)
        strokes_xml = _extract_strokes_section(strokes_xml_full)

        # mapping info for token->pixel
        grid_size_px = None
        if "grid_size_px" in grid_cfg:
            try:
                grid_size_px = (int(grid_cfg["grid_size_px"][0]), int(grid_cfg["grid_size_px"][1]))
            except Exception:
                grid_size_px = None
        cell_size = float((grid_cfg.get("cell_size") or 0.0) if (grid_cfg and args.use_json_cell_size) else args.cell_size)

        # Infer stroke width like the in-run code: stroke_width = cell_size * 0.6
        # Note: for base=orig with grid-token runs, we mimic the in-run behavior which uses
        # a pure translation (no scaling), so stroke width should NOT be scaled.
        stroke_width = (float(args.stroke_width) if float(args.stroke_width) > 0 else (float(cell_size) * 0.6 if cell_size else 3.0))

        blocks = _iter_stroke_blocks(strokes_xml)
        if not blocks:
            print(f"[WARN] No <sN> blocks found in item {item_id:05d}.")
            continue

        # Build an SVG overlay that matches the in-run rendering (beziers, cap/join styles).
        svg_parts: List[str] = [_svg_root_open(Wb, Hb)]

        for s_idx, body in blocks:
            m_pts = POINTS_BLOCK_RE.search(body)
            if not m_pts:
                continue

            # parse optional id for stable group ids
            m_id = re.search(r"<id>(.*?)</id>", body, re.S | re.I)
            stroke_label = (m_id.group(1).strip() if m_id else f"s{s_idx}")
            stroke_label = re.sub(r"[^\w\-]", "_", stroke_label)

            pts_text = m_pts.group(1)
            pts_px, _kind = _map_points_to_pixels(
                pts_text=pts_text,
                base_mode=args.base,
                origin=args.origin,
                res_x=res_x,
                res_y=res_y,
                base_size=(Wb, Hb),
                grid_size_px=grid_size_px,
                cell_pixel_map=cell_pixel_map,
                cell_size=cell_size if cell_size > 0 else None,
            )
            if not pts_px:
                continue

            stroke_color = (
                alt_colors[(s_idx - 1) % 2] if args.alt_colors else default_stroke_color
            )

            # text stroke?
            m_txt = TEXT_BLOCK_RE.search(body)
            if m_txt:
                attrs = m_txt.group(1) or ""
                text_raw = (m_txt.group(2) or m_txt.group(3) or m_txt.group(4) or "").strip()
                text_val = html.escape(text_raw)
                size_override, color_override = _parse_text_style(attrs)
                # Match collab_sketch_with_label.py behavior:
                # - default font size is cell_size * text_font_scale
                # - if <text size="k"> is present, interpret k as a MULTIPLIER of cell_size (not pixels)
                # - clamp to a sane range like the in-run code.
                if cell_size:
                    default_px = float(cell_size) * float(text_font_scale)
                    if size_override is not None:
                        # interpret as multiplier of cell_size
                        mult = float(size_override)
                        # clamp multipliers similarly to the main code
                        if mult < 0.8:
                            mult = 0.8
                        if mult > 6.0:
                            mult = 6.0
                        base_px = float(cell_size) * mult
                    else:
                        base_px = float(default_px)
                else:
                    base_px = float(size_override) if size_override is not None else 14.0
                font_px = int(round(base_px))
                fill = color_override or stroke_color or "black"
                x0, y0 = pts_px[0]
                svg_parts.append(
                    f'<g id="{stroke_label}_s{s_idx}">'
                    f'<text x="{x0:.1f}" y="{y0:.1f}" text-anchor="middle" dominant-baseline="central" '
                    f'font-family="{text_font_family}" font-size="{font_px}" fill="{fill}">{text_val}</text>'
                    f'</g>'
                )
                continue

            # t-values (optional)
            m_t = re.search(r"<t_values>(.*?)</t_values>", body, re.S | re.I)
            t_values: List[float] = []
            if m_t:
                raw = m_t.group(1).strip().strip("[]")
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                try:
                    t_values = [float(p) for p in parts]
                except Exception:
                    t_values = []
            n = len(pts_px)
            if (not t_values) or (len(t_values) != n):
                t_values = [i / (n - 1) if n > 1 else 0.0 for i in range(n)]

            sampled_points = [[float(x), float(y)] for (x, y) in pts_px]
            cps = utils.estimate_bezier_control_points(sampled_points, t_values)
            svg_parts.append(
                utils.format_svg_single_stroke(
                    cps,
                    dim=(Wb, Hb),
                    stroke_width=stroke_width,
                    stroke_counter=s_idx,
                    group_id=stroke_label,
                    stroke_color=stroke_color,
                )
            )

        svg_parts.append("</svg>")
        overlay_svg = "\n".join(svg_parts)

        out_png = out_dir / f"item_{item_id:05d}_annotated_post_{args.base}.png"
        _composite_svg_on_image(base_img, overlay_svg, out_png)
        print(f"[OK] Wrote {out_png}")

        if args.save_svg:
            out_svg = out_dir / f"item_{item_id:05d}_annotated_post_{args.base}.svg"
            out_svg.write_text(overlay_svg, encoding="utf-8")
            print(f"[OK] Wrote {out_svg}")


if __name__ == "__main__":
    main()


'''

Render onto the grid image (easy path, uses cell_pixel_map directly):

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base grid \
  --origin bottom-left \
  --only "1,2,3"
  

Render onto the orig image (grid→orig mapping happens automatically if cell_pixel_map exists):

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base orig \
  --origin bottom-left \
  --only "1,2,3"


If you want SVG too:

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base orig \
  --origin bottom-left \
  --only "1,2,3" \
  --save-svg


If your no-grid run used res_x=res_y=1000 coords:

python render_strokes_postprocess.py \
  --results-dir results/mix_eval/20251221_210419 \
  --base orig \
  --origin top-left \
  --res-x 1000 --res-y 1000


'''
