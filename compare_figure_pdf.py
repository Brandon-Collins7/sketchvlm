#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_figure_pdf.py

Generate a *standalone PDF* comparison figure directly from Python (no LaTeX / pdflatex).

- Arbitrary number of columns: pass --col-dir multiple times with matching --col-title
- Choose which dataset rows to include: --rows "0,3,5,9" or "0-9,12"
- Includes GT + Pred + OK/X under each image (and optional [row] + filename under original)
- No whitespace between images in a row (gutter=0)
- Auto-paginates if rows don't fit on one page

Expected VPCT-style data (same as your earlier script):
- --gt-root contains:
    sim_<id>_initial.png
    sim_<id>_results.json   ({"finalBucket": 1|2|3})
- Each run dir contains item_*.json with "source_image" pointing to sim_*_initial.png
  and optional images:
    item_<idx>_annotated_color.png (preferred)
    item_<idx>_annotated.(png|jpg|jpeg|webp)
    item_<idx>_orig.(png|jpg|jpeg|webp)

Example:
python compare_figure_pdf.py \
  --gt-root vpct-1 \
  --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro (SketchVLM)" \
  --col-dir results/mix_eval/gpt5low_vpct_test --col-title "GPT-5 (low) (No grid)" \
  --rows "0,3,5,9" \
  --out-pdf fig_vpct_compare.pdf
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# reportlab (PDF)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors


SIM_IMG_RE = re.compile(r"sim_(\d+)_initial\.png$", re.I)


# -----------------------------
# parsing helpers
# -----------------------------

def to_int_123(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        iv = int(str(v).strip())
        return iv if iv in (1, 2, 3) else None
    except Exception:
        return None

def extract_last_int_token(text: str) -> Optional[int]:
    if not text:
        return None
    vals: List[int] = []
    for m in re.finditer(r"-?\d+", text):
        s, e = m.span()
        prev = text[s - 1] if s > 0 else ""
        prevprev = text[s - 2] if s > 1 else ""
        nxt = text[e] if e < len(text) else ""
        nxtnxt = text[e + 1] if e + 1 < len(text) else ""
        # reject decimals like 21.5
        if nxt == "." and nxtnxt.isdigit():
            continue
        if prev == "." and prevprev.isdigit():
            continue
        try:
            vals.append(int(m.group()))
        except Exception:
            pass
    return vals[-1] if vals else None

def extract_last_bucket_from_output(text: str) -> Optional[int]:
    iv = extract_last_int_token(text or "")
    return iv if iv in (1, 2, 3) else None

def parse_rows_spec(spec: Optional[str]) -> Optional[List[int]]:
    if not spec or not spec.strip():
        return None
    s = spec.replace(" ", ",").strip()
    out: List[int] = []
    seen = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit() and b.isdigit():
                lo = int(a); hi = int(b)
                step = 1 if hi >= lo else -1
                for x in range(lo, hi + step, step):
                    if x not in seen:
                        seen.add(x); out.append(x)
            continue
        if tok.isdigit():
            x = int(tok)
            if x not in seen:
                seen.add(x); out.append(x)
    return out


# -----------------------------
# data models / loaders
# -----------------------------

@dataclass
class RunExample:
    pred: Optional[int]
    img_path: Optional[Path]
    item_idx: Optional[str]

def load_gt(gt_root: Path) -> Tuple[List[str], Dict[str, int]]:
    gt_map: Dict[str, int] = {}
    for jf in sorted(gt_root.glob("sim_*_results.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        gt = to_int_123(j.get("finalBucket"))
        if gt is None:
            continue
        m = re.search(r"sim_(\d+)_results\.json$", jf.name, re.I)
        if not m:
            continue
        img = f"sim_{m.group(1)}_initial.png"
        gt_map[img] = gt

    def sort_key(name: str):
        m = SIM_IMG_RE.search(name)
        return (int(m.group(1)) if m else 10**9, name)

    ordered = sorted(gt_map.keys(), key=sort_key)
    return ordered, gt_map

def pick_best_run_image(run_dir: Path, item_idx: str) -> Optional[Path]:
    cand_color = run_dir / f"item_{item_idx}_annotated_color.png"
    if cand_color.exists():
        return cand_color
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = run_dir / f"item_{item_idx}_annotated{ext}"
        if cand.exists():
            return cand
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = run_dir / f"item_{item_idx}_orig{ext}"
        if cand.exists():
            return cand
    return None

def load_run_dir(run_dir: Path) -> Dict[str, RunExample]:
    out: Dict[str, RunExample] = {}
    for jf in sorted(run_dir.glob("item_*.json")):
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        src = str(j.get("source_image", "") or j.get("image", "") or j.get("file", "")).replace("\\", "/")
        basename = Path(src).name
        if not SIM_IMG_RE.search(basename):
            continue

        pred = to_int_123(j.get("answer"))

        mo_full_any = j.get("model_output_full") or j.get("model_out_full") or ""
        mo_full_text = mo_full_any if isinstance(mo_full_any, str) else str(mo_full_any or "")
        if pred is None and mo_full_text.strip():
            pred = to_int_123(mo_full_text) or extract_last_bucket_from_output(mo_full_text)

        item_idx = None
        m = re.match(r"item_(\d+)\.json$", jf.name, re.I)
        if m:
            item_idx = m.group(1)

        img_path = pick_best_run_image(run_dir, item_idx) if item_idx else None
        out[basename] = RunExample(pred=pred, img_path=img_path, item_idx=item_idx)
    return out


# -----------------------------
# PDF layout / drawing
# -----------------------------

def draw_image_fit(c: canvas.Canvas, img_path: Path, x: float, y: float, w: float, h: float) -> None:
    """
    Draw image into (x,y,w,h) preserving aspect ratio, centered in the box.
    reportlab uses bottom-left origin.
    """
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
        if iw <= 0 or ih <= 0:
            raise ValueError("bad image size")
    except Exception:
        # missing or unreadable
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, "missing")
        return

    # fit
    scale = min(w / iw, h / ih)
    dw = iw * scale
    dh = ih * scale
    dx = x + (w - dw) / 2
    dy = y + (h - dh) / 2

    c.drawImage(ImageReader(str(img_path)), dx, dy, width=dw, height=dh, preserveAspectRatio=True, mask='auto')

def draw_pred_label(
    c: canvas.Canvas,
    x_center: float,
    y: float,
    pred_str: str,
    ok: Optional[bool],
    font_size: int,
) -> None:
    # Plain ASCII mark to avoid missing-glyph boxes
    if ok is True:
        mark = "OK"
    elif ok is False:
        mark = "X"
    else:
        mark = ""

    text = f"Pred={pred_str}" if not mark else f"Pred={pred_str} {mark}"
    c.setFillColor(colors.black)
    c.setFont("Times-Roman", font_size)
    c.drawCentredString(x_center, y, text)


def make_pdf(
    out_pdf: Path,
    ordered_imgs: List[str],
    gt_map: Dict[str, int],
    gt_root: Path,
    cols: List[Tuple[str, Path, Dict[str, RunExample]]],  # (title, dir, map)
    picked_rows: List[int],
    include_original: bool = True,
    show_ids: bool = True,
    original_title: str = "Source Image",
    page: str = "letter",
    landscape_mode: bool = False,
    pt_per_inch: float = 55.0,
    margin_in: float = 0.35,
    header_h_in: float = 0.35,
    label_h_in: float = 0.26,
    row_gap_in: float = 0.10,
    font_size: int = 7,
    header_font_size: int = 9,
) -> None:
    pagesize = letter if page.lower() == "letter" else A4
    pw, ph = pagesize
    if landscape_mode:
        pw, ph = ph, pw

    inch = float(pt_per_inch)
    margin = margin_in * inch
    header_h = header_h_in * inch
    label_h = label_h_in * inch
    row_gap = row_gap_in * inch

    col_titles = ([original_title] if include_original else []) + [t for t, _, _ in cols]
    ncols = len(col_titles)

    usable_w = pw - 2 * margin
    cell_w = usable_w / ncols  # no gutters => no whitespace horizontally

    # We'll choose image height based on remaining height and rows per page, with auto pagination.
    # Define a target image height, then compute how many rows fit.
    target_img_h = 2.2 * inch  # adjustable if needed

    def rows_per_page(img_h: float) -> int:
        usable_h = ph - 2 * margin - header_h
        per_row = img_h + label_h + row_gap
        if per_row <= 0:
            return 1
        return max(1, int(usable_h // per_row))

    rpp = rows_per_page(target_img_h)

    c = canvas.Canvas(str(out_pdf), pagesize=(pw, ph))
    c.setTitle(out_pdf.stem)

    def draw_header():
        c.setFont("Times-Bold", header_font_size)
        y_top = ph - margin
        y = y_top - header_h
        for i, title in enumerate(col_titles):
            x0 = margin + i * cell_w
            c.drawCentredString(x0 + cell_w / 2, y, title)

    def draw_page(page_rows: List[int]):
        draw_header()
        y_cursor = ph - margin - header_h  # top of content area
        c.setFont("Times-Roman", font_size)


        for ridx in page_rows:
            img_name = ordered_imgs[ridx]
            gt = gt_map.get(img_name, None)
            gt_str = str(gt) if gt in (1, 2, 3) else "?"

            # Row box top/bottom
            img_box_h = target_img_h
            row_total_h = img_box_h + label_h
            y_top = y_cursor
            y_img = y_top - img_box_h
            y_label = y_img - label_h

            # Original cell
            col_i = 0
            if include_original:
                orig_path = gt_root / img_name
                x = margin + col_i * cell_w
                draw_image_fit(c, orig_path, x, y_img, cell_w, img_box_h)

                # label under original (no filename)
                label = f"GT={gt_str}"
                if show_ids:
                    label = f"[{ridx}]  GT={gt_str}"
                c.drawCentredString(x + cell_w / 2, y_label + 0.02 * inch, label[:120])
                col_i += 1

            # Run cells
            for title, run_dir, run_map in cols:
                x = margin + col_i * cell_w
                ex = run_map.get(img_name)
                pred = ex.pred if ex else None
                pred_str = str(pred) if pred in (1, 2, 3) else "N/A"
                ok = (pred == gt) if (pred in (1,2,3) and gt in (1,2,3)) else None
                mark = None  # drawn separately as ✓/✗

                img_path = (ex.img_path if (ex and ex.img_path and ex.img_path.exists()) else None)
                if img_path is None:
                    # draw missing box
                    c.rect(x, y_img, cell_w, img_box_h, stroke=1, fill=0)
                    c.setFont("Times-Roman", 7)
                    c.drawCentredString(x + cell_w / 2, y_img + img_box_h / 2 - 3, "missing")
                    c.setFont("Times-Roman", font_size)
                else:
                    draw_image_fit(c, img_path, x, y_img, cell_w, img_box_h)

                draw_pred_label(c, x + cell_w / 2, y_label + 0.02 * inch, pred_str, ok, font_size)
                col_i += 1

            y_cursor = y_label - row_gap

    # paginate
    all_rows = [r for r in picked_rows if 0 <= r < len(ordered_imgs)]
    for start in range(0, len(all_rows), rpp):
        page_rows = all_rows[start:start + rpp]
        draw_page(page_rows)
        if start + rpp < len(all_rows):
            c.showPage()

    c.save()


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate a standalone PDF comparison figure (no LaTeX needed).")
    ap.add_argument("--gt-root", type=Path, required=True, help="Folder with sim_*_initial.png and sim_*_results.json")
    ap.add_argument("--col-dir", type=Path, action="append", default=[], help="Run dir to add as a column (repeatable).")
    ap.add_argument("--col-title", type=str, action="append", default=[], help="Title for each --col-dir (repeatable).")

    ap.add_argument("--rows", type=str, default="", help='Row indices to include (0-based). E.g. "0,3,5,9" or "0-9,12".')
    ap.add_argument("--max-rows", type=int, default=12, help="If --rows not provided, take the first N rows.")

    ap.add_argument("--out-pdf", type=Path, required=True, help="Output PDF file")

    ap.add_argument("--no-original", action="store_true", help="If set, do NOT include the original image column.")
    ap.add_argument("--no-ids", action="store_true", help="If set, do NOT show [row] and filename under originals.")

    ap.add_argument("--page", choices=["letter", "a4"], default="letter", help="Page size")
    ap.add_argument("--landscape", action="store_true", help="Landscape orientation")

    ap.add_argument("--margin-in", type=float, default=0.35, help="Page margin (inches)")
    ap.add_argument("--header-h-in", type=float, default=0.35, help="Header height (inches)")
    ap.add_argument("--label-h-in", type=float, default=0.26, help="Label band height under images (inches)")
    ap.add_argument("--row-gap-in", type=float, default=0.10, help="Gap between rows (inches)")

    ap.add_argument("--font-size", type=int, default=7, help="Body font size")
    ap.add_argument("--header-font-size", type=int, default=9, help="Header font size")
    ap.add_argument("--pt-per-inch", type=float, default=55.0,
                help="PDF points per inch. Lower = tighter spacing. Default 55.")


    args = ap.parse_args()

    if not args.col_dir:
        raise SystemExit("Need at least one --col-dir")

    titles = list(args.col_title or [])
    while len(titles) < len(args.col_dir):
        titles.append(args.col_dir[len(titles)].name)
    titles = titles[:len(args.col_dir)]

    ordered, gt_map = load_gt(args.gt_root)
    if not ordered:
        raise SystemExit(f"No GT found in {args.gt_root} (expected sim_*_results.json)")

    picked = parse_rows_spec(args.rows)
    if picked is None:
        picked = list(range(min(args.max_rows, len(ordered))))

    cols: List[Tuple[str, Path, Dict[str, RunExample]]] = []
    for title, d in zip(titles, args.col_dir):
        cols.append((title, d, load_run_dir(d)))

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    make_pdf(
        out_pdf=args.out_pdf,
        ordered_imgs=ordered,
        gt_map=gt_map,
        gt_root=args.gt_root,
        cols=cols,
        picked_rows=picked,
        include_original=(not args.no_original),
        show_ids=(not args.no_ids),
        page=args.page,
        landscape_mode=args.landscape,
        margin_in=float(args.margin_in),
        header_h_in=float(args.header_h_in),
        label_h_in=float(args.label_h_in),
        row_gap_in=float(args.row_gap_in),
        font_size=int(args.font_size),
        header_font_size=int(args.header_font_size),
        pt_per_inch=float(args.pt_per_inch),
    )
    print(f"Wrote PDF: {args.out_pdf}")

if __name__ == "__main__":
    main()


'''

python compare_figure_pdf.py --gt-root vpct-1 `
  --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro (SketchVLM)" `
  --col-dir results/mix_eval/gpt5low_vpct_test --col-title "GPT-5 (low) (No grid)" `
  --rows "0,3,5,9" `
  --out-pdf fig_vpct_compare.pdf
  
  
  
python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro (SketchVLM)" --col-dir results/mix_eval/gpt5low_vpct_test --col-title "GPT-5 (low) (No grid)" --rows "0,3,5,9" --out-pdf fig_vpct_compare.pdf

python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --rows "0,3,5,9" --out-pdf fig_vpct_compare.pdf

python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --rows "0,3,5,9" --no-ids --out-pdf fig_vpct_compare.pdf 



python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --rows "0,3,5,9" --no-ids --out-pdf fig_vpct_compare.pdf 

python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --rows "0,3,5,9" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --out-pdf fig_vpct_compare.pdf 

--row-gap-in
gem3pro_vpct_multi_withtextstrokes



python compare_figure_pdf.py --gt-root vpct-1 --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" --col-dir results/mix_eval/gpt5low_vpct_multiturn --col-title "GPT-5 (low) Multiturn" --rows "0,3,5,8,14,17" --no-ids --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 --pt-per-inch 45 --out-pdf fig_vpct_compare.pdf 



python compare_figure_pdf.py --gt-root vpct-1 `
  --col-dir results/mix_eval/geminipro3_vpct --col-title "Gemini-3-Pro No Grid" `
  --col-dir results/mix_eval/gem3pro_vpct_multi_withtextstrokes --col-title "Gemini-3-Pro Multi-turn" `
  --col-dir results/mix_eval/vpct_ball_gpt5low --col-title "GPT-5 (low) With Grid" `
  --rows "0,3,5,9" --no-ids `
  --margin-in 0.15 --header-h-in 0.16 --label-h-in 0.10 --row-gap-in 0 `
  --pt-per-inch 55 `
  --out-pdf fig_vpct_compare.pdf

'''