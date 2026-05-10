#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_maze_quality_pdf.py

Generate a PDF comparison figure for maze quality scores across multiple models.
Each column shows a different model's quality scores, each row shows a different sample.
Similar in style to compare_figure_pdf.py but focused on maze quality scores.

By default, rows with any errors (API failures or missing quality scores) are excluded.

Usage:
  python consistency/compare_maze_quality_pdf.py \
    --judge-dir consistency/judge_output/grid_world_quality \
    --source-root datasets/maze_v2/sketch_valid_flattened \
    --models gemini3_pro_valid gpt5_low_valid thinkmorph_valid nano_banana_valid vilasr_valid \
    --rows "0-20" \
    --out-pdf consistency/fig_maze_quality_compare.pdf

For invalid paths:
  python consistency/compare_maze_quality_pdf.py \
    --judge-dir consistency/judge_output/grid_world_quality \
    --source-root datasets/maze_v2/sketch_invalid_flattened \
    --models gemini3_pro_invalid gpt5_low_invalid thinkmorph_invalid \
    --titles "Gemini-3-Pro" "GPT-5 (low)" "ThinkMorph" \
    --rows "0-10" \
    --out-pdf consistency/fig_maze_quality_invalid.pdf
"""

import argparse
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth


@dataclass
class QualityEntry:
    index: int
    image_path: Optional[Path] = None
    quality_score: Optional[int] = None
    judge_response: Optional[str] = None
    success: bool = False
    validity_answer: Optional[str] = None
    source_image_path: Optional[Path] = None
    proposed_path: Optional[str] = None


def extract_quality_score(text: str) -> Optional[int]:
    """Extract quality score from judge response."""
    if not text:
        return None

    # Try to find "Quality Score: X" pattern
    score_match = re.search(r'Quality\s+Score:\s*(\d+)', text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 5:
            return score

    return None


def extract_proposed_path(prompt: str) -> Optional[str]:
    """Extract proposed path from the prompt."""
    if not prompt:
        return None

    # Look for "Proposed path: ..." pattern
    match = re.search(r'Proposed\s+path:\s*([^\n]+)', prompt, re.IGNORECASE)
    if match:
        path = match.group(1).strip()
        # Clean up the path
        path = path.rstrip('.')
        return path

    return None


def load_model_data(judge_dir: Path, model_name: str, source_root: Optional[Path] = None) -> Dict[int, QualityEntry]:
    """Load quality data for a single model."""
    json_file = judge_dir / f"{model_name}.json"

    if not json_file.exists():
        print(f"Warning: {json_file} not found")
        return {}

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return {}

    results = {}
    for entry in data:
        idx = entry.get('index')
        if idx is None:
            continue

        img_path_str = entry.get('image_path', '')
        img_path = Path(img_path_str) if img_path_str else None
        if img_path and not img_path.is_absolute():
            # Make relative paths absolute
            img_path = Path.cwd() / img_path

        judge_response = entry.get('consistency_check_response', '')
        success = entry.get('success', False)
        validity = entry.get('original_extracted_answer')
        prompt = entry.get('prompt', '')

        quality_score = None
        if success and judge_response:
            quality_score = extract_quality_score(judge_response)

        # Extract proposed path from prompt
        proposed_path = extract_proposed_path(prompt)

        # Try to find source image
        source_image_path = None
        if img_path and img_path.exists() and source_root:
            # Try to load the corresponding item JSON to get source image
            item_dir = img_path.parent
            item_file = item_dir / f"item_{idx:05d}.json"
            if item_file.exists():
                try:
                    with open(item_file) as f:
                        item_data = json.load(f)
                    source_str = item_data.get('source_image') or item_data.get('image') or item_data.get('file', '')
                    if source_str:
                        # Convert backslashes and construct path
                        source_str = source_str.replace('\\', '/')
                        source_basename = Path(source_str).name
                        source_image_path = source_root / source_basename
                        if not source_image_path.exists():
                            source_image_path = None
                except Exception:
                    pass

        results[idx] = QualityEntry(
            index=idx,
            image_path=img_path,
            quality_score=quality_score,
            judge_response=judge_response,
            success=success,
            validity_answer=validity,
            source_image_path=source_image_path,
            proposed_path=proposed_path
        )

    return results


def parse_rows_spec(spec: Optional[str]) -> Optional[List[int]]:
    """Parse row specification like '0,3,5,9' or '0-10'."""
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
                lo = int(a)
                hi = int(b)
                step = 1 if hi >= lo else -1
                for x in range(lo, hi + step, step):
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
            continue
        if tok.isdigit():
            x = int(tok)
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def draw_image_fit(c: canvas.Canvas, img_path: Optional[Path], x: float, y: float, w: float, h: float,
                   crop_left: int = 0, crop_bottom: int = 0) -> None:
    """Draw image into (x,y,w,h) box, cropping to exact dimensions.

    Args:
        crop_left: Number of pixels to crop from left edge
        crop_bottom: Number of pixels to crop from bottom edge
    """
    if not img_path or not img_path.exists():
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, "missing")
        return

    try:
        # Load image and apply crop if needed
        if crop_left > 0 or crop_bottom > 0:
            with Image.open(img_path) as im:
                iw, ih = im.size
                # Crop: (left, top, right, bottom)
                # Remove crop_left from left, crop_bottom from bottom
                cropped = im.crop((crop_left, 0, iw, ih - crop_bottom))
                iw, ih = cropped.size

                if iw <= 0 or ih <= 0:
                    raise ValueError("bad image size")

                # Save cropped image to BytesIO buffer
                buffer = BytesIO()
                cropped.save(buffer, format='PNG')
                buffer.seek(0)
                img_reader = ImageReader(buffer)
        else:
            # No crop needed, use file path directly
            with Image.open(img_path) as im:
                iw, ih = im.size

            if iw <= 0 or ih <= 0:
                raise ValueError("bad image size")
            img_reader = ImageReader(str(img_path))

    except Exception as e:
        c.rect(x, y, w, h, stroke=1, fill=0)
        c.setFont("Times-Roman", 7)
        c.drawCentredString(x + w / 2, y + h / 2 - 3, "error")
        return

    # Fill mode: scale to fill the box, then clip to exact dimensions
    scale = max(w / iw, h / ih)
    dw = iw * scale
    dh = ih * scale
    dx = x + (w - dw) / 2
    dy = y + (h - dh) / 2

    # Clip to exact column boundaries
    c.saveState()
    p = c.beginPath()
    p.rect(x, y, w, h)
    c.clipPath(p, stroke=0, fill=0)
    c.drawImage(img_reader, dx, dy, width=dw, height=dh,
                preserveAspectRatio=True, mask="auto")
    c.restoreState()


def wrap_label_text(text: str, cell_w: float, font_name: str, font_size: int) -> List[str]:
    """Wrap label text to fit within a column width."""
    if not text:
        return [""]

    # Split by commas first for path-like text
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        lines = []
        current_line = parts[0]

        for part in parts[1:]:
            test_line = current_line + ", " + part
            if stringWidth(test_line, font_name, font_size) <= (cell_w * 0.95):
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = part
        lines.append(current_line)
        return lines

    # For non-comma text, wrap by words
    words = text.split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = current_line + " " + word
        if stringWidth(test_line, font_name, font_size) <= (cell_w * 0.95):
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def draw_label(c: canvas.Canvas, x_center: float, y: float, text: str,
               font_size: int, color=colors.black, cell_w: Optional[float] = None,
               max_lines: int = 5, top_justify: bool = False,
               top_padding: Optional[float] = None) -> None:
    """Draw centered label text, with optional wrapping.

    Args:
        top_justify: If True, y is the TOP of the text area and text flows downward
        top_padding: Optional amount of space (in points) between the image above and the first line
    """
    c.setFillColor(color)
    c.setFont("Times-Roman", font_size)

    if cell_w is not None:
        # Wrap text to fit within cell width
        lines = wrap_label_text(text, cell_w, "Times-Roman", font_size)
        lines = lines[:max_lines]  # Limit to max_lines
    else:
        lines = [text]

    line_height = font_size * 1.2

    if top_justify:
        padding = top_padding if top_padding is not None else (font_size * 0.9)
        start_y = y - max(padding, 2)  # Ensure at least a small gap below the image
        for i, line in enumerate(lines):
            c.drawCentredString(x_center, start_y - i * line_height, line)
        return

    if len(lines) == 1:
        # Single line, no wrapping or centering needed
        c.drawCentredString(x_center, y, lines[0])
        return

    # Draw centered block of wrapped lines
    start_y = y + (len(lines) - 1) * line_height / 2
    for i, line in enumerate(lines):
        c.drawCentredString(x_center, start_y - i * line_height, line)


def wrap_header_text(title: str, cell_w: float, font_name: str, font_size: int,
                     max_lines: int = 2) -> List[str]:
    """Wrap header text to fit within a column width."""
    if title is None:
        return [""]
    t = str(title)
    if "\n" in t:
        lines = [ln.strip() for ln in t.split("\n")]
        return [ln for ln in lines if ln] or [""]
    words = t.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = cur + " " + w
        if stringWidth(cand, font_name, font_size) <= (cell_w * 0.98):
            cur = cand
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines - 1:
                break
    lines.append(cur)

    # If we still have leftover words, append ellipsis to last line
    used = sum(len(ln.split()) for ln in lines)
    if used < len(words):
        last = lines[-1]
        if not last.endswith("…"):
            lines[-1] = (last + " …")
    return lines


def draw_header_cell(c: canvas.Canvas, x0: float, y_top: float, cell_w: float,
                     header_h: float, title: str, font_name: str, font_size: int,
                     max_lines: int = 2, leading: Optional[float] = None) -> None:
    """Draw a (possibly multi-line) header centered inside the header band."""
    lines = wrap_header_text(title, cell_w, font_name, font_size, max_lines=max_lines)
    lead = leading if leading is not None else (font_size * 1.05)
    total_h = lead * len(lines)
    # Vertical centering inside [y_top-header_h, y_top]
    y_start = (y_top - header_h) + (header_h - total_h) / 2 + (len(lines) - 1) * lead
    c.setFont(font_name, font_size)
    for i, ln in enumerate(lines):
        c.drawCentredString(x0 + cell_w / 2, y_start - i * lead, ln)


def make_quality_pdf(out_pdf: Path, model_data: Dict[str, Dict[int, QualityEntry]],
                    model_titles: List[str], picked_rows: List[int],
                    include_source: bool = True, source_title: str = "Source Image",
                    page: str = "letter", landscape_mode: bool = True,
                    pt_per_inch: float = 45.0, margin_in: float = 0.15,
                    header_h_in: float = 0.36, label_h_in: float = 0.10,
                    row_gap_in: float = 0.0, col_gap_pt: float = 1.0,
                    font_size: int = 7, header_font_size: int = 8,
                    header_max_lines: int = 2, header_leading: Optional[float] = None,
                    img_h_in: Optional[float] = None, fixed_img_aspect: float = 1.0) -> None:
    """Generate the quality comparison PDF with fixed aspect ratio columns."""

    pagesize = letter if page.lower() == "letter" else A4
    pw, ph = pagesize
    if landscape_mode:
        pw, ph = ph, pw

    inch = float(pt_per_inch)
    margin = margin_in * inch
    header_h = header_h_in * inch
    label_h = label_h_in * inch
    row_gap = row_gap_in * inch
    col_gap = col_gap_pt  # Column gap in points

    # Add source column if requested
    all_titles = ([source_title] if include_source else []) + model_titles
    ncols = len(all_titles)

    # Calculate target image height first
    target_img_h = (img_h_in * inch) if (img_h_in is not None) else (2.2 * inch)

    # Cell width is determined by the fixed aspect ratio (maze images are typically square)
    cell_w = target_img_h * fixed_img_aspect

    def rows_per_page() -> int:
        usable_h = ph - 2 * margin - header_h
        per_row = target_img_h + label_h + row_gap
        return max(1, int(usable_h // per_row)) if per_row > 0 else 1

    rpp = rows_per_page()
    c = canvas.Canvas(str(out_pdf), pagesize=(pw, ph))
    c.setTitle(out_pdf.stem)

    def draw_header():
        font_name = "Times-Bold"
        y_top = ph - margin

        for i, title in enumerate(all_titles):
            x0 = margin + i * (cell_w + col_gap)
            draw_header_cell(
                c, x0=x0, y_top=y_top, cell_w=cell_w, header_h=header_h,
                title=title, font_name=font_name, font_size=header_font_size,
                max_lines=header_max_lines, leading=header_leading
            )

    def draw_page(page_rows: List[int]):
        draw_header()
        y_cursor = ph - margin - header_h
        path_label_padding = max(font_size * 1.1, 5)
        score_label_padding = max(font_size * 0.9, 5)

        for ridx in page_rows:
            y_top = y_cursor
            y_img = y_top - target_img_h
            y_label = y_img - label_h

            col_i = 0

            # Draw source image column if requested
            if include_source:
                x = margin + col_i * (cell_w + col_gap)

                # Get source image and proposed path from first model's data
                source_path = None
                proposed_path = None
                for model_name in model_data.keys():
                    entry = model_data[model_name].get(ridx)
                    if entry and entry.source_image_path:
                        source_path = entry.source_image_path
                        proposed_path = entry.proposed_path
                        break

                draw_image_fit(c, source_path, x, y_img, cell_w, target_img_h)

                # Draw proposed path label under source image (with wrapping, top-justified)
                if proposed_path:
                    path_text = f"Path: {proposed_path}"
                    draw_label(c, x + cell_w / 2, y_label + label_h,
                              path_text, font_size, color=colors.black,
                              cell_w=cell_w, max_lines=5, top_justify=True,
                              top_padding=path_label_padding)

                col_i += 1

            # Draw model columns
            for model_name, title in zip(model_data.keys(), model_titles):
                x = margin + col_i * (cell_w + col_gap)

                entry = model_data[model_name].get(ridx)

                # Draw image with crop settings for specific models
                img_path = entry.image_path if entry else None
                model_lower = model_name.lower()
                crop_left = 22 if ('gpt5' in model_lower or 'gemini3' in model_lower or 'gemini_3' in model_lower) else 0
                crop_bottom = 20 if ('gpt5' in model_lower or 'gemini3' in model_lower or 'gemini_3' in model_lower) else 0
                draw_image_fit(c, img_path, x, y_img, cell_w, target_img_h,
                              crop_left=crop_left, crop_bottom=crop_bottom)

                # Draw label with score (black text only, "Score: X" format)
                if entry and entry.success and entry.quality_score is not None:
                    label_text = f"Score: {entry.quality_score}"
                elif entry and not entry.success:
                    label_text = "Error"
                else:
                    label_text = "N/A"

                # Draw score label (top-justified to match source column)
                draw_label(c, x + cell_w / 2, y_label + label_h,
                          label_text, font_size, color=colors.black,
                          cell_w=None, top_justify=True,
                          top_padding=score_label_padding)  # Single-line labels with explicit padding

                col_i += 1

            y_cursor = y_label - row_gap

    all_rows = [r for r in picked_rows if any(r in model_data[m] for m in model_data.keys())]

    for start in range(0, len(all_rows), rpp):
        draw_page(all_rows[start:start + rpp])
        if start + rpp < len(all_rows):
            c.showPage()

    c.save()
    print(f"✓ Wrote PDF: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF comparison figure for maze quality scores"
    )
    parser.add_argument("--judge-dir", type=Path, required=True,
                       help="Directory containing maze quality judge JSON files")
    parser.add_argument("--source-root", type=Path, default=None,
                       help="Directory containing source maze images (e.g., datasets/maze_v2/sketch_valid_flattened)")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model names to include (JSON file stems)")
    parser.add_argument("--titles", nargs="+", default=None,
                       help="Custom titles for each model (optional)")
    parser.add_argument("--rows", type=str, default="0-20",
                       help="Row indices to include (e.g., '0,3,5' or '0-10')")
    parser.add_argument("--out-pdf", type=Path, required=True,
                       help="Output PDF file path")
    parser.add_argument("--no-source", action="store_true",
                       help="Don't include source image column")

    # Exclude entries with errors by default
    parser.add_argument("--include-errors", action="store_true",
                       help="Include entries with API errors or missing scores")

    # Page layout options
    parser.add_argument("--page", choices=["letter", "a4"], default="letter")
    parser.add_argument("--landscape", action="store_true", default=True)
    parser.add_argument("--margin-in", type=float, default=0.15)
    parser.add_argument("--header-h-in", type=float, default=0.36)
    parser.add_argument("--label-h-in", type=float, default=0.60)
    parser.add_argument("--row-gap-in", type=float, default=0.0)
    parser.add_argument("--col-gap-pt", type=float, default=1.0,
                       help="Column gap in points (default: 1.0)")
    parser.add_argument("--font-size", type=int, default=7)
    parser.add_argument("--header-font-size", type=int, default=8)
    parser.add_argument("--header-max-lines", type=int, default=2)
    parser.add_argument("--header-leading", type=float, default=None)
    parser.add_argument("--pt-per-inch", type=float, default=30.0)
    parser.add_argument("--img-h-in", type=float, default=None)
    parser.add_argument("--img-aspect", type=float, default=1.0,
                       help="Image aspect ratio (width/height), default 1.0 for square")

    args = parser.parse_args()

    # Load data for each model
    model_data = {}
    for model_name in args.models:
        data = load_model_data(args.judge_dir, model_name, source_root=args.source_root)
        if data:
            model_data[model_name] = data
        else:
            print(f"Warning: No data loaded for {model_name}")

    if not model_data:
        print("Error: No model data could be loaded")
        return

    # Use custom titles if provided, otherwise use model names
    if args.titles:
        if len(args.titles) != len(model_data):
            print(f"Warning: Number of titles ({len(args.titles)}) doesn't match number of models ({len(model_data)})")
            model_titles = list(model_data.keys())
        else:
            model_titles = args.titles
    else:
        # Clean up model names for display
        model_titles = []
        for name in model_data.keys():
            # Remove common prefixes/suffixes
            clean_name = name.replace('consistency_results_', '').replace('_valid', '').replace('_invalid', '')
            clean_name = clean_name.replace('_', ' ').title()
            model_titles.append(clean_name)

    # Parse row specification
    picked_rows = parse_rows_spec(args.rows)
    if picked_rows is None:
        picked_rows = list(range(21))  # Default to first 21 rows

    # Filter out rows with errors if requested
    if not args.include_errors:
        valid_rows = []
        for ridx in picked_rows:
            # Check if this row has valid data for all models
            all_valid = True
            for model_name in model_data.keys():
                entry = model_data[model_name].get(ridx)
                if not entry or not entry.success or entry.quality_score is None:
                    all_valid = False
                    break
            if all_valid:
                valid_rows.append(ridx)

        excluded_count = len(picked_rows) - len(valid_rows)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} rows with errors or missing scores")
        picked_rows = valid_rows

    if not picked_rows:
        print("Error: No valid rows to display after filtering")
        return

    print(f"Creating PDF with {len(model_data)} models and {len(picked_rows)} rows...")

    # Create output directory if needed
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    make_quality_pdf(
        out_pdf=args.out_pdf,
        model_data=model_data,
        model_titles=model_titles,
        picked_rows=picked_rows,
        include_source=(not args.no_source),
        page=args.page,
        landscape_mode=args.landscape,
        pt_per_inch=args.pt_per_inch,
        margin_in=args.margin_in,
        header_h_in=args.header_h_in,
        label_h_in=args.label_h_in,
        row_gap_in=args.row_gap_in,
        col_gap_pt=args.col_gap_pt,
        font_size=args.font_size,
        header_font_size=args.header_font_size,
        header_max_lines=args.header_max_lines,
        header_leading=args.header_leading,
        img_h_in=args.img_h_in,
        fixed_img_aspect=args.img_aspect
    )


if __name__ == "__main__":
    main()


'''
EXAMPLE USAGE:

# Valid paths
python consistency/compare_maze_quality_pdf.py \
  --judge-dir consistency/judge_output/grid_world_quality \
  --source-root datasets/maze_v2/sketch_valid_flattened \
  --models gemini3_pro_valid gpt5_low_valid thinkmorph_valid \
  --titles "Gemini-3-Pro-Preview" "GPT-5 (low)" "ThinkMorph" \
  --rows "0-15" \
  --margin-in 0.15 --header-h-in 0.36 --label-h-in 0.20 --row-gap-in 0 --col-gap-pt 1.0 \
  --pt-per-inch 30 --header-font-size 8 --header-max-lines 2 \
  --out-pdf consistency/fig_maze_quality_valid_compare.pdf

# Invalid paths
python consistency/compare_maze_quality_pdf.py \
  --judge-dir consistency/judge_output/grid_world_quality \
  --source-root datasets/maze_v2/sketch_invalid_flattened \
  --models gemini3_pro_invalid gpt5_low_invalid thinkmorph_invalid \
  --titles "Gemini-3-Pro-Preview" "GPT-5 (low)" "ThinkMorph" \
  --rows "0-15" \
  --margin-in 0.15 --header-h-in 0.36 --label-h-in 0.20 --row-gap-in 0 --col-gap-pt 1.0 \
  --pt-per-inch 30 --header-font-size 8 --header-max-lines 2 \
  --out-pdf consistency/fig_maze_quality_invalid_compare.pdf

# With NanoBanana
python consistency/compare_maze_quality_pdf.py \
  --judge-dir consistency/judge_output/grid_world_quality \
  --source-root datasets/maze_v2/sketch_valid_flattened \
  --models gemini3_pro_valid gpt5_low_valid thinkmorph_valid consistency_results_nano_banana_valid vilasr_valid \
  --titles "Gemini-3-Pro-Preview" "GPT-5 (low)" "ThinkMorph" "NanoBanana Pro" "ViLaSR" \
  --rows "0,4,5,6,7,8,11,12,13,14" \
  --margin-in 0.15 --header-h-in 0.36 --label-h-in 0.60 --row-gap-in 0 --col-gap-pt 1.0 \
  --pt-per-inch 30 --header-font-size 8 --header-max-lines 2 \
  --out-pdf consistency/fig_maze_quality_compare.pdf
'''
