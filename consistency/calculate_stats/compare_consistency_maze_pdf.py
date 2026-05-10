#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_consistency_maze_pdf.py

Generate a PDF comparison figure for maze consistency results across multiple models.
Each column shows a different model's consistency checks, each row shows a different sample.
Similar in style to compare_maze_quality_pdf.py.

Usage:
  python consistency/compare_consistency_maze_pdf.py \
    --judge-dir consistency/judge_output/grid_world_consistency \
    --models consistency_results_gemini3pro_valid consistency_results_gpt5_low_valid \
    --titles "Gemini-3-Pro" "GPT-5 (low)" \
    --rows "0-15" \
    --out-pdf consistency/figure_pdfs/consistency_maze_valid.pdf \
    --answer-type word
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
class ConsistencyEntry:
    """Represents a single consistency check result."""
    index: int
    image_path: Optional[Path] = None
    original_answer: Optional[str] = None
    judge_answer: Optional[str] = None
    success: bool = False
    norm_original: Optional[str] = None
    norm_judge: Optional[str] = None
    is_consistent: Optional[bool] = None
    source_image_path: Optional[Path] = None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from $\boxed{...}$ format."""
    if not text:
        return None

    # Try to find $\boxed{X}$ pattern
    boxed_match = re.search(r'\$\\boxed\{([^}]+)\}\$', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try without dollar signs
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()

    return None


def normalize_answer(answer: str, answer_type: str = 'number') -> Optional[str]:
    """Normalize answer to just the number or valid/invalid."""
    if not answer:
        return None

    answer = str(answer).strip().lower()

    if answer_type == 'word':
        # Word mode: only extract valid/invalid
        if 'invalid' in answer:
            return 'invalid'
        elif 'valid' in answer:
            return 'valid'
        return None
    else:
        # Number mode: extract numbers
        number_match = re.search(r'\d+', answer)
        if number_match:
            return number_match.group(0)

        # Handle special cases
        if 'none' in answer:
            return 'none'
        if 'multiple' in answer:
            return 'multiple'

    return answer


def load_model_data(judge_dir: Path, model_name: str, answer_type: str = 'number', source_root: Optional[Path] = None) -> Dict[int, ConsistencyEntry]:
    """Load consistency data for a single model."""
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

        original_extracted = entry.get('original_extracted_answer', '')
        judge_response = entry.get('consistency_check_response', '')
        success = entry.get('success', False)

        # Extract judge's answer from boxed format
        judge_answer = extract_boxed_answer(judge_response)

        # Normalize answers
        norm_original = normalize_answer(original_extracted, answer_type) if original_extracted else None
        norm_judge = normalize_answer(judge_answer, answer_type) if judge_answer else None

        # Check if consistent
        is_consistent = (norm_original == norm_judge) if (norm_original and norm_judge) else None

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

        results[idx] = ConsistencyEntry(
            index=idx,
            image_path=img_path,
            original_answer=original_extracted,
            judge_answer=judge_answer or '',
            success=success,
            norm_original=norm_original,
            norm_judge=norm_judge,
            is_consistent=is_consistent,
            source_image_path=source_image_path
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


def make_consistency_pdf(out_pdf: Path, model_data: Dict[str, Dict[int, ConsistencyEntry]],
                        model_titles: List[str], picked_rows: List[int],
                        include_source: bool = True, source_title: str = "Source Image",
                        page: str = "letter", landscape_mode: bool = True,
                        pt_per_inch: float = 30.0, margin_in: float = 0.15,
                        header_h_in: float = 0.36, label_h_in: float = 0.50,
                        row_gap_in: float = 0.0, col_gap_pt: float = 1.0,
                        font_size: int = 7, header_font_size: int = 8,
                        header_max_lines: int = 2, header_leading: Optional[float] = None,
                        img_h_in: Optional[float] = None, fixed_img_aspect: float = 1.0) -> None:
    """Generate the consistency comparison PDF with fixed aspect ratio columns."""

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

    # Ensure columns fit across the available page width even when we add a new column
    usable_w = max(pw - 2 * margin, 1)
    if ncols > 0:
        max_cell_w = (usable_w - (ncols - 1) * col_gap) / ncols
        if max_cell_w <= 0:
            max_cell_w = usable_w / ncols
        if cell_w > max_cell_w:
            cell_w = max_cell_w
            if fixed_img_aspect != 0:
                target_img_h = cell_w / fixed_img_aspect

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
        label_padding = max(font_size * 0.9, 5)

        for ridx in page_rows:
            y_top = y_cursor
            y_img = y_top - target_img_h
            y_label = y_img - label_h

            col_i = 0

            # Draw source image column if requested
            if include_source:
                x = margin + col_i * (cell_w + col_gap)

                # Get source image from first model's data
                source_path = None
                for model_name in model_data.keys():
                    entry = model_data[model_name].get(ridx)
                    if entry and entry.source_image_path:
                        source_path = entry.source_image_path
                        break

                draw_image_fit(c, source_path, x, y_img, cell_w, target_img_h)
                col_i += 1

            # Draw model columns
            for model_name, title in zip(model_data.keys(), model_titles):
                x = margin + col_i * (cell_w + col_gap)

                entry = model_data[model_name].get(ridx)

                # Draw image with crop settings for specific models
                img_path = entry.image_path if entry else None
                model_lower = model_name.lower()
                crop_left = 22 if 'gpt5' in model_lower else 0
                crop_bottom = 20 if 'gpt5' in model_lower else 0
                draw_image_fit(c, img_path, x, y_img, cell_w, target_img_h,
                              crop_left=crop_left, crop_bottom=crop_bottom)

                # Draw consistency labels
                if entry and entry.norm_original and entry.norm_judge:
                    # Build label text
                    sketch_text = f"Sketch: {entry.norm_original}"
                    judge_text = f"Judge: {entry.norm_judge}"

                    # Choose emoji based on consistency
                    if entry.is_consistent:
                        emoji = "✓"
                        emoji_color = colors.green
                    else:
                        emoji = "✗"
                        emoji_color = colors.red

                    # Draw two lines with emoji on second line
                    c.setFillColor(colors.black)
                    c.setFont("Times-Roman", font_size)

                    # Start position
                    line_y = y_label + label_h - label_padding
                    line_height = font_size * 1.3

                    # Line 1: Sketch answer
                    c.drawCentredString(x + cell_w / 2, line_y, sketch_text)

                    # Line 2: Judge answer (left side) + emoji (right side)
                    line_y -= line_height

                    # Calculate text width to position emoji
                    judge_width = stringWidth(judge_text, "Times-Roman", font_size)

                    # Draw judge text
                    c.drawCentredString(x + cell_w / 2, line_y, judge_text)

                    # Draw emoji slightly to the right of the text
                    c.setFillColor(emoji_color)
                    c.setFont("Times-Roman", 10)
                    emoji_x = x + cell_w / 2 + judge_width / 2 + 3  # Reduced spacing
                    c.drawString(emoji_x, line_y, emoji)

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
        description="Generate PDF comparison figure for maze consistency results"
    )
    parser.add_argument("--judge-dir", type=Path, required=True,
                       help="Directory containing maze consistency judge JSON files")
    parser.add_argument("--source-root", type=Path, default=None,
                       help="Directory containing source maze images (e.g., datasets/maze_v2/sketch_valid_flattened)")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model names to include (JSON file stems)")
    parser.add_argument("--titles", nargs="+", default=None,
                       help="Custom titles for each model (optional)")
    parser.add_argument("--rows", type=str, default="0-20",
                       help="Row indices to include (e.g., '0,3,5' or '0-10')")
    parser.add_argument("--exclude-rows", type=str, default=None,
                       help="Row indices to exclude (same format as --rows)")
    parser.add_argument("--original-filter", type=str, choices=['valid', 'invalid'],
                       default=None,
                       help="Only include rows where the normalized original answer matches this label")
    parser.add_argument("--out-pdf", type=Path, required=True,
                       help="Output PDF file path")
    parser.add_argument("--answer-type", type=str, choices=['number', 'word'],
                       default='word',
                       help='Answer type: "number" for ball drop, "word" for maze')
    parser.add_argument("--no-source", action="store_true",
                       help="Don't include source image column")

    # Exclude entries with errors by default
    parser.add_argument("--include-errors", action="store_true",
                       help="Include entries with API errors or missing answers")

    # Page layout options
    parser.add_argument("--page", choices=["letter", "a4"], default="letter")
    parser.add_argument("--landscape", action="store_true", default=True)
    parser.add_argument("--margin-in", type=float, default=0.15)
    parser.add_argument("--header-h-in", type=float, default=0.36)
    parser.add_argument("--label-h-in", type=float, default=0.50)
    parser.add_argument("--row-gap-in", type=float, default=0.08)
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
        data = load_model_data(args.judge_dir, model_name, answer_type=args.answer_type, source_root=args.source_root)
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

    # Optionally remove specific rows
    excluded_rows = parse_rows_spec(args.exclude_rows)
    if excluded_rows:
        excluded_set = set(excluded_rows)
        before = len(picked_rows)
        picked_rows = [r for r in picked_rows if r not in excluded_set]
        removed = before - len(picked_rows)
        if removed > 0:
            print(f"Excluded {removed} rows via --exclude-rows")

    def row_matches_filters(ridx: int) -> bool:
        has_entry = False
        for model_name in model_data.keys():
            entry = model_data[model_name].get(ridx)
            if not entry:
                return False
            has_entry = True

            if not args.include_errors:
                if not entry.success or not entry.norm_original or not entry.norm_judge:
                    return False

            if args.original_filter and entry.norm_original != args.original_filter:
                return False

        return has_entry

    filtered_rows = [r for r in picked_rows if row_matches_filters(r)]
    excluded_count = len(picked_rows) - len(filtered_rows)
    if excluded_count > 0:
        msg_parts = []
        if not args.include_errors:
            msg_parts.append("errors or missing answers")
        if args.original_filter:
            msg_parts.append(f"original != '{args.original_filter}'")
        reason = " and ".join(msg_parts) if msg_parts else "filters"
        print(f"Excluded {excluded_count} rows due to {reason}")
    picked_rows = filtered_rows

    if not picked_rows:
        print("Error: No valid rows to display after filtering")
        return

    print(f"Creating PDF with {len(model_data)} models and {len(picked_rows)} rows...")

    # Create output directory if needed
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    make_consistency_pdf(
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
