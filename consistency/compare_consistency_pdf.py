"""
Generate PDF comparison figure for consistency results.

Usage:
    python compare_consistency_pdf.py --judge-file consistency/judge_output/.../consistency_results_*.json --output-pdf output.pdf
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth

from PIL import Image


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


class ConsistencyEntry:
    """Represents a single consistency check result."""

    def __init__(self, index: int, image_path: Optional[Path],
                 original_answer: str, judge_answer: str,
                 success: bool, answer_type: str = 'number'):
        self.index = index
        self.image_path = image_path
        self.original_answer = original_answer
        self.judge_answer = judge_answer
        self.success = success
        self.answer_type = answer_type

        # Normalize answers
        self.norm_original = normalize_answer(original_answer, answer_type) if original_answer else None
        self.norm_judge = normalize_answer(judge_answer, answer_type) if judge_answer else None

        # Check if consistent
        self.is_consistent = (self.norm_original == self.norm_judge) if (self.norm_original and self.norm_judge) else None


def load_consistency_data(judge_file: Path, answer_type: str = 'number') -> List[ConsistencyEntry]:
    """Load consistency results from JSON file."""
    with open(judge_file, 'r') as f:
        data = json.load(f)

    entries = []
    for item in data:
        index = item.get('index', 0)
        image_path = Path(item.get('image_path', '')) if item.get('image_path') else None
        original_extracted = item.get('original_extracted_answer', '')
        judge_response = item.get('consistency_check_response', '')
        success = item.get('success', False)

        # Extract judge's answer from boxed format
        judge_answer = extract_boxed_answer(judge_response)

        entry = ConsistencyEntry(
            index=index,
            image_path=image_path,
            original_answer=original_extracted,
            judge_answer=judge_answer or '',
            success=success,
            answer_type=answer_type
        )

        # Only include entries where both answers could be extracted
        if entry.success and entry.norm_original and entry.norm_judge:
            entries.append(entry)

    return entries


def draw_image_fit(c: canvas.Canvas, img_path: Optional[Path], x: float, y: float,
                   w: float, h: float, crop_left: int = 0, crop_bottom: int = 0) -> None:
    """Draw an image fitted to the given dimensions with optional cropping."""
    if not img_path or not img_path.exists():
        # Draw placeholder if image missing
        c.setStrokeColor(colors.grey)
        c.setFillColor(colors.lightgrey)
        c.rect(x, y, w, h, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        c.drawCentredString(x + w/2, y + h/2, "Image not found")
        return

    try:
        if crop_left > 0 or crop_bottom > 0:
            # Crop the image if needed
            with Image.open(img_path) as im:
                iw, ih = im.size
                cropped = im.crop((crop_left, 0, iw, ih - crop_bottom))
                iw, ih = cropped.size

                buffer = BytesIO()
                cropped.save(buffer, format='PNG')
                buffer.seek(0)
                img_reader = ImageReader(buffer)
        else:
            img_reader = ImageReader(str(img_path))
            iw, ih = img_reader.getSize()

        # Fit image to cell preserving aspect ratio
        img_aspect = iw / ih if ih > 0 else 1.0
        target_aspect = w / h if h > 0 else 1.0

        if img_aspect > target_aspect:
            # Image is wider, fit to width
            draw_w = w
            draw_h = w / img_aspect
            draw_x = x
            draw_y = y + (h - draw_h) / 2
        else:
            # Image is taller, fit to height
            draw_h = h
            draw_w = h * img_aspect
            draw_x = x + (w - draw_w) / 2
            draw_y = y

        c.drawImage(img_reader, draw_x, draw_y, draw_w, draw_h)
    except Exception as e:
        print(f"Error drawing image {img_path}: {e}")
        c.setStrokeColor(colors.grey)
        c.setFillColor(colors.lightgrey)
        c.rect(x, y, w, h, fill=1, stroke=1)


def draw_label(c: canvas.Canvas, x_center: float, y: float, text: str,
               font_size: int, color=colors.black) -> None:
    """Draw a text label centered at the given position."""
    c.setFillColor(color)
    c.setFont("Helvetica", font_size)
    c.drawCentredString(x_center, y, text)


def generate_comparison_pdf(
    entries: List[ConsistencyEntry],
    output_pdf: Path,
    title: str = "Consistency Check Results",
    images_per_page: int = 12,
    cols: int = 4
) -> None:
    """Generate a PDF comparing consistency results."""

    # Page setup
    page_w, page_h = letter
    margin_x = 0.5 * inch
    margin_y = 0.5 * inch

    # Title space
    title_h = 0.3 * inch

    # Label space
    label_h_in = 0.60  # Space for two lines of text (Sketch answer + Judge answer)
    label_h = label_h_in * inch

    # Calculate grid dimensions
    rows = (images_per_page + cols - 1) // cols

    # Column gap
    col_gap_pt = 1.0
    col_gap = col_gap_pt / 72.0 * inch

    # Available space
    usable_w = page_w - 2 * margin_x - (cols - 1) * col_gap
    usable_h = page_h - 2 * margin_y - title_h - rows * label_h

    # Cell dimensions (square aspect ratio for consistency images)
    fixed_img_aspect = 1.0  # 1:1 aspect ratio
    target_img_h = usable_h / rows
    cell_w = target_img_h * fixed_img_aspect
    cell_h = target_img_h

    # Create PDF
    c = canvas.Canvas(str(output_pdf), pagesize=letter)

    page_num = 1
    for page_start in range(0, len(entries), images_per_page):
        page_entries = entries[page_start:page_start + images_per_page]

        # Draw title
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_w / 2, page_h - margin_y - 0.15 * inch, title)

        # Draw grid
        for idx, entry in enumerate(page_entries):
            row = idx // cols
            col = idx % cols

            # Calculate position
            x = margin_x + col * (cell_w + col_gap)
            y = page_h - margin_y - title_h - (row + 1) * cell_h - row * label_h
            y_label = y - label_h

            # Draw image
            draw_image_fit(c, entry.image_path, x, y, cell_w, cell_h)

            # Draw labels
            sketch_text = f"Sketch answer: {entry.norm_original}"
            judge_text = f"Judge answer: {entry.norm_judge}"

            # Choose emoji based on consistency
            if entry.is_consistent:
                emoji = "✓"
                emoji_color = colors.green
            else:
                emoji = "✗"
                emoji_color = colors.red

            # Draw sketch answer
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 10)
            c.drawCentredString(x + cell_w / 2, y_label + 0.40 * inch, sketch_text)

            # Draw judge answer
            c.drawCentredString(x + cell_w / 2, y_label + 0.25 * inch, judge_text)

            # Draw emoji
            c.setFillColor(emoji_color)
            c.setFont("Helvetica", 16)
            c.drawCentredString(x + cell_w / 2, y_label + 0.05 * inch, emoji)

        c.showPage()
        page_num += 1

    c.save()
    print(f"Generated PDF: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description='Generate consistency comparison PDF')
    parser.add_argument('--judge-file', type=str, required=True,
                       help='Path to consistency results JSON file')
    parser.add_argument('--output-pdf', type=str, required=True,
                       help='Output PDF file path')
    parser.add_argument('--title', type=str, default='Consistency Check Results',
                       help='Title for the PDF')
    parser.add_argument('--images-per-page', type=int, default=12,
                       help='Number of images per page (default: 12)')
    parser.add_argument('--cols', type=int, default=4,
                       help='Number of columns (default: 4)')
    parser.add_argument('--answer-type', type=str, choices=['number', 'word'],
                       default='number',
                       help='Answer type: "number" for ball drop, "word" for maze')

    args = parser.parse_args()

    judge_file = Path(args.judge_file)
    output_pdf = Path(args.output_pdf)

    if not judge_file.exists():
        print(f"Error: File not found: {judge_file}")
        return

    # Load data
    print(f"Loading consistency data from {judge_file}...")
    entries = load_consistency_data(judge_file, answer_type=args.answer_type)
    print(f"Loaded {len(entries)} valid entries")

    if not entries:
        print("No valid entries found!")
        return

    # Generate PDF
    generate_comparison_pdf(
        entries=entries,
        output_pdf=output_pdf,
        title=args.title,
        images_per_page=args.images_per_page,
        cols=args.cols
    )


if __name__ == '__main__':
    main()
