#!/usr/bin/env python3
"""
Create an HTML table showing model answers for maze validation using pre-graded CSV data.

Usage:
    python3 analysis/maze/visualize_model_answers_html.py [--csv-path=PATH]

The CSV is produced by analysis/maze/create_combined_csv.py and contains the answers,
output snippets, and annotated image paths for every model. This visualizer simply
renders those saved results without re-grading.
"""

import base64
import csv
import html
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path('/Users/log/Github/sketchvlm')
DEFAULT_CSV_PATH = PROJECT_ROOT / 'analysis/maze/maze_v2_combined_results.csv'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'analysis/maze/model_answers_comparison.html'

MODEL_CONFIGS = [
    {'key': 'gemini_flash_sketch', 'label': 'Flash (Sketch)', 'has_annotation': True},
    {'key': 'gemini_pro_sketch', 'label': 'Pro (Sketch)', 'has_annotation': True},
    {'key': 'gemini3_pro_sketch', 'label': 'Pro3 (Sketch)', 'has_annotation': True},
    {'key': 'gemini3_pro_0_1000_sketch', 'label': 'Pro3 0-1000 (Sketch)', 'has_annotation': True},
    {'key': 'gemini_flash_vqa', 'label': 'Flash (Direct VQA)', 'has_annotation': False},
    {'key': 'gemini_pro_vqa', 'label': 'Pro (Direct VQA)', 'has_annotation': False},
    {'key': 'gemini3_pro_vqa', 'label': 'Pro3 (Direct VQA)', 'has_annotation': False},
    {'key': 'qwen3_235b_sketch', 'label': 'Qwen3 (Sketch)', 'has_annotation': True},
    {'key': 'qwen3_235b_vqa', 'label': 'Qwen3 (Direct VQA)', 'has_annotation': False},
    {'key': 'gpt5_med_sketch', 'label': 'GPT-5 Med (Sketch)', 'has_annotation': True},
    {'key': 'gpt5_med_vqa', 'label': 'GPT-5 Med (Direct VQA)', 'has_annotation': False},
    {'key': 'gpt5_low_sketch', 'label': 'GPT-5 Low (Sketch)', 'has_annotation': True},
    {'key': 'gpt5_low_vqa', 'label': 'GPT-5 Low (Direct VQA)', 'has_annotation': False},
    # Show the combined Nano Banana + Gemini column (VQA preferred, then Consistency)
    {'key': 'nanob_gemini3', 'label': 'NanoB + Gemini (Combined)', 'has_annotation': True},
]


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolve a path relative to the project root if needed."""
    if not path_str:
        return None

    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path if path.exists() else None


def image_to_base64(path_str: Optional[str]) -> Optional[str]:
    """Convert an image path into a base64 data URI."""
    resolved = resolve_path(path_str)
    if not resolved:
        return None

    try:
        data = resolved.read_bytes()
        encoded = base64.b64encode(data).decode('utf-8')
        suffix = resolved.suffix.lower()
        if suffix in {'.jpg', '.jpeg'}:
            mime = 'image/jpeg'
        else:
            mime = 'image/png'
        return f'data:{mime};base64,{encoded}'
    except Exception:
        return None


def load_csv_results(csv_path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load the CSV into a maze_id -> {validity: row} mapping."""
    maze_entries: Dict[str, Dict[str, Dict[str, str]]] = {}

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            maze_id = row['maze_id']
            validity = row['validity']
            maze_entries.setdefault(maze_id, {'invalid': None, 'valid': None})[validity] = row

    return maze_entries


def classify_answer(answer: str, expected: str) -> str:
    """Return CSS class for an answer cell."""
    answer_norm = (answer or '').strip().lower()
    expected_norm = (expected or '').strip().lower()

    if not answer or answer_norm in {'unknown', 'missing', ''}:
        return 'unknown'
    if answer_norm == expected_norm:
        return 'correct'
    return 'incorrect'


def render_annotation_cell(path_str: Optional[str], alt_text: str) -> str:
    uri = image_to_base64(path_str)
    if uri:
        return f'<img src="{uri}" alt="{html.escape(alt_text)}" />'
    return '<span style="color: #999;">No image</span>'


def render_maze_image(path_str: Optional[str], maze_id: str) -> str:
    uri = image_to_base64(path_str)
    if uri:
        return f'<img src="{uri}" alt="{html.escape(maze_id)}" />'
    return '<span style="color: #999;">No image</span>'


def build_model_cells(row_data: Dict[str, str], ground_truth: str) -> str:
    cells = []
    for config in MODEL_CONFIGS:
        answer = row_data.get(config['key'], 'missing')
        css_class = classify_answer(answer, ground_truth)
        cells.append(
            f'<td class="answer-cell {css_class}">{html.escape(str(answer))}</td>'
        )

        if config['has_annotation']:
            annotation_path = row_data.get(f"{config['key']}__annotated_image", '')
            annotation_html = render_annotation_cell(
                annotation_path,
                f"{row_data['maze_id']} {config['label']} annotated"
            )
            cells.append(f'<td class="maze-image">{annotation_html}</td>')

        output_tail = row_data.get(f"{config['key']}__output_tail", '')
        cells.append(f'<td class="output-cell">{html.escape(output_tail)}</td>')

    return ''.join(cells)


def build_table_rows(maze_entries: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    rows_html = []

    for maze_id in sorted(maze_entries.keys()):
        entry = maze_entries[maze_id]
        invalid_row = entry.get('invalid')
        valid_row = entry.get('valid')

        available_rows = [row for row in (invalid_row, valid_row) if row]
        if not available_rows:
            continue

        row_span = len(available_rows)
        base_row = invalid_row or valid_row
        image_html = render_maze_image(base_row.get('source_image'), maze_id)

        first_row = True
        for validity in ('invalid', 'valid'):
            row_data = entry.get(validity)
            if not row_data:
                continue

            gt = row_data.get('ground_truth', validity)
            row_cells = ['<tr>']

            if first_row:
                row_cells.append(f'<td class="maze-id" rowspan="{row_span}">{html.escape(maze_id)}</td>')
                row_cells.append(f'<td class="maze-image" rowspan="{row_span}">{image_html}</td>')
                first_row = False

            gt_class = 'gt-valid' if validity == 'valid' else 'gt-invalid'
            row_cells.append(f'<td class="gt-cell {gt_class}">{html.escape(gt)}</td>')
            row_cells.append(build_model_cells(row_data, gt))
            row_cells.append('</tr>')
            rows_html.append(''.join(row_cells))

    return '\n'.join(rows_html)


def build_table_header() -> str:
    header = ['<thead>', '<tr>']
    header.append('<th rowspan="2" style="width: 150px;">Maze ID</th>')
    header.append('<th rowspan="2" style="width: 160px;">Maze Image</th>')
    header.append('<th rowspan="2" style="width: 80px; text-align: center;">GT</th>')

    for config in MODEL_CONFIGS:
        col_span = 3 if config['has_annotation'] else 2
        header.append(f'<th colspan="{col_span}" class="model-col">{config["label"]}</th>')
    header.append('</tr>')

    header.append('<tr>')
    for config in MODEL_CONFIGS:
        header.append('<th class="model-col" style="width: 80px;">Answer</th>')
        if config['has_annotation']:
            header.append('<th class="model-col" style="width: 160px;">Annotated</th>')
        header.append('<th class="model-col" style="width: 180px;">Last 50 chars</th>')
    header.append('</tr>')
    header.append('</thead>')

    return '\n'.join(header)


def compute_model_stats(maze_entries: Dict[str, Dict[str, Dict[str, str]]]):
    """Compute accuracy stats for each model column."""
    stats = {
        cfg['key']: {'label': cfg['label'], 'correct': 0, 'total': 0}
        for cfg in MODEL_CONFIGS
    }

    skip_answers = {'', 'missing', 'unknown'}

    for entry in maze_entries.values():
        for validity in ('invalid', 'valid'):
            row = entry.get(validity)
            if not row:
                continue

            gt_norm = (row.get('ground_truth', validity) or '').strip().lower()

            for cfg in MODEL_CONFIGS:
                answer = row.get(cfg['key'], '')
                answer_norm = (answer or '').strip().lower()
                if answer_norm in skip_answers:
                    continue

                stat = stats[cfg['key']]
                stat['total'] += 1
                if answer_norm == gt_norm:
                    stat['correct'] += 1

    for stat in stats.values():
        total = stat['total']
        stat['accuracy'] = (stat['correct'] / total * 100) if total else 0.0

    return stats


def build_model_stats_block(model_stats: Dict[str, Dict[str, float]]) -> str:
    items = []
    for cfg in MODEL_CONFIGS:
        stat = model_stats.get(cfg['key'])
        if not stat:
            continue
        if stat['total']:
            detail = f"{stat['accuracy']:.1f}% ({stat['correct']}/{stat['total']})"
        else:
            detail = 'No data'
        items.append(
            f'<div class="legend-item"><span style="font-weight: bold;">{cfg["label"]}:</span> {detail}</div>'
        )
    return '\n'.join(items)


def build_html_document(maze_entries: Dict[str, Dict[str, Dict[str, str]]],
                        model_stats: Dict[str, Dict[str, float]]) -> str:
    total_invalid = sum(1 for data in maze_entries.values() if data.get('invalid'))
    total_valid = sum(1 for data in maze_entries.values() if data.get('valid'))
    total = len(maze_entries)

    table_header = build_table_header()
    table_rows = build_table_rows(maze_entries)
    model_stats_block = build_model_stats_block(model_stats)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <title>Maze Model Answers Comparison</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #4472C4;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #2c4a7c;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th.model-col {{
            background-color: #C55A11;
            border: 1px solid #8a3d0b;
        }}
        td {{
            padding: 8px;
            border: 1px solid #ddd;
            font-size: 13px;
        }}
        .maze-id {{
            background-color: #E7E6E6;
            font-weight: 500;
            font-family: monospace;
        }}
        .maze-image {{
            background-color: #f9f9f9;
            text-align: center;
            padding: 5px;
        }}
        .maze-image img {{
            max-width: 150px;
            max-height: 150px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .gt-cell {{
            font-weight: bold;
            text-align: center;
        }}
        .gt-invalid {{
            background-color: #D9E1F2;
        }}
        .gt-valid {{
            background-color: #E2EFDA;
        }}
        .answer-cell {{
            text-align: center;
            font-weight: bold;
        }}
        .correct {{
            background-color: #C6E0B4;
            color: #375623;
        }}
        .incorrect {{
            background-color: #F8CBAD;
            color: #C55A11;
        }}
        .unknown {{
            background-color: #F4B084;
            color: #833C0C;
            font-style: italic;
        }}
        .output-cell {{
            font-family: 'Courier New', monospace;
            font-size: 11px;
            color: #555;
            background-color: #f9f9f9;
            max-width: 250px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }}
        tr:hover {{
            background-color: #f0f0f0;
        }}
        .legend {{
            margin: 20px auto;
            max-width: 800px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #999;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <h1>Maze Model Answers Comparison</h1>
    <div class="stats">
        Total Mazes: {total} ({total_invalid} invalid, {total_valid} valid)
    </div>

    <div class="legend">
        <div class="legend-title">Model Accuracy:</div>
        {model_stats_block}
    </div>

    <div class="legend">
        <div class="legend-title">Legend:</div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #C6E0B4;"></span>
            <span>Correct Answer</span>
        </div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #F8CBAD;"></span>
            <span>Incorrect Answer</span>
        </div>
        <div class="legend-item">
            <span class="legend-box" style="background-color: #F4B084;"></span>
            <span>Unknown/Missing</span>
        </div>
    </div>

    <table>
        {table_header}
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""


def create_html_from_csv(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')

    maze_entries = load_csv_results(csv_path)
    if not maze_entries:
        raise SystemExit(f'No rows found in {csv_path}')

    model_stats = compute_model_stats(maze_entries)
    html_content = build_html_document(maze_entries, model_stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')

    print(f'Saved HTML table to: {output_path}')
    print(f'Total mazes visualized: {len(maze_entries)}')


def parse_args():
    csv_path = DEFAULT_CSV_PATH
    output_path = DEFAULT_OUTPUT_PATH
    index_mode = False

    for arg in sys.argv[1:]:
        if arg == '--index-mode':
            index_mode = True
        elif arg.startswith('--csv-path='):
            csv_path = Path(arg.split('=', 1)[1])
        elif arg.startswith('--output='):
            output_path = Path(arg.split('=', 1)[1])

    if index_mode:
        raise SystemExit('Index mode is not supported by the CSV visualizer yet.')

    return csv_path, output_path


def main():
    csv_path, output_path = parse_args()

    print('=' * 80)
    print('Creating HTML Model Answer Comparison Table (CSV mode)')
    print('=' * 80)
    print()

    create_html_from_csv(csv_path, output_path)

    print()
    print('=' * 80)
    print('Done!')
    print('=' * 80)


if __name__ == '__main__':
    main()
