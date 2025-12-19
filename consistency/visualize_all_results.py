"""
Generate a single-page HTML visualization combining all model results.
Sorted by inconsistent results first, then by model name.
"""

import os
import json
import base64
import argparse
import re
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(image_path: str) -> str:
    """Get image MIME type."""
    ext = Path(image_path).suffix.lower()
    return {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')


def extract_boxed_answer(text: str) -> str:
    """Extract answer from $\\boxed{...}$ format."""
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


def normalize_answer(answer: str) -> str:
    """Normalize answer to just the number."""
    if not answer:
        return None

    answer = str(answer).strip().lower()

    # Extract just the number
    number_match = re.search(r'\d+', answer)
    if number_match:
        return number_match.group(0)

    # Handle special cases
    if 'none' in answer:
        return 'none'
    if 'multiple' in answer:
        return 'multiple'

    return answer


def is_consistent(entry: dict) -> bool:
    """Check if an entry is consistent."""
    if not entry.get('success', False):
        return False

    original_answer = entry.get('original_extracted_answer', '')
    judge_response = entry.get('consistency_check_response', '')

    judge_answer = extract_boxed_answer(judge_response)

    norm_original = normalize_answer(original_answer)
    norm_judge = normalize_answer(judge_answer)

    if not norm_original or not norm_judge:
        return False

    return norm_original == norm_judge


def load_all_results(judge_dir: str) -> list:
    """Load all judge output files from directory."""
    judge_path = Path(judge_dir)

    if not judge_path.exists():
        print(f"Error: Directory not found: {judge_dir}")
        return []

    # Find all JSON files
    json_files = sorted(judge_path.glob('*.json'))

    all_entries = []
    for json_file in json_files:
        model_name = json_file.stem

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Add model name and consistency status to each entry
            for entry in data:
                entry['model_name'] = model_name
                entry['is_consistent'] = is_consistent(entry)

            all_entries.extend(data)
            print(f"Loaded {len(data)} entries from {model_name}")

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return all_entries


def generate_html(data: list, output_file: str):
    """Generate single HTML page with all results."""

    # Sort by: inconsistent first (is_consistent=False), then by model name
    data.sort(key=lambda x: (x.get('is_consistent', True), x.get('model_name', '')))

    rows = []
    for entry in data:
        index = entry.get('index', 'N/A')
        image_path = entry.get('image_path', '')
        original_answer = entry.get('original_extracted_answer', 'N/A')
        model_name = entry.get('model_name', 'N/A')
        judge_model = entry.get('consistency_check_model', 'N/A')
        consistency_response = entry.get('consistency_check_response', 'N/A')
        is_consistent_flag = entry.get('is_consistent', False)
        success = entry.get('success', False)

        # Encode image
        try:
            if os.path.exists(image_path):
                base64_img = encode_image_to_base64(image_path)
                mime = get_mime_type(image_path)
                img_tag = f'<img src="data:{mime};base64,{base64_img}" style="max-width: 300px; height: auto;">'
            else:
                img_tag = '<p>Image not found</p>'
        except:
            img_tag = '<p>Error loading image</p>'

        # Format text
        if consistency_response:
            consistency_text = consistency_response.replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
        else:
            consistency_text = '<em>No response</em>'

        # Status icon and color
        if not success:
            status = '✗ API Failed'
            status_color = '#ef4444'
        elif is_consistent_flag:
            status = '✓ Consistent'
            status_color = '#10b981'
        else:
            status = '✗ Inconsistent'
            status_color = '#ef4444'

        # Row background color for inconsistent
        row_style = 'background: #fee2e2;' if not is_consistent_flag else ''

        rows.append(f"""
        <tr style="{row_style}">
            <td style="text-align: center; font-weight: bold;">{model_name}</td>
            <td style="text-align: center; font-weight: bold;">{index}</td>
            <td>{img_tag}</td>
            <td style="text-align: center; font-size: 20px; font-weight: bold; color: #667eea;">{original_answer}</td>
            <td style="font-size: 14px; max-width: 400px;">{consistency_text}</td>
            <td style="text-align: center; font-size: 16px; color: {status_color}; font-weight: bold;">{status}</td>
        </tr>
        """)

    # Calculate stats
    total = len(data)
    inconsistent = sum(1 for x in data if not x.get('is_consistent', False))
    consistent = sum(1 for x in data if x.get('is_consistent', False))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>All Results Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1f2937;
        }}
        .stats {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .stat-box {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 14px;
            color: #6b7280;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 15px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
    </style>
</head>
<body>
    <h1>All Models Consistency Check Results</h1>

    <div class="stats">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number">{total}</div>
                <div class="stat-label">Total Entries</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #10b981;">{consistent}</div>
                <div class="stat-label">Consistent</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #ef4444;">{inconsistent}</div>
                <div class="stat-label">Inconsistent</div>
            </div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th style="width: 150px;">Model</th>
                <th style="width: 60px;">Index</th>
                <th style="width: 320px;">Image</th>
                <th style="width: 100px;">Original Answer</th>
                <th>Consistency Check Response</th>
                <th style="width: 120px;">Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nGenerated: {output_file}")
    print(f"Total entries: {total}")
    print(f"Consistent: {consistent}")
    print(f"Inconsistent: {inconsistent}")
    print(f"\nOpen in browser:")
    print(f"  file://{os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description='Generate combined HTML visualization for all models')
    parser.add_argument('--judge-dir', type=str,
                       default='consistency/judge_output',
                       help='Directory containing judge output JSON files')
    parser.add_argument('--output', type=str,
                       default='consistency/html_output/all_models.html',
                       help='Output HTML file path')

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all results
    print(f"Loading results from: {args.judge_dir}")
    all_data = load_all_results(args.judge_dir)

    if not all_data:
        print("No data found!")
        return

    # Generate HTML
    generate_html(all_data, args.output)


if __name__ == '__main__':
    main()
