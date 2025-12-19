"""
Generate a single-page HTML visualization of all results.
"""

import os
import json
import base64
import argparse
from pathlib import Path


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(image_path: str) -> str:
    """Get image MIME type."""
    ext = Path(image_path).suffix.lower()
    return {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')


def generate_html(data: list, output_file: str):
    """Generate single HTML page with all results."""

    rows = []
    for entry in data:
        index = entry.get('index', 'N/A')
        image_path = entry.get('image_path', '')
        original_answer = entry.get('original_extracted_answer', 'N/A')
        original_model = entry.get('original_model', 'N/A')
        judge_model = entry.get('consistency_check_model', 'N/A')
        consistency_response = entry.get('consistency_check_response', 'N/A')
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

        status = '✓' if success else '✗'
        status_color = '#10b981' if success else '#ef4444'

        rows.append(f"""
        <tr>
            <td style="text-align: center; font-weight: bold;">{index}</td>
            <td>{img_tag}</td>
            <td style="text-align: center; font-size: 20px; font-weight: bold; color: #667eea;">{original_answer}</td>
            <td style="font-size: 14px; max-width: 400px;">{consistency_text}</td>
            <td style="text-align: center; font-size: 20px; color: {status_color};">{status}</td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Results Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1f2937;
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
    <h1>Consistency Check Results ({len(data)} entries)</h1>
    <table>
        <thead>
            <tr>
                <th style="width: 60px;">Index</th>
                <th style="width: 320px;">Image</th>
                <th style="width: 100px;">Original Answer</th>
                <th>Consistency Check Response</th>
                <th style="width: 80px;">Status</th>
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

    print(f"Generated: {output_file}")
    print(f"Total entries: {len(data)}")
    print(f"\nOpen in browser:")
    print(f"  file://{os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description='Generate simple single-page HTML visualization')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file (default: auto-generated in html_output/)')

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        # Extract model name from input file path
        input_path = Path(args.input)
        input_filename = input_path.stem  # e.g., ball_batch1_gemini3

        # Create html_output directory
        base_dir = input_path.parent  # judge_output directory
        html_output_dir = base_dir.parent / 'html_output'  # Go up to consistency, then html_output
        html_output_dir.mkdir(exist_ok=True)

        output_file = str(html_output_dir / f'{input_filename}.html')
    else:
        output_file = args.output

    with open(args.input, 'r') as f:
        data = json.load(f)

    generate_html(data, output_file)


if __name__ == '__main__':
    main()
