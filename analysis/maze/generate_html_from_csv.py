#!/usr/bin/env python3
"""
Generate an interactive HTML comparison from the CSV results.
"""

import csv
import json
import base64
from pathlib import Path


def load_annotated_images(base_path, model_prefix, validity):
    """Load mapping of maze_id to annotated image path."""
    mapping = {}
    results_dir = base_path / f'{model_prefix}_{validity}'

    if not results_dir.exists():
        return mapping

    for json_file in results_dir.glob('item_*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            maze_id = Path(source_image).stem

            # The annotated image is in the same directory as the JSON
            item_num = json_file.stem.replace('item_', '')
            annotated_path = results_dir / f'item_{item_num}_annotated.png'

            if maze_id and annotated_path.exists():
                mapping[maze_id] = annotated_path
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return mapping


def image_to_base64(image_path):
    """Convert image file to base64 data URI."""
    try:
        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return f'data:image/png;base64,{data}'
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None


def generate_html(csv_path, output_path):
    """Generate HTML file from CSV data."""

    print("Loading annotated images...")
    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze_v2')

    # Load annotated images for sketch models
    sketch_images = {}
    for model_base, model_prefix in [
        ('gemini', 'gemini25_flash'),
        ('gemini', 'gemini25_pro'),
        ('gemini', 'gemini3_pro'),
        ('gpt5', 'gpt5_low'),
        ('qwen3', 'qwen3_235b')
    ]:
        for validity in ['invalid', 'valid']:
            key = f'{model_prefix}_{validity}'
            print(f"  Loading {key}...")
            sketch_images[key] = load_annotated_images(
                base_path / model_base, model_prefix, validity
            )

    # Load annotated images for two_turn models
    for model_base, model_prefix in [
        ('gemini', 'gemini25_flash'),
        ('gemini', 'gemini25_pro'),
        ('gpt5', 'gpt5_low')
    ]:
        for validity in ['invalid', 'valid']:
            key = f'{model_prefix}_two_turn_{validity}'
            print(f"  Loading {key}...")
            sketch_images[key] = load_annotated_images(
                base_path / model_base / 'two_turn', model_prefix, validity
            )

    # Read CSV data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by maze_id
    maze_data = {}
    for row in rows:
        maze_id = row['maze_id']
        if maze_id not in maze_data:
            maze_data[maze_id] = {
                'path_length': row['path_length'],
                'invalid': None,
                'valid': None
            }

        if row['validity'] == 'invalid':
            maze_data[maze_id]['invalid'] = row
        else:
            maze_data[maze_id]['valid'] = row

    # Model definitions with image columns for sketch models
    models = [
        ('gemini_flash_sketch', 'Flash (Sketch)', True),
        ('gemini_flash_vqa', 'Flash (VQA)', False),
        ('gemini_flash_two_turn', 'Flash (Two-Turn)', True),
        ('gemini_pro_sketch', 'Pro (Sketch)', True),
        ('gemini_pro_vqa', 'Pro (VQA)', False),
        ('gemini_pro_two_turn', 'Pro (Two-Turn)', True),
        ('gemini3_pro_sketch', 'Pro3 (Sketch)', True),
        ('gemini3_pro_vqa', 'Pro3 (VQA)', False),
        ('gpt5_low_sketch', 'GPT-5 (Sketch)', True),
        ('gpt5_low_vqa', 'GPT-5 (VQA)', False),
        ('gpt5_low_two_turn', 'GPT-5 (Two-Turn)', True),
        ('qwen3_235b_sketch', 'Qwen3 (Sketch)', True),
        ('qwen3_235b_vqa', 'Qwen3 (VQA)', False),
    ]

    # Count statistics
    total_mazes = len(maze_data)

    # Calculate overall accuracy for each model
    model_stats = {}
    for model_key, label, has_image in models:
        correct = 0
        total = 0
        for maze_id, data in maze_data.items():
            for validity in ['invalid', 'valid']:
                if data[validity]:
                    row = data[validity]
                    answer = row[model_key]
                    gt = row['ground_truth']
                    total += 1
                    if answer == gt:
                        correct += 1
        accuracy = (correct / total * 100) if total > 0 else 0
        model_stats[label] = {'correct': correct, 'total': total, 'accuracy': accuracy}

    # Start HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Maze V2 - Model Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .stats {
            text-align: center;
            margin-bottom: 20px;
            color: #666;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color: #4472C4;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #2c4a7c;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        th.model-col {
            background-color: #C55A11;
            border: 1px solid #8a3d0b;
        }
        td {
            padding: 8px;
            border: 1px solid #ddd;
            font-size: 13px;
        }
        .maze-id {
            background-color: #E7E6E6;
            font-weight: 500;
            font-family: monospace;
        }
        .gt-cell {
            font-weight: bold;
            text-align: center;
        }
        .gt-invalid {
            background-color: #D9E1F2;
        }
        .gt-valid {
            background-color: #E2EFDA;
        }
        .answer-cell {
            text-align: center;
            font-weight: bold;
        }
        .correct {
            background-color: #C6E0B4;
            color: #375623;
        }
        .incorrect {
            background-color: #F8CBAD;
            color: #C55A11;
        }
        .unknown, .missing {
            background-color: #F4B084;
            color: #833C0C;
            font-style: italic;
        }
        tr:hover {
            background-color: #f0f0f0;
        }
        .legend {
            margin: 20px auto;
            max-width: 800px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
        }
        .legend-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #999;
            vertical-align: middle;
        }
        .path-length {
            text-align: center;
            font-weight: 500;
            background-color: #f9f9f9;
        }
        .maze-image {
            background-color: #f9f9f9;
            text-align: center;
            padding: 5px;
        }
        .maze-image img {
            max-width: 150px;
            max-height: 150px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Maze V2 - Model Answers Comparison</h1>
    <div class="stats">
        Total Mazes: """ + str(total_mazes) + """
    </div>

    <div class="legend">
        <div class="legend-title">Model Accuracy:</div>
"""

    # Add model stats
    for _, label, _ in models:
        stats = model_stats[label]
        html += f"""        <div class="legend-item">
            <span style="font-weight: bold;">{label}:</span> {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})
        </div>
"""

    html += """    </div>

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
        <thead>
            <tr>
                <th rowspan="2" style="width: 150px;">Maze ID</th>
                <th rowspan="2" style="width: 80px;">Path Len</th>
                <th rowspan="2" style="width: 80px; text-align: center;">GT</th>
"""

    # Add model column headers
    for _, label, has_image in models:
        if has_image:
            html += f'                <th class="model-col" colspan="2">{label}</th>\n'
        else:
            html += f'                <th class="model-col" rowspan="2" style="width: 100px;">{label}</th>\n'

    html += """            </tr>
            <tr>
"""

    # Add subheaders for sketch models
    for _, label, has_image in models:
        if has_image:
            html += '                <th class="model-col" style="width: 100px;">Answer</th>\n'
            html += '                <th class="model-col" style="width: 160px;">Annotated</th>\n'

    html += """            </tr>
        </thead>
        <tbody>
"""

    # Add rows for each maze
    for maze_id in sorted(maze_data.keys()):
        data = maze_data[maze_id]
        path_len = data['path_length']

        # Invalid row
        if data['invalid']:
            invalid_row = data['invalid']
            html += f"""            <tr>
                <td class="maze-id" rowspan="2">{maze_id}</td>
                <td class="path-length" rowspan="2">{path_len}</td>
                <td class="gt-cell gt-invalid">invalid</td>
"""

            for model_key, _, has_image in models:
                answer = invalid_row[model_key]
                css_class = 'correct' if answer == 'invalid' else ('unknown' if answer == 'unknown' else 'incorrect')
                html += f'                <td class="answer-cell {css_class}">{answer}</td>\n'

                if has_image:
                    # Map model keys to actual prefixes
                    prefix_map = {
                        'gemini_flash_sketch': 'gemini25_flash',
                        'gemini_flash_two_turn': 'gemini25_flash_two_turn',
                        'gemini_pro_sketch': 'gemini25_pro',
                        'gemini_pro_two_turn': 'gemini25_pro_two_turn',
                        'gemini3_pro_sketch': 'gemini3_pro',
                        'gpt5_low_sketch': 'gpt5_low',
                        'gpt5_low_two_turn': 'gpt5_low_two_turn',
                        'qwen3_235b_sketch': 'qwen3_235b'
                    }
                    prefix = prefix_map.get(model_key, model_key)
                    img_key = f'{prefix}_invalid'
                    if maze_id in sketch_images.get(img_key, {}):
                        img_path = sketch_images[img_key][maze_id]
                        img_data = image_to_base64(img_path)
                        if img_data:
                            html += f'                <td class="maze-image"><img src="{img_data}" /></td>\n'
                        else:
                            html += '                <td class="maze-image">-</td>\n'
                    else:
                        html += '                <td class="maze-image">-</td>\n'

            html += """            </tr>
"""

        # Valid row
        if data['valid']:
            valid_row = data['valid']
            html += f"""            <tr>
                <td class="gt-cell gt-valid">valid</td>
"""

            for model_key, _, has_image in models:
                answer = valid_row[model_key]
                css_class = 'correct' if answer == 'valid' else ('unknown' if answer == 'unknown' else 'incorrect')
                html += f'                <td class="answer-cell {css_class}">{answer}</td>\n'

                if has_image:
                    prefix_map = {
                        'gemini_flash_sketch': 'gemini25_flash',
                        'gemini_flash_two_turn': 'gemini25_flash_two_turn',
                        'gemini_pro_sketch': 'gemini25_pro',
                        'gemini_pro_two_turn': 'gemini25_pro_two_turn',
                        'gemini3_pro_sketch': 'gemini3_pro',
                        'gpt5_low_sketch': 'gpt5_low',
                        'gpt5_low_two_turn': 'gpt5_low_two_turn',
                        'qwen3_235b_sketch': 'qwen3_235b'
                    }
                    prefix = prefix_map.get(model_key, model_key)
                    img_key = f'{prefix}_valid'
                    if maze_id in sketch_images.get(img_key, {}):
                        img_path = sketch_images[img_key][maze_id]
                        img_data = image_to_base64(img_path)
                        if img_data:
                            html += f'                <td class="maze-image"><img src="{img_data}" /></td>\n'
                        else:
                            html += '                <td class="maze-image">-</td>\n'
                    else:
                        html += '                <td class="maze-image">-</td>\n'

            html += """            </tr>
"""

    html += """        </tbody>
    </table>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Generated HTML: {output_path}")
    print(f"Total mazes: {total_mazes}")


if __name__ == '__main__':
    csv_path = Path('/Users/log/Github/sketchvlm/analysis/maze/maze_v2_combined_results.csv')
    output_path = Path('/Users/log/Github/sketchvlm/analysis/maze/maze_v2_comparison.html')

    generate_html(csv_path, output_path)
