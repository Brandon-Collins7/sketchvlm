#!/usr/bin/env python3
"""
Generate an interactive HTML comparison from the ball physics CSV results.
Shows annotated images for sketch models and answers for all models side by side.
"""

import csv
import json
import base64
from pathlib import Path


def load_annotated_images(results_dir):
    """Load mapping of image name to annotated image path."""
    mapping = {}

    if not results_dir.exists():
        return mapping

    for json_file in results_dir.glob('item_*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            image_name = Path(source_image).name

            # The annotated image is in the same directory as the JSON
            item_num = json_file.stem.replace('item_', '')
            annotated_path = results_dir / f'item_{item_num}_annotated.png'

            if image_name and annotated_path.exists():
                mapping[image_name] = annotated_path
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
    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/ball_paths/batch1')

    # Load annotated images for sketch models
    sketch_images = {}
    model_dirs = {
        'Gemini-2.5-Flash': base_path / 'gemini_25_flash_ball_paths',
        'Gemini-2.5-Pro': base_path / 'gemini_25_pro_ball_paths',
        'GPT-5-low': base_path / 'gpt5_low_ball_paths',
        'GPT-5-med': base_path / 'gpt5_med_ball_paths',
        'Qwen-235B': base_path / 'qwen3_235b_thinking_ball_paths',
        'Qwen-8B': base_path / 'qwen3_8b_thinking_ball_paths',
    }

    for model_name, model_dir in model_dirs.items():
        print(f"  Loading {model_name}...")
        sketch_images[model_name] = load_annotated_images(model_dir)

    # Read CSV data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by image
    image_data = {}
    for row in rows:
        image = row['image']
        model = row['model']
        type_ = row['type']

        if image not in image_data:
            image_data[image] = {
                'gold': int(row['gold']),
                'models': {}
            }

        if model not in image_data[image]['models']:
            image_data[image]['models'][model] = {}

        image_data[image]['models'][model][type_] = row

    # Model definitions
    models = [
        'Gemini-2.5-Flash',
        'Gemini-2.5-Pro',
        'GPT-5-low',
        'GPT-5-med',
        'Qwen-235B',
        'Qwen-8B',
    ]

    # Count statistics
    total_images = len(image_data)

    # Calculate accuracy for each model and type
    model_stats = {}
    for model in models:
        model_stats[model] = {}
        for type_ in ['direct_vqa', 'paths']:
            correct = 0
            total = 0
            for image, data in image_data.items():
                if model in data['models'] and type_ in data['models'][model]:
                    row = data['models'][model][type_]
                    total += 1
                    if row['correct'] == 'True':
                        correct += 1
            accuracy = (correct / total * 100) if total > 0 else 0
            model_stats[model][type_] = {'correct': correct, 'total': total, 'accuracy': accuracy}

    # Start HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Ball Physics - Model Comparison</title>
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
            text-align: center;
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
        .image-cell {
            background-color: #E7E6E6;
            font-weight: 500;
            font-family: monospace;
            text-align: center;
        }
        .gt-cell {
            font-weight: bold;
            text-align: center;
            background-color: #D9E1F2;
            font-size: 16px;
        }
        .answer-cell {
            text-align: center;
            font-weight: bold;
            font-size: 14px;
        }
        .correct {
            background-color: #C6E0B4;
            color: #375623;
        }
        .incorrect {
            background-color: #F8CBAD;
            color: #C55A11;
        }
        .missing {
            background-color: #F4B084;
            color: #833C0C;
            font-style: italic;
        }
        tr:hover {
            background-color: #f0f0f0;
        }
        .legend {
            margin: 20px auto;
            max-width: 1200px;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
            font-size: 13px;
        }
        .legend-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #999;
            vertical-align: middle;
        }
        .sketch-image {
            background-color: #f9f9f9;
            text-align: center;
            padding: 5px;
        }
        .sketch-image img {
            max-width: 180px;
            max-height: 180px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }
        .response-cell {
            max-width: 200px;
            font-size: 11px;
            white-space: pre-wrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .response-text {
            max-height: 120px;
            overflow-y: auto;
            padding: 5px;
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            text-align: left;
        }
        .boxed-answer {
            font-family: monospace;
            font-size: 12px;
            color: #444;
            margin-top: 3px;
        }
    </style>
</head>
<body>
    <h1>Ball Physics - Model Answers Comparison</h1>
    <div class="stats">
        Total Images: """ + str(total_images) + """
    </div>

    <div class="legend">
        <div class="legend-title">Model Accuracy:</div>
"""

    # Add model stats
    for model in models:
        for type_ in ['direct_vqa', 'paths']:
            type_label = 'Direct VQA' if type_ == 'direct_vqa' else 'SketchVLM'
            stats = model_stats[model][type_]
            if stats['total'] > 0:
                html += f"""        <div class="legend-item">
            <span style="font-weight: bold;">{model} ({type_label}):</span> {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})
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
            <span>Missing Data</span>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th rowspan="2" style="width: 120px;">Image</th>
                <th rowspan="2" style="width: 60px;">Gold</th>
"""

    # Add model column headers
    for model in models:
        html += f'                <th class="model-col" colspan="4">{model}</th>\n'

    html += """            </tr>
            <tr>
"""

    # Add subheaders for each model (Direct VQA and SketchVLM)
    for model in models:
        html += '                <th class="model-col" style="width: 70px;">Direct VQA</th>\n'
        html += '                <th class="model-col" style="width: 200px;">VQA Response</th>\n'
        html += '                <th class="model-col" style="width: 70px;">SketchVLM</th>\n'
        html += '                <th class="model-col" style="width: 180px;">Sketch</th>\n'

    html += """            </tr>
        </thead>
        <tbody>
"""

    # Add rows for each image
    for image in sorted(image_data.keys()):
        data = image_data[image]
        gold = data['gold']

        html += f"""            <tr>
                <td class="image-cell">{image.replace('.png', '')}</td>
                <td class="gt-cell">{gold}</td>
"""

        for model in models:
            # Direct VQA answer
            if model in data['models'] and 'direct_vqa' in data['models'][model]:
                row = data['models'][model]['direct_vqa']
                answer = row['prediction']
                css_class = 'correct' if row['correct'] == 'True' else 'incorrect'
                html += f'                <td class="answer-cell {css_class}">{answer}</td>\n'

                # Direct VQA response
                response = row.get('model_output', '')
                boxed = row.get('boxed_answer', '')
                # Truncate response for display
                display_response = response[:300] + '...' if len(response) > 300 else response
                html += f'                <td class="response-cell"><div class="boxed-answer">{boxed}</div><div class="response-text">{display_response}</div></td>\n'
            else:
                html += '                <td class="missing">-</td>\n'
                html += '                <td class="missing">-</td>\n'

            # SketchVLM answer
            if model in data['models'] and 'paths' in data['models'][model]:
                row = data['models'][model]['paths']
                answer = row['prediction']
                css_class = 'correct' if row['correct'] == 'True' else 'incorrect'
                html += f'                <td class="answer-cell {css_class}">{answer}</td>\n'

                # SketchVLM annotated image
                if image in sketch_images.get(model, {}):
                    img_path = sketch_images[model][image]
                    img_data = image_to_base64(img_path)
                    if img_data:
                        html += f'                <td class="sketch-image"><img src="{img_data}" /></td>\n'
                    else:
                        html += '                <td class="sketch-image">-</td>\n'
                else:
                    html += '                <td class="sketch-image">-</td>\n'
            else:
                html += '                <td class="missing">-</td>\n'
                html += '                <td class="missing">-</td>\n'

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

    print(f"\n✓ Generated HTML: {output_path}")
    print(f"  Total images: {total_images}")

    # Print accuracy summary
    print("\n" + "="*80)
    print("ACCURACY SUMMARY")
    print("="*80)
    for model in models:
        print(f"\n{model}:")
        for type_ in ['direct_vqa', 'paths']:
            type_label = 'Direct VQA' if type_ == 'direct_vqa' else 'SketchVLM'
            stats = model_stats[model][type_]
            if stats['total'] > 0:
                print(f"  {type_label:12s}: {stats['accuracy']:5.1f}% ({stats['correct']:2d}/{stats['total']:2d})")


if __name__ == '__main__':
    csv_path = Path('/Users/log/Github/sketchvlm/analysis/ball_physics/batch1_all_models_results.csv')
    output_path = Path('/Users/log/Github/sketchvlm/analysis/ball_physics/batch1_comparison.html')

    generate_html(csv_path, output_path)
