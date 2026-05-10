#!/usr/bin/env python3
"""
Generate HTML comparison of GPT-5 low/med/high variants.

Shows side-by-side:
- Annotated comparison images
- Model answers
- Metrics (distance, MSE)

Usage:
    python3 create_gpt5_comparison_html.py
"""

import os
import json
import base64
from pathlib import Path
import pandas as pd

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def load_model_json(json_path):
    """Load model JSON file and extract key information."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        return {
            'answer': data.get('answer', 'N/A'),
            'model_output': data.get('model_output', 'N/A'),
        }
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None

def create_comparison_html(output_path='analysis/line_visualizations/gpt5_comparison.html'):
    """Create HTML comparing gpt5_low vs gpt5_med vs gpt5_high."""

    # Paths
    base_dir = Path('/Users/log/Github/sketchvlm')
    comp_dir = base_dir / 'analysis/line_visualizations/comparisons'

    model_low_dir = comp_dir / 'gpt5_low'
    model_med_dir = comp_dir / 'gpt5_med'
    model_high_dir = comp_dir / 'gpt5_high'

    json_low_dir = base_dir / 'results/mix_eval/ball_paths/gpt5/gpt5_low_ball_paths'
    json_med_dir = base_dir / 'results/mix_eval/ball_paths/gpt5/gpt5_med_ball_paths'
    json_high_dir = base_dir / 'results/mix_eval/ball_paths/gpt5/gpt5_high_ball_paths'

    # Load metrics CSVs
    metrics_low = pd.read_csv(model_low_dir / 'metrics_summary.csv')
    metrics_med = pd.read_csv(model_med_dir / 'metrics_summary.csv')
    metrics_high = pd.read_csv(model_high_dir / 'metrics_summary.csv')

    # Start HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GPT-5 Low/Med/High Model Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 10px;
            background-color: #f5f5f5;
            max-width: 100%;
            overflow-x: hidden;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .summary {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary table {
            width: 100%;
            border-collapse: collapse;
        }
        .summary th, .summary td {
            padding: 10px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .summary th {
            background-color: #4CAF50;
            color: white;
        }
        .comparison-row {
            background: white;
            margin: 15px 0;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-row h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .model-comparison {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 5px;
            margin-top: 10px;
        }
        .model-section {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background: #fafafa;
            min-width: 0;
            overflow: hidden;
        }
        .model-section h4 {
            margin: 0 0 5px 0;
            text-align: center;
            font-size: 0.9em;
        }
        .model-section.model-low h4 {
            color: #FF5722;
        }
        .model-section.model-med h4 {
            color: #2196F3;
        }
        .model-section.model-high h4 {
            color: #9C27B0;
        }
        .model-section img {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #ccc;
            display: block;
        }
        .metrics {
            margin-top: 5px;
            padding: 5px;
            background: white;
            border-radius: 3px;
            font-size: 0.7em;
        }
        .metrics p {
            margin: 2px 0;
            word-wrap: break-word;
        }
        .answer {
            margin-top: 5px;
            padding: 5px;
            background: #e3f2fd;
            border-radius: 3px;
            border-left: 3px solid #2196F3;
            font-size: 0.65em;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .answer code {
            word-break: break-all;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>GPT-5 Model Comparison: Low vs Med vs High</h1>
"""

    # Add summary statistics
    avg_low = metrics_low['avg_min_distance'].mean()
    std_low = metrics_low['avg_min_distance'].std()
    mse_low = metrics_low['mse_min_distance'].mean()

    avg_med = metrics_med['avg_min_distance'].mean()
    std_med = metrics_med['avg_min_distance'].std()
    mse_med = metrics_med['mse_min_distance'].mean()

    avg_high = metrics_high['avg_min_distance'].mean()
    std_high = metrics_high['avg_min_distance'].std()
    mse_high = metrics_high['mse_min_distance'].mean()

    html += f"""
    <div class="summary">
        <h2>Overall Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Avg Distance (px)</th>
                <th>MSE (px²)</th>
                <th>N Files</th>
            </tr>
            <tr>
                <td><strong>GPT-5 Low</strong></td>
                <td>{avg_low:.2f} ± {std_low:.2f}</td>
                <td>{mse_low:.0f}</td>
                <td>{len(metrics_low)}</td>
            </tr>
            <tr>
                <td><strong>GPT-5 Med</strong></td>
                <td>{avg_med:.2f} ± {std_med:.2f}</td>
                <td>{mse_med:.0f}</td>
                <td>{len(metrics_med)}</td>
            </tr>
            <tr>
                <td><strong>GPT-5 High</strong></td>
                <td>{avg_high:.2f} ± {std_high:.2f}</td>
                <td>{mse_high:.0f}</td>
                <td>{len(metrics_high)}</td>
            </tr>
        </table>
    </div>
"""

    # Get all comparison images for low model
    comparison_images_low = sorted(model_low_dir.glob('*_comparison.png'))

    print(f"Found {len(comparison_images_low)} comparison images for GPT-5 Low model")

    # Create comparison rows
    for img_low_path in comparison_images_low:
        item_name = img_low_path.stem.replace('_comparison', '')

        # Find corresponding med and high images
        img_med_path = model_med_dir / f'{item_name}_comparison.png'
        img_high_path = model_high_dir / f'{item_name}_comparison.png'

        if not img_med_path.exists() or not img_high_path.exists():
            print(f"Warning: Missing comparison for {item_name}")
            continue

        # Get metrics for this item
        row_low = metrics_low[metrics_low['svg_file'] == item_name]
        row_med = metrics_med[metrics_med['svg_file'] == item_name]
        row_high = metrics_high[metrics_high['svg_file'] == item_name]

        if row_low.empty or row_med.empty or row_high.empty:
            print(f"Warning: No metrics found for {item_name}")
            continue

        dist_low = row_low['avg_min_distance'].values[0]
        mse_low = row_low['mse_min_distance'].values[0]

        dist_med = row_med['avg_min_distance'].values[0]
        mse_med = row_med['mse_min_distance'].values[0]

        dist_high = row_high['avg_min_distance'].values[0]
        mse_high = row_high['mse_min_distance'].values[0]

        # Load model JSON data
        json_low_path = json_low_dir / f'{item_name}.json'
        json_med_path = json_med_dir / f'{item_name}.json'
        json_high_path = json_high_dir / f'{item_name}.json'

        model_low_data = load_model_json(json_low_path)
        model_med_data = load_model_json(json_med_path)
        model_high_data = load_model_json(json_high_path)

        # Encode images
        img_low_b64 = encode_image_to_base64(img_low_path)
        img_med_b64 = encode_image_to_base64(img_med_path)
        img_high_b64 = encode_image_to_base64(img_high_path)

        # Create comparison row
        html += f"""
    <div class="comparison-row">
        <h3>{item_name}</h3>
        <div class="model-comparison">
            <div class="model-section model-low">
                <h4>GPT-5 Low</h4>
                <img src="data:image/png;base64,{img_low_b64}" alt="{item_name} Low">
                <div class="metrics">
                    <p><strong>Avg Distance:</strong> {dist_low:.2f} px</p>
                    <p><strong>MSE:</strong> {mse_low:.0f} px²</p>
                </div>
"""

        if model_low_data:
            html += f"""
                <div class="answer">
                    <strong>Answer:</strong> Bucket {model_low_data['answer']}<br>
                    <strong>Output:</strong> <code>{model_low_data['model_output'][:40]}...</code>
                </div>
"""

        html += """
            </div>

            <div class="model-section model-med">
                <h4>GPT-5 Med</h4>
"""
        html += f"""
                <img src="data:image/png;base64,{img_med_b64}" alt="{item_name} Med">
                <div class="metrics">
                    <p><strong>Avg Distance:</strong> {dist_med:.2f} px</p>
                    <p><strong>MSE:</strong> {mse_med:.0f} px²</p>
                </div>
"""

        if model_med_data:
            html += f"""
                <div class="answer">
                    <strong>Answer:</strong> Bucket {model_med_data['answer']}<br>
                    <strong>Output:</strong> <code>{model_med_data['model_output'][:40]}...</code>
                </div>
"""

        html += """
            </div>

            <div class="model-section model-high">
                <h4>GPT-5 High</h4>
"""
        html += f"""
                <img src="data:image/png;base64,{img_high_b64}" alt="{item_name} High">
                <div class="metrics">
                    <p><strong>Avg Distance:</strong> {dist_high:.2f} px</p>
                    <p><strong>MSE:</strong> {mse_high:.0f} px²</p>
                </div>
"""

        if model_high_data:
            html += f"""
                <div class="answer">
                    <strong>Answer:</strong> Bucket {model_high_data['answer']}<br>
                    <strong>Output:</strong> <code>{model_high_data['model_output'][:40]}...</code>
                </div>
"""

        html += """
            </div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    # Write HTML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML comparison saved to: {output_file}")
    print(f"Open in browser: file://{output_file.absolute()}")

if __name__ == "__main__":
    create_comparison_html()
