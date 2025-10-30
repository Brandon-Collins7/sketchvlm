#!/usr/bin/env python3
"""
Generate HTML comparison of qwen3 8b vs 235b models.

Shows side-by-side:
- Annotated comparison images
- Model answers
- Metrics (distance, MSE)

Usage:
    python3 create_comparison_html.py
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

        # Extract reasoning if available
        reasoning = ""
        if 'provider_debug' in data and 'repr' in data['provider_debug']:
            repr_str = data['provider_debug']['repr']
            # Extract reasoning field from the ChatCompletion repr
            if 'reasoning=' in repr_str:
                reasoning_start = repr_str.find("reasoning='") + len("reasoning='")
                reasoning_end = repr_str.find("')", reasoning_start)
                if reasoning_end == -1:
                    reasoning_end = len(repr_str)
                reasoning = repr_str[reasoning_start:reasoning_end]
                # Truncate long reasoning
                if len(reasoning) > 500:
                    reasoning = reasoning[:500] + "..."

        return {
            'answer': data.get('answer', 'N/A'),
            'model_output': data.get('model_output', 'N/A'),
            'reasoning': reasoning
        }
    except Exception as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None

def create_comparison_html(output_path='analysis/line_visualizations/qwen_comparison.html'):
    """Create HTML comparing qwen3_8b vs qwen3_235b."""

    # Paths
    base_dir = Path('/Users/log/Github/sketchvlm')
    comp_dir = base_dir / 'analysis/line_visualizations/comparisons'

    model_8b_dir = comp_dir / 'qwen3_8b_thinking'
    model_235b_dir = comp_dir / 'qwen3_235b_thinking'

    json_8b_dir = base_dir / 'results/mix_eval/ball_paths/qwen3/qwen3_8b_thinking_ball_paths'
    json_235b_dir = base_dir / 'results/mix_eval/ball_paths/qwen3/qwen3_235b_thinking_ball_paths'

    # Load metrics CSVs
    metrics_8b = pd.read_csv(model_8b_dir / 'metrics_summary.csv')
    metrics_235b = pd.read_csv(model_235b_dir / 'metrics_summary.csv')

    # Start HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Qwen3 8B vs 235B Model Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
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
            margin: 20px 0;
            padding: 20px;
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
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .model-section {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }
        .model-section h4 {
            margin: 0 0 10px 0;
            color: #2196F3;
            text-align: center;
        }
        .model-section.model-8b h4 {
            color: #2196F3;
        }
        .model-section.model-235b h4 {
            color: #FF9800;
        }
        .model-section img {
            width: 100%;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        .metrics {
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .metrics p {
            margin: 5px 0;
        }
        .answer {
            margin-top: 10px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 4px;
            border-left: 4px solid #2196F3;
        }
        .reasoning {
            margin-top: 10px;
            padding: 10px;
            background: #fff3e0;
            border-radius: 4px;
            font-size: 0.85em;
            max-height: 150px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>Qwen3 8B vs 235B: Ball Trajectory Comparison</h1>
"""

    # Add summary statistics
    avg_8b = metrics_8b['avg_min_distance'].mean()
    std_8b = metrics_8b['avg_min_distance'].std()
    mse_8b = metrics_8b['mse_min_distance'].mean()

    avg_235b = metrics_235b['avg_min_distance'].mean()
    std_235b = metrics_235b['avg_min_distance'].std()
    mse_235b = metrics_235b['mse_min_distance'].mean()

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
                <td><strong>Qwen3 8B Thinking</strong></td>
                <td>{avg_8b:.2f} ± {std_8b:.2f}</td>
                <td>{mse_8b:.0f}</td>
                <td>{len(metrics_8b)}</td>
            </tr>
            <tr>
                <td><strong>Qwen3 235B Thinking</strong></td>
                <td>{avg_235b:.2f} ± {std_235b:.2f}</td>
                <td>{mse_235b:.0f}</td>
                <td>{len(metrics_235b)}</td>
            </tr>
        </table>
    </div>
"""

    # Get all comparison images for 8b model
    comparison_images_8b = sorted(model_8b_dir.glob('*_comparison.png'))

    print(f"Found {len(comparison_images_8b)} comparison images for 8B model")

    # Create comparison rows
    for img_8b_path in comparison_images_8b:
        item_name = img_8b_path.stem.replace('_comparison', '')

        # Find corresponding 235b image
        img_235b_path = model_235b_dir / f'{item_name}_comparison.png'

        if not img_235b_path.exists():
            print(f"Warning: No 235B comparison found for {item_name}")
            continue

        # Get metrics for this item
        row_8b = metrics_8b[metrics_8b['svg_file'] == item_name]
        row_235b = metrics_235b[metrics_235b['svg_file'] == item_name]

        if row_8b.empty or row_235b.empty:
            print(f"Warning: No metrics found for {item_name}")
            continue

        dist_8b = row_8b['avg_min_distance'].values[0]
        mse_8b = row_8b['mse_min_distance'].values[0]

        dist_235b = row_235b['avg_min_distance'].values[0]
        mse_235b = row_235b['mse_min_distance'].values[0]

        # Load model JSON data
        json_8b_path = json_8b_dir / f'{item_name}.json'
        json_235b_path = json_235b_dir / f'{item_name}.json'

        model_8b_data = load_model_json(json_8b_path)
        model_235b_data = load_model_json(json_235b_path)

        # Encode images
        img_8b_b64 = encode_image_to_base64(img_8b_path)
        img_235b_b64 = encode_image_to_base64(img_235b_path)

        # Create comparison row
        html += f"""
    <div class="comparison-row">
        <h3>{item_name}</h3>
        <div class="model-comparison">
            <div class="model-section model-8b">
                <h4>Qwen3 8B Thinking</h4>
                <img src="data:image/png;base64,{img_8b_b64}" alt="{item_name} 8B">
                <div class="metrics">
                    <p><strong>Avg Distance:</strong> {dist_8b:.2f} px</p>
                    <p><strong>MSE:</strong> {mse_8b:.0f} px²</p>
                </div>
"""

        if model_8b_data:
            html += f"""
                <div class="answer">
                    <strong>Answer:</strong> Bucket {model_8b_data['answer']}<br>
                    <strong>Output:</strong> <code>{model_8b_data['model_output'][:100]}...</code>
                </div>
"""
            if model_8b_data['reasoning']:
                html += f"""
                <div class="reasoning">
                    <strong>Reasoning:</strong> {model_8b_data['reasoning'][:300]}...
                </div>
"""

        html += """
            </div>

            <div class="model-section model-235b">
                <h4>Qwen3 235B Thinking</h4>
"""
        html += f"""
                <img src="data:image/png;base64,{img_235b_b64}" alt="{item_name} 235B">
                <div class="metrics">
                    <p><strong>Avg Distance:</strong> {dist_235b:.2f} px</p>
                    <p><strong>MSE:</strong> {mse_235b:.0f} px²</p>
                </div>
"""

        if model_235b_data:
            html += f"""
                <div class="answer">
                    <strong>Answer:</strong> Bucket {model_235b_data['answer']}<br>
                    <strong>Output:</strong> <code>{model_235b_data['model_output'][:100]}...</code>
                </div>
"""
            if model_235b_data['reasoning']:
                html += f"""
                <div class="reasoning">
                    <strong>Reasoning:</strong> {model_235b_data['reasoning'][:300]}...
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
