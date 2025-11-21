#!/usr/bin/env python3
"""
Create an HTML table showing all model answers for maze validation.
Includes the last ~50 characters of model output for verification.

Example usage:
    python3 analysis/maze/visualize_model_answers_html.py [--index-mode]
"""

import json
import re
import base64
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import html


def extract_maze_id(source_image_path: str) -> str:
    """Extract maze ID from source image path."""
    if not source_image_path:
        return None
    filename = Path(source_image_path).stem
    return filename


def extract_answer_from_response(response_text: str) -> str:
    """Extract answer from model response."""
    if not response_text or response_text.strip() == '':
        return 'unknown'

    # Try to extract from <final_answer> tags first
    final_answer_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>',
                                   response_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        answer_text = final_answer_match.group(1).strip()
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'
            else:
                return 'valid'
        return 'unknown'

    # Fallback: check last 30 characters
    last_chars = response_text[-30:].lower()
    if 'invalid' in last_chars:
        return 'invalid'
    elif 'valid' in last_chars:
        return 'valid'

    return 'unknown'


def extract_index_answer_from_response(response_text: str):
    """Extract answer from model response for index-based evaluation."""
    if not response_text or response_text.strip() == '':
        return 'unknown'

    # Try to extract from <final_answer> tags first
    final_answer_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', response_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        answer_text = final_answer_match.group(1).strip()

        # Check for "valid" first
        if answer_text.lower() == 'valid':
            return 'valid'

        # Try to extract number from $\boxed{N}$ notation
        boxed_match = re.search(r'\$\\boxed\{(\d+)\}\$', answer_text)
        if boxed_match:
            return int(boxed_match.group(1))

        # Try without the $ signs
        boxed_match = re.search(r'\\boxed\{(\d+)\}', answer_text)
        if boxed_match:
            return int(boxed_match.group(1))

        # Try to find any number in the answer
        number_match = re.search(r'\b(\d+)\b', answer_text)
        if number_match:
            return int(number_match.group(1))

    # If no <final_answer> tags, look for boxed notation anywhere in the text
    boxed_matches = list(re.finditer(r'\$\\boxed\{(\d+)\}\$', response_text))
    if boxed_matches:
        return int(boxed_matches[-1].group(1))

    # Try without the $ signs
    boxed_matches = list(re.finditer(r'\\boxed\{(\d+)\}', response_text))
    if boxed_matches:
        return int(boxed_matches[-1].group(1))

    # Check for "valid" in boxed notation
    if re.search(r'\$\\boxed\{valid\}\$', response_text, re.IGNORECASE):
        return 'valid'
    if re.search(r'\\boxed\{valid\}', response_text, re.IGNORECASE):
        return 'valid'

    # Fallback: check for "valid" in the text (but not if "invalid" is present)
    if 'valid' in response_text.lower() and 'invalid' not in response_text.lower():
        return 'valid'

    return 'unknown'


def load_results_with_output(parent_dir: str, model_name: str, expected_answer: str, index_mode: bool = True, base_dir: str = 'gemini') -> Dict[str, Tuple[Union[str, int], str, str, str, str]]:
    """
    Load results from a directory and return a dict mapping maze_id to (gt_answer, extracted_answer, last_50_chars, image_path, annotated_path).

    Args:
        parent_dir: Parent directory name (e.g., '.', 'direct_vqa')
        model_name: Model name (e.g., 'gemini25_flash', 'gemini25_pro', 'qwen3_235b')
        expected_answer: Expected answer ('valid' or 'invalid')
        index_mode: If True, use index-based evaluation mode
        base_dir: Base directory name (e.g., 'gemini', 'gpt5', 'qwen3')

    Returns:
        Dictionary mapping maze_id to (gt_answer, extracted_answer, last_50_chars, image_path, annotated_path)
    """
    if index_mode:
        base_path = Path(f'/Users/log/Github/sketchvlm/results/mix_eval/maze_v2/{base_dir}/index')
    else:
        base_path = Path(f'/Users/log/Github/sketchvlm/results/mix_eval/maze_v2/{base_dir}')

    if parent_dir == '.':
        dir_path = base_path / f'{model_name}_{expected_answer}'
    else:
        dir_path = base_path / parent_dir / f'{model_name}_{expected_answer}'

    results = {}

    if not dir_path.exists():
        return results

    # Build path length mapping once for efficiency
    maze_to_path_length = {}
    for pl in range(1, 8):
        pl_dir = Path(f'/Users/log/Github/sketchvlm/datasets/maze_v1/path_length_{pl}')
        if pl_dir.exists():
            for mdir in pl_dir.iterdir():
                if mdir.is_dir() and mdir.name.startswith('maze_'):
                    maze_to_path_length[mdir.name] = pl

    for json_file in sorted(dir_path.glob('item_*.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            maze_id = extract_maze_id(source_image)
            model_output = data.get('model_output_full', '')

            # Extract answer based on mode
            if index_mode:
                answer = extract_index_answer_from_response(model_output)
            else:
                answer = extract_answer_from_response(model_output)

            # Get ground truth
            if expected_answer == 'valid':
                gt_answer = 'valid'
            else:
                # For invalid paths
                if index_mode:
                    # Read from metadata.json for index mode
                    path_length = maze_to_path_length.get(maze_id)
                    if path_length is not None:
                        metadata_path = Path(f'/Users/log/Github/sketchvlm/datasets/maze_v1/path_length_{path_length}/{maze_id}/metadata.json')
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as meta_file:
                                metadata = json.load(meta_file)
                                gt_answer = metadata.get('incorrect_paths', {}).get('substitution', {}).get('modified_index')
                        else:
                            gt_answer = None
                    else:
                        gt_answer = None
                else:
                    # Binary mode: ground truth is just 'invalid'
                    gt_answer = 'invalid'

            # Get last 50 characters
            last_chars = model_output[-50:] if model_output else ''

            # Get annotated image path (for sketch-based models)
            annotated_path = None
            annotated_file = json_file.parent / f"{json_file.stem}_annotated.png"
            if annotated_file.exists():
                # Make path relative to project root
                annotated_path = str(annotated_file.relative_to(Path('/Users/log/Github/sketchvlm')))

            if maze_id:
                results[maze_id] = (gt_answer, answer, last_chars, source_image, annotated_path)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def image_to_base64(image_path: str) -> Optional[str]:
    """
    Convert an image file to a base64 data URI.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded data URI or None if error
    """
    try:
        # Resolve the path relative to the project root
        full_path = Path('/Users/log/Github/sketchvlm') / image_path

        if not full_path.exists():
            return None

        with open(full_path, 'rb') as f:
            image_data = f.read()
            encoded = base64.b64encode(image_data).decode('utf-8')

            # Determine the image format
            suffix = full_path.suffix.lower()
            if suffix == '.png':
                mime_type = 'image/png'
            elif suffix in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # default

            return f"data:{mime_type};base64,{encoded}"

    except Exception as e:
        print(f"Error converting image {image_path} to base64: {e}")
        return None


def create_html_table(index_mode: bool = True):
    """Create an HTML table showing all model answers with output snippets."""

    print("Loading results...")

    # Load all results for all models
    flash_sketch_invalid = load_results_with_output('.', 'gemini25_flash', 'invalid', index_mode, 'gemini')
    flash_sketch_valid = load_results_with_output('.', 'gemini25_flash', 'valid', index_mode, 'gemini')

    pro_sketch_invalid = load_results_with_output('.', 'gemini25_pro', 'invalid', index_mode, 'gemini')
    pro_sketch_valid = load_results_with_output('.', 'gemini25_pro', 'valid', index_mode, 'gemini')

    pro3_sketch_invalid = load_results_with_output('.', 'gemini3_pro', 'invalid', index_mode, 'gemini')
    pro3_sketch_valid = load_results_with_output('.', 'gemini3_pro', 'valid', index_mode, 'gemini')

    flash_vqa_invalid = load_results_with_output('direct_vqa', 'gemini25_flash', 'invalid', index_mode, 'gemini')
    flash_vqa_valid = load_results_with_output('direct_vqa', 'gemini25_flash', 'valid', index_mode, 'gemini')

    pro_vqa_invalid = load_results_with_output('direct_vqa', 'gemini25_pro', 'invalid', index_mode, 'gemini')
    pro_vqa_valid = load_results_with_output('direct_vqa', 'gemini25_pro', 'valid', index_mode, 'gemini')

    pro3_vqa_invalid = load_results_with_output('direct_vqa', 'gemini3_pro', 'invalid', index_mode, 'gemini')
    pro3_vqa_valid = load_results_with_output('direct_vqa', 'gemini3_pro', 'valid', index_mode, 'gemini')

    # Load Qwen3 results
    qwen3_sketch_invalid = load_results_with_output('.', 'qwen3_235b', 'invalid', index_mode, 'qwen3')
    qwen3_sketch_valid = load_results_with_output('.', 'qwen3_235b', 'valid', index_mode, 'qwen3')

    qwen3_vqa_invalid = load_results_with_output('direct_vqa', 'qwen3_235b', 'invalid', index_mode, 'qwen3')
    qwen3_vqa_valid = load_results_with_output('direct_vqa', 'qwen3_235b', 'valid', index_mode, 'qwen3')

    # Combine all maze IDs with their ground truth and image paths
    all_mazes = []

    # Add invalid mazes
    invalid_mazes = sorted(set(flash_sketch_invalid.keys()) | set(pro_sketch_invalid.keys()) |
                          set(pro3_sketch_invalid.keys()) | set(flash_vqa_invalid.keys()) |
                          set(pro_vqa_invalid.keys()) | set(pro3_vqa_invalid.keys()) |
                          set(qwen3_sketch_invalid.keys()) | set(qwen3_vqa_invalid.keys()))
    for maze_id in invalid_mazes:
        # Get ground truth and image path from any available source
        gt_answer = None
        image_path = None
        for results_dict in [flash_sketch_invalid, pro_sketch_invalid, pro3_sketch_invalid,
                            flash_vqa_invalid, pro_vqa_invalid, pro3_vqa_invalid,
                            qwen3_sketch_invalid, qwen3_vqa_invalid]:
            if maze_id in results_dict:
                gt_answer = results_dict[maze_id][0]  # first element is gt_answer
                image_path = results_dict[maze_id][3]  # fourth element is image path
                break
        all_mazes.append((maze_id, gt_answer, flash_sketch_invalid, pro_sketch_invalid, pro3_sketch_invalid,
                         flash_vqa_invalid, pro_vqa_invalid, pro3_vqa_invalid,
                         qwen3_sketch_invalid, qwen3_vqa_invalid, image_path))

    # Add valid mazes
    valid_mazes = sorted(set(flash_sketch_valid.keys()) | set(pro_sketch_valid.keys()) |
                        set(pro3_sketch_valid.keys()) | set(flash_vqa_valid.keys()) |
                        set(pro_vqa_valid.keys()) | set(pro3_vqa_valid.keys()) |
                        set(qwen3_sketch_valid.keys()) | set(qwen3_vqa_valid.keys()))
    for maze_id in valid_mazes:
        # Get image path from any available source
        image_path = None
        for results_dict in [flash_sketch_valid, pro_sketch_valid, pro3_sketch_valid,
                            flash_vqa_valid, pro_vqa_valid, pro3_vqa_valid,
                            qwen3_sketch_valid, qwen3_vqa_valid]:
            if maze_id in results_dict:
                image_path = results_dict[maze_id][3]  # fourth element is image path
                break
        all_mazes.append((maze_id, 'valid', flash_sketch_valid, pro_sketch_valid, pro3_sketch_valid,
                         flash_vqa_valid, pro_vqa_valid, pro3_vqa_valid,
                         qwen3_sketch_valid, qwen3_vqa_valid, image_path))

    print(f"Found {len(invalid_mazes)} invalid mazes and {len(valid_mazes)} valid mazes")
    print(f"Creating HTML table with {len(all_mazes)} total rows...")

    # Generate HTML
    html_content = generate_html(all_mazes)

    # Write to file
    if index_mode:
        output_dir = Path('/Users/log/Github/sketchvlm/analysis/maze')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'model_answers_comparison_index.html'
    else:
        output_dir = Path('/Users/log/Github/sketchvlm/analysis/maze')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'model_answers_comparison.html'

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Saved HTML table to: {output_path}")


def generate_html(all_mazes: List[Tuple]) -> str:
    """Generate the HTML content."""

    # Count stats first
    invalid_count = sum(1 for _, gt, _, _, _, _, _, _, _, _, _ in all_mazes if gt != 'valid')
    valid_count = sum(1 for _, gt, _, _, _, _, _, _, _, _, _ in all_mazes if gt == 'valid')

    html_start = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Maze Index Identification - Model Answers Comparison</title>
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
        .model-section {{
            border-left: 3px solid #C55A11;
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
    <h1>Maze Index Identification - Model Answers Comparison</h1>
    <div class="stats">
        Total Mazes: {len(all_mazes)} ({invalid_count} invalid, {valid_count} valid)<br>
        Invalid paths: Ground truth is the index of the incorrect move (0-based)
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
            <span>Unknown/Unparseable</span>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th rowspan="2" style="width: 150px;">Maze ID</th>
                <th rowspan="2" style="width: 160px;">Maze Image</th>
                <th rowspan="2" style="width: 80px; text-align: center;">GT</th>
                <th colspan="3" class="model-col">Flash (Sketch)</th>
                <th colspan="3" class="model-col">Pro (Sketch)</th>
                <th colspan="3" class="model-col">Pro3 (Sketch)</th>
                <th colspan="2" class="model-col">Flash (Direct VQA)</th>
                <th colspan="2" class="model-col">Pro (Direct VQA)</th>
                <th colspan="2" class="model-col">Pro3 (Direct VQA)</th>
                <th colspan="3" class="model-col">Qwen3 (Sketch)</th>
                <th colspan="2" class="model-col">Qwen3 (Direct VQA)</th>
            </tr>
            <tr>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 160px;">Annotated</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 160px;">Annotated</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 160px;">Annotated</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 160px;">Annotated</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
                <th class="model-col" style="width: 80px;">Answer</th>
                <th class="model-col" style="width: 180px;">Last 50 chars</th>
            </tr>
        </thead>
        <tbody>
"""

    html_rows = []

    for maze_id, gt, flash_sketch_results, pro_sketch_results, pro3_sketch_results, flash_vqa_results, pro_vqa_results, pro3_vqa_results, qwen3_sketch_results, qwen3_vqa_results, image_path in all_mazes:
        # Get results or defaults (gt_answer, extracted_answer, last_chars, image_path, annotated_path)
        flash_sketch_data = flash_sketch_results.get(maze_id, (None, 'N/A', '', '', None))
        pro_sketch_data = pro_sketch_results.get(maze_id, (None, 'N/A', '', '', None))
        pro3_sketch_data = pro3_sketch_results.get(maze_id, (None, 'N/A', '', '', None))
        flash_vqa_data = flash_vqa_results.get(maze_id, (None, 'N/A', '', '', None))
        pro_vqa_data = pro_vqa_results.get(maze_id, (None, 'N/A', '', '', None))
        pro3_vqa_data = pro3_vqa_results.get(maze_id, (None, 'N/A', '', '', None))
        qwen3_sketch_data = qwen3_sketch_results.get(maze_id, (None, 'N/A', '', '', None))
        qwen3_vqa_data = qwen3_vqa_results.get(maze_id, (None, 'N/A', '', '', None))

        _, flash_sketch_answer, flash_sketch_output, _, flash_sketch_annotated = flash_sketch_data
        _, pro_sketch_answer, pro_sketch_output, _, pro_sketch_annotated = pro_sketch_data
        _, pro3_sketch_answer, pro3_sketch_output, _, pro3_sketch_annotated = pro3_sketch_data
        _, flash_vqa_answer, flash_vqa_output, _, _ = flash_vqa_data
        _, pro_vqa_answer, pro_vqa_output, _, _ = pro_vqa_data
        _, pro3_vqa_answer, pro3_vqa_output, _, _ = pro3_vqa_data
        _, qwen3_sketch_answer, qwen3_sketch_output, _, qwen3_sketch_annotated = qwen3_sketch_data
        _, qwen3_vqa_answer, qwen3_vqa_output, _, _ = qwen3_vqa_data

        # Convert main maze image to base64 if available
        image_html = ''
        if image_path:
            image_data_uri = image_to_base64(image_path)
            if image_data_uri:
                image_html = f'<img src="{image_data_uri}" alt="{maze_id}" />'
            else:
                image_html = '<span style="color: #999;">No image</span>'
        else:
            image_html = '<span style="color: #999;">No image</span>'

        # Convert Flash Sketch annotated image to base64
        flash_sketch_annotated_html = ''
        if flash_sketch_annotated:
            flash_annotated_uri = image_to_base64(flash_sketch_annotated)
            if flash_annotated_uri:
                flash_sketch_annotated_html = f'<img src="{flash_annotated_uri}" alt="{maze_id} Flash Sketch annotated" />'
            else:
                flash_sketch_annotated_html = '<span style="color: #999;">No image</span>'
        else:
            flash_sketch_annotated_html = '<span style="color: #999;">No image</span>'

        # Convert Pro Sketch annotated image to base64
        pro_sketch_annotated_html = ''
        if pro_sketch_annotated:
            pro_annotated_uri = image_to_base64(pro_sketch_annotated)
            if pro_annotated_uri:
                pro_sketch_annotated_html = f'<img src="{pro_annotated_uri}" alt="{maze_id} Pro Sketch annotated" />'
            else:
                pro_sketch_annotated_html = '<span style="color: #999;">No image</span>'
        else:
            pro_sketch_annotated_html = '<span style="color: #999;">No image</span>'

        # Convert Pro3 Sketch annotated image to base64
        pro3_sketch_annotated_html = ''
        if pro3_sketch_annotated:
            pro3_annotated_uri = image_to_base64(pro3_sketch_annotated)
            if pro3_annotated_uri:
                pro3_sketch_annotated_html = f'<img src="{pro3_annotated_uri}" alt="{maze_id} Pro3 Sketch annotated" />'
            else:
                pro3_sketch_annotated_html = '<span style="color: #999;">No image</span>'
        else:
            pro3_sketch_annotated_html = '<span style="color: #999;">No image</span>'

        # Convert Qwen3 Sketch annotated image to base64
        qwen3_sketch_annotated_html = ''
        if qwen3_sketch_annotated:
            qwen3_annotated_uri = image_to_base64(qwen3_sketch_annotated)
            if qwen3_annotated_uri:
                qwen3_sketch_annotated_html = f'<img src="{qwen3_annotated_uri}" alt="{maze_id} Qwen3 Sketch annotated" />'
            else:
                qwen3_sketch_annotated_html = '<span style="color: #999;">No image</span>'
        else:
            qwen3_sketch_annotated_html = '<span style="color: #999;">No image</span>'

        # Determine cell classes
        def get_class(answer, gt):
            if answer == 'N/A':
                return 'unknown'
            elif answer == gt:
                return 'correct'
            elif answer == 'unknown':
                return 'unknown'
            else:
                return 'incorrect'

        flash_sketch_class = get_class(flash_sketch_answer, gt)
        pro_sketch_class = get_class(pro_sketch_answer, gt)
        pro3_sketch_class = get_class(pro3_sketch_answer, gt)
        flash_vqa_class = get_class(flash_vqa_answer, gt)
        pro_vqa_class = get_class(pro_vqa_answer, gt)
        pro3_vqa_class = get_class(pro3_vqa_answer, gt)
        qwen3_sketch_class = get_class(qwen3_sketch_answer, gt)
        qwen3_vqa_class = get_class(qwen3_vqa_answer, gt)

        # Format answers for display
        def format_answer(ans):
            if isinstance(ans, int):
                return str(ans)
            return str(ans)

        row = f"""            <tr>
                <td class="maze-id">{html.escape(maze_id)}</td>
                <td class="maze-image">{image_html}</td>
                <td class="gt-cell">{html.escape(format_answer(gt))}</td>
                <td class="answer-cell {flash_sketch_class}">{html.escape(format_answer(flash_sketch_answer))}</td>
                <td class="maze-image">{flash_sketch_annotated_html}</td>
                <td class="output-cell">{html.escape(flash_sketch_output)}</td>
                <td class="answer-cell {pro_sketch_class}">{html.escape(format_answer(pro_sketch_answer))}</td>
                <td class="maze-image">{pro_sketch_annotated_html}</td>
                <td class="output-cell">{html.escape(pro_sketch_output)}</td>
                <td class="answer-cell {pro3_sketch_class}">{html.escape(format_answer(pro3_sketch_answer))}</td>
                <td class="maze-image">{pro3_sketch_annotated_html}</td>
                <td class="output-cell">{html.escape(pro3_sketch_output)}</td>
                <td class="answer-cell {flash_vqa_class}">{html.escape(format_answer(flash_vqa_answer))}</td>
                <td class="output-cell">{html.escape(flash_vqa_output)}</td>
                <td class="answer-cell {pro_vqa_class}">{html.escape(format_answer(pro_vqa_answer))}</td>
                <td class="output-cell">{html.escape(pro_vqa_output)}</td>
                <td class="answer-cell {pro3_vqa_class}">{html.escape(format_answer(pro3_vqa_answer))}</td>
                <td class="output-cell">{html.escape(pro3_vqa_output)}</td>
                <td class="answer-cell {qwen3_sketch_class}">{html.escape(format_answer(qwen3_sketch_answer))}</td>
                <td class="maze-image">{qwen3_sketch_annotated_html}</td>
                <td class="output-cell">{html.escape(qwen3_sketch_output)}</td>
                <td class="answer-cell {qwen3_vqa_class}">{html.escape(format_answer(qwen3_vqa_answer))}</td>
                <td class="output-cell">{html.escape(qwen3_vqa_output)}</td>
            </tr>
"""
        html_rows.append(row)

    html_end = """        </tbody>
    </table>
</body>
</html>
"""

    return html_start + ''.join(html_rows) + html_end


def main():
    # Check for --index-mode flag
    index_mode = '--index-mode' in sys.argv

    print("=" * 80)
    if index_mode:
        print("Creating HTML Model Answer Comparison Table (Index Mode)")
    else:
        print("Creating HTML Model Answer Comparison Table (Binary Mode)")
    print("=" * 80)
    print()

    create_html_table(index_mode)

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
