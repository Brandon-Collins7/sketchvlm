#!/usr/bin/env python3
"""
Create a comprehensive CSV file with all maze results.
Each row represents a unique maze + validity combination.
Columns include ground truth and all model responses.
"""

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def extract_answer_from_response(response_text: str) -> str:
    """Extract answer from model response."""
    if not response_text or response_text.strip() == '':
        return 'unknown'

    # Try <final_answer> tags first
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

    # Try <answer> tags (for ViLaSR)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>',
                            response_text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'
            else:
                return 'valid'
        return 'unknown'

    # Try \boxed{} format (for ViLaSR)
    boxed_match = re.search(r'\$?\\boxed\{(.*?)\}\$?',
                           response_text, re.IGNORECASE | re.DOTALL)
    if boxed_match:
        answer_text = boxed_match.group(1).strip()
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


def extract_maze_id(source_image_path: str) -> str:
    """Extract maze ID from source image path."""
    if not source_image_path:
        return None
    filename = Path(source_image_path).stem
    return filename


def get_path_length_from_maze_id(maze_id: str, maze_to_path_length: Dict[str, int]) -> Optional[int]:
    """Get path length for a maze ID."""
    return maze_to_path_length.get(maze_id)


def build_maze_to_path_length_mapping() -> Dict[str, int]:
    """Build mapping of maze_id to path_length."""
    maze_to_path_length = {}

    for path_length in range(1, 10):  # Check up to path_length_9
        path_dir = Path(f'/Users/log/Github/sketchvlm/datasets/maze_v2/path_length_{path_length}')
        if not path_dir.exists():
            continue
        for maze_dir in path_dir.iterdir():
            if maze_dir.is_dir() and maze_dir.name.startswith('maze_'):
                maze_to_path_length[maze_dir.name] = path_length

    return maze_to_path_length


def load_model_results(base_path: Path, model_prefix: str, validity: str) -> Dict[str, str]:
    """
    Load all results for a model.
    Returns: dict mapping maze_id -> extracted_answer
    """
    results = {}

    results_dir = base_path / f'{model_prefix}_{validity}'
    if not results_dir.exists():
        return results

    json_files = list(results_dir.glob('item_*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            maze_id = extract_maze_id(source_image)

            if maze_id:
                model_output = data.get('model_output_full', '')
                extracted_answer = extract_answer_from_response(model_output)
                results[maze_id] = extracted_answer

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def load_vilasr_jsonl_results(jsonl_path: Path) -> Dict[str, str]:
    """
    Load ViLaSR results from a JSONL file.
    Returns: dict mapping maze_id -> extracted_answer
    """
    results = {}

    if not jsonl_path.exists():
        return results

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Extract maze_id from image_path
                    image_path = data.get('image_path', [])
                    if isinstance(image_path, list) and len(image_path) > 0:
                        maze_id = extract_maze_id(image_path[0])
                    else:
                        maze_id = extract_maze_id(image_path)

                    if maze_id:
                        model_output = data.get('model_output', '')
                        extracted_answer = extract_answer_from_response(model_output)
                        results[maze_id] = extracted_answer

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
                    continue

    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")

    return results


def load_thinkmorph_results(thinkmorph_dir: Path) -> Dict[str, str]:
    """
    Load ThinkMorph results from directories with text_data.json files.
    Returns: dict mapping maze_id -> extracted_answer
    """
    results = {}

    if not thinkmorph_dir.exists():
        return results

    # Process all sample directories
    for sample_dir in thinkmorph_dir.iterdir():
        if not sample_dir.is_dir() or not sample_dir.name.startswith('sample_'):
            continue

        # Extract maze_id from directory name
        # Pattern: sample_YYYYMMDD_HHMMSS_maze_XXX_HASH
        dir_name = sample_dir.name
        parts = dir_name.split('_')

        # Find the index where 'maze' starts
        maze_idx = None
        for i, part in enumerate(parts):
            if part == 'maze':
                maze_idx = i
                break

        if maze_idx is None:
            continue

        # Extract maze_id (everything from 'maze' onwards)
        maze_id = '_'.join(parts[maze_idx:])

        # Look for text_data.json
        json_file = sample_dir / 'text_data.json'
        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract answer from text_outputs
            text_outputs = data.get('text_outputs', [])
            model_output = '\n'.join(text_outputs) if text_outputs else ''
            extracted_answer = extract_answer_from_response(model_output)
            results[maze_id] = extracted_answer

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return results


def main():
    print("Building maze to path length mapping...")
    maze_to_path_length = build_maze_to_path_length_mapping()
    print(f"Mapped {len(maze_to_path_length)} mazes from dataset")

    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze_v2')

    # Define all models
    # Format: (model_name, model_base_path, model_prefix, format_type)
    # format_type: 'json' for standard, 'jsonl' for ViLaSR, 'thinkmorph' for ThinkMorph
    models = [
        ('gemini_flash_sketch', base_path / 'gemini', 'gemini25_flash', 'json'),
        ('gemini_flash_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_flash', 'json'),
        ('gemini_flash_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_flash', 'json'),
        ('gemini_pro_sketch', base_path / 'gemini', 'gemini25_pro', 'json'),
        ('gemini_pro_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_pro', 'json'),
        ('gemini_pro_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_pro', 'json'),
        ('gemini3_pro_sketch', base_path / 'gemini', 'gemini3_pro', 'json'),
        ('gemini3_pro_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini3_pro', 'json'),
        ('gpt5_low_sketch', base_path / 'gpt5', 'gpt5_low', 'json'),
        ('gpt5_low_vqa', base_path / 'gpt5' / 'direct_vqa', 'gpt5_low', 'json'),
        ('gpt5_low_two_turn', base_path / 'gpt5' / 'two_turn', 'gpt5_low', 'json'),
        ('gpt5_low_1000_sketch', base_path / 'gpt5', 'gpt5_low_1000', 'json'),
        ('qwen3_235b_sketch', base_path / 'qwen3', 'qwen3_235b', 'json'),
        ('qwen3_235b_vqa', base_path / 'qwen3' / 'direct_vqa', 'qwen3_235b', 'json'),
        ('qwen25_7b_sketch', base_path / 'qwen25_7b', 'qwen25_7b', 'json'),
        ('qwen25_7b_vqa', base_path / 'qwen25_7b' / 'direct_vqa', 'qwen25_7b', 'json'),
        ('vilasr_sketch', base_path / 'vilasr', 'vilasr', 'jsonl'),
        ('thinkmorph_sketch', base_path / 'thinkmorph', 'thinkmorph', 'thinkmorph'),
    ]

    # Collect all maze IDs from results files (not just dataset)
    print("\nCollecting maze IDs from results...")
    all_maze_ids = set()

    # Scan one model's results to get all maze IDs
    ref_model_base = base_path / 'gemini'
    ref_model_prefix = 'gemini25_flash'

    for validity in ['invalid', 'valid']:
        results_dir = ref_model_base / f'{ref_model_prefix}_{validity}'
        if results_dir.exists():
            for json_file in results_dir.glob('item_*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    source_image = data.get('source_image', '')
                    maze_id = extract_maze_id(source_image)
                    if maze_id:
                        all_maze_ids.add(maze_id)
                except Exception as e:
                    print(f"  Error reading {json_file}: {e}")

    print(f"Found {len(all_maze_ids)} unique mazes in results")

    # Load all model results
    print("\nLoading model results...")
    all_results = {}

    for model_name, model_base, model_prefix, format_type in models:
        print(f"  Loading {model_name}...")
        if format_type == 'jsonl':
            # Load from JSONL files
            invalid_jsonl = model_base / f'{model_prefix}_invalid' / 'results.jsonl'
            valid_jsonl = model_base / f'{model_prefix}_valid' / 'results.jsonl'
            all_results[model_name] = {
                'invalid': load_vilasr_jsonl_results(invalid_jsonl),
                'valid': load_vilasr_jsonl_results(valid_jsonl)
            }
        elif format_type == 'thinkmorph':
            # Load from ThinkMorph directories
            invalid_dir = model_base / f'{model_prefix}_invalid'
            valid_dir = model_base / f'{model_prefix}_valid'
            all_results[model_name] = {
                'invalid': load_thinkmorph_results(invalid_dir),
                'valid': load_thinkmorph_results(valid_dir)
            }
        else:
            # Load from individual JSON files
            all_results[model_name] = {
                'invalid': load_model_results(model_base, model_prefix, 'invalid'),
                'valid': load_model_results(model_base, model_prefix, 'valid')
            }

    # Build CSV rows
    print("\nBuilding CSV data...")
    rows = []

    # Check for mazes without path length info
    mazes_without_path_length = [m for m in all_maze_ids if m not in maze_to_path_length]
    if mazes_without_path_length:
        print(f"  Warning: {len(mazes_without_path_length)} mazes not found in dataset directories")
        print(f"  These will be marked with path_length='unknown'")

    for maze_id in sorted(all_maze_ids):
        path_length = maze_to_path_length.get(maze_id, 'unknown')

        # Create row for invalid case
        invalid_row = {
            'maze_id': maze_id,
            'path_length': path_length,
            'validity': 'invalid',
            'ground_truth': 'invalid'
        }

        for model_name, _, _, _ in models:
            invalid_row[model_name] = all_results.get(model_name, {}).get('invalid', {}).get(maze_id, 'missing')

        rows.append(invalid_row)

        # Create row for valid case
        valid_row = {
            'maze_id': maze_id,
            'path_length': path_length,
            'validity': 'valid',
            'ground_truth': 'valid'
        }

        for model_name, _, _, _ in models:
            valid_row[model_name] = all_results.get(model_name, {}).get('valid', {}).get(maze_id, 'missing')

        rows.append(valid_row)

    # Write CSV
    output_path = Path('/Users/log/Github/sketchvlm/analysis/maze/maze_v2_combined_results.csv')

    print(f"\nWriting CSV to {output_path}...")

    fieldnames = ['maze_id', 'path_length', 'validity', 'ground_truth']
    for model_name, _, _, _ in models:
        fieldnames.append(model_name)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} rows to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total unique mazes: {len(all_maze_ids)}")
    print(f"  Total rows (maze × validity): {len(rows)}")
    print(f"  Models included: {len(models)}")
    print(f"  Columns: {len(fieldnames)}")

    # Check for missing data
    print("\nData completeness check:")
    for model_name, _, _, _ in models:
        invalid_count = sum(1 for row in rows if row['validity'] == 'invalid' and row[model_name] != 'missing')
        valid_count = sum(1 for row in rows if row['validity'] == 'valid' and row[model_name] != 'missing')
        total_invalid = sum(1 for row in rows if row['validity'] == 'invalid')
        total_valid = sum(1 for row in rows if row['validity'] == 'valid')

        print(f"  {model_name}:")
        print(f"    Invalid: {invalid_count}/{total_invalid}")
        print(f"    Valid: {valid_count}/{total_valid}")


if __name__ == '__main__':
    main()
