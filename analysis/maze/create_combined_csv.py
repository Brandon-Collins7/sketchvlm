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


def main():
    print("Building maze to path length mapping...")
    maze_to_path_length = build_maze_to_path_length_mapping()
    print(f"Mapped {len(maze_to_path_length)} mazes from dataset")

    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze_v2')

    # Define all models
    models = [
        ('gemini_flash_sketch', base_path / 'gemini', 'gemini25_flash'),
        ('gemini_flash_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_flash'),
        ('gemini_flash_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_flash'),
        ('gemini_pro_sketch', base_path / 'gemini', 'gemini25_pro'),
        ('gemini_pro_vqa', base_path / 'gemini' / 'direct_vqa', 'gemini25_pro'),
        ('gemini_pro_two_turn', base_path / 'gemini' / 'two_turn', 'gemini25_pro'),
        ('gpt5_low_sketch', base_path / 'gpt5', 'gpt5_low'),
        ('gpt5_low_vqa', base_path / 'gpt5' / 'direct_vqa', 'gpt5_low'),
        ('gpt5_low_two_turn', base_path / 'gpt5' / 'two_turn', 'gpt5_low'),
        ('qwen3_235b_sketch', base_path / 'qwen3', 'qwen3_235b'),
        ('qwen3_235b_vqa', base_path / 'qwen3' / 'direct_vqa', 'qwen3_235b'),
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

    for model_name, model_base, model_prefix in models:
        print(f"  Loading {model_name}...")
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

        for model_name, _, _ in models:
            invalid_row[model_name] = all_results[model_name]['invalid'].get(maze_id, 'missing')

        rows.append(invalid_row)

        # Create row for valid case
        valid_row = {
            'maze_id': maze_id,
            'path_length': path_length,
            'validity': 'valid',
            'ground_truth': 'valid'
        }

        for model_name, _, _ in models:
            valid_row[model_name] = all_results[model_name]['valid'].get(maze_id, 'missing')

        rows.append(valid_row)

    # Write CSV
    output_path = Path('/Users/log/Github/sketchvlm/analysis/maze/maze_v2_combined_results.csv')

    print(f"\nWriting CSV to {output_path}...")

    fieldnames = ['maze_id', 'path_length', 'validity', 'ground_truth']
    for model_name, _, _ in models:
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
    for model_name, _, _ in models:
        invalid_count = sum(1 for row in rows if row['validity'] == 'invalid' and row[model_name] != 'missing')
        valid_count = sum(1 for row in rows if row['validity'] == 'valid' and row[model_name] != 'missing')
        total_invalid = sum(1 for row in rows if row['validity'] == 'invalid')
        total_valid = sum(1 for row in rows if row['validity'] == 'valid')

        print(f"  {model_name}:")
        print(f"    Invalid: {invalid_count}/{total_invalid}")
        print(f"    Valid: {valid_count}/{total_valid}")


if __name__ == '__main__':
    main()
