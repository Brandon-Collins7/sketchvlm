#!/usr/bin/env python3
"""
Bootstrap analysis for maze results.
Randomly partitions results into K groups and reports mean/std accuracy.

Example usage:
    python3 analysis/maze/analyze_maze_bootstrap.py ":gemini3pro_gridworld_paths_0_to_1000:Gemini 0_1000"
"""

import json
import re
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List

# Reuse extraction logic from the main analysis script
from analyze_maze_results import (
    extract_maze_id,
    extract_answer_from_response,
)

K = 4
GROUP_SIZE = 100


def load_results(dir_path: Path, expected_answer: str) -> List[Dict]:
    """Load all item_*.json results from a directory and evaluate correctness."""
    items = []
    for json_file in sorted(dir_path.glob('item_*.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            model_output = data.get('model_output_full', '')
            extracted = extract_answer_from_response(model_output)
            items.append({
                'file': json_file.name,
                'expected': expected_answer,
                'extracted': extracted,
                'correct': extracted == expected_answer,
            })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    return items


def load_jsonl_results(jsonl_path: Path, expected_answer: str) -> List[Dict]:
    """Load results from a JSONL file (e.g. ViLaSR)."""
    items = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                model_output = data.get('model_output', '')
                extracted = extract_answer_from_response(model_output)
                items.append({
                    'file': data.get('image_path', ''),
                    'expected': expected_answer,
                    'extracted': extracted,
                    'correct': extracted == expected_answer,
                })
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
    return items


def load_thinkmorph_results(dir_path: Path, expected_answer: str) -> List[Dict]:
    """Load results from ThinkMorph sample directories with text_data.json."""
    items = []
    for sample_dir in sorted(dir_path.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith('sample_'):
            continue
        json_file = sample_dir / 'text_data.json'
        if not json_file.exists():
            continue
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            text_outputs = data.get('text_outputs', [])
            model_output = '\n'.join(text_outputs) if text_outputs else ''
            extracted = extract_answer_from_response(model_output)
            items.append({
                'file': sample_dir.name,
                'expected': expected_answer,
                'extracted': extracted,
                'correct': extracted == expected_answer,
            })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    return items


def load_consistency_results(json_path: Path, expected_answer: str) -> List[Dict]:
    """Load results from a consistency-check JSON array file."""
    items = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return items
        for entry in data:
            response = entry.get('consistency_check_response', '')
            extracted = extract_answer_from_response(response)
            items.append({
                'file': f"index_{entry.get('index')}",
                'expected': expected_answer,
                'extracted': extracted,
                'correct': extracted == expected_answer,
            })
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
    return items


def compute_accuracy(items: List[Dict]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it['correct']) / len(items) * 100


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_maze_bootstrap.py <parent_dir:model_name:label>")
        sys.exit(1)

    parts = sys.argv[1].split(':')
    if len(parts) == 3:
        parent_dir, model_name, label = parts
        fmt = 'json'
    elif len(parts) == 4:
        parent_dir, model_name, label, fmt = parts
    else:
        print("Expected format: parent_dir:model_name:label[:format]")
        print("  format: json (default), jsonl, thinkmorph, or consistency")
        sys.exit(1)

    # Parse remaining args for flags
    remaining = sys.argv[2:]
    base_dir = 'gemini'
    seed = 42
    for arg in remaining:
        if arg.startswith('--base-dir='):
            base_dir = arg.split('=', 1)[1]
        else:
            seed = int(arg)
    random.seed(seed)
    np.random.seed(seed)

    base_path = Path(f'/Users/log/Github/sketchvlm/results/mix_eval/maze_v2/{base_dir}')

    print(f"Loading results for: {label}")

    def resolve_paths(suffix_invalid, suffix_valid):
        if parent_dir:
            return base_path / parent_dir / suffix_invalid, base_path / parent_dir / suffix_valid
        return base_path / suffix_invalid, base_path / suffix_valid

    if fmt == 'consistency':
        inv_path, val_path = resolve_paths(f'{model_name}_invalid.json', f'{model_name}_valid.json')
        invalid_items = load_consistency_results(inv_path, 'invalid')
        valid_items = load_consistency_results(val_path, 'valid')
    elif fmt == 'jsonl':
        inv_path, val_path = resolve_paths(f'{model_name}_invalid/results.jsonl', f'{model_name}_valid/results.jsonl')
        invalid_items = load_jsonl_results(inv_path, 'invalid')
        valid_items = load_jsonl_results(val_path, 'valid')
    elif fmt == 'thinkmorph':
        inv_path, val_path = resolve_paths(f'{model_name}_invalid', f'{model_name}_valid')
        invalid_items = load_thinkmorph_results(inv_path, 'invalid')
        valid_items = load_thinkmorph_results(val_path, 'valid')
    else:
        inv_path, val_path = resolve_paths(f'{model_name}_invalid', f'{model_name}_valid')
        invalid_items = load_results(inv_path, 'invalid')
        valid_items = load_results(val_path, 'valid')
    all_items = invalid_items + valid_items
    print(f"  Loaded {len(invalid_items)} invalid + {len(valid_items)} valid = {len(all_items)} total\n")

    expected_total = K * GROUP_SIZE
    if len(all_items) < expected_total:
        print(f"Error: need {expected_total} items but only have {len(all_items)}")
        sys.exit(1)

    # Shuffle and partition into K groups of GROUP_SIZE
    random.shuffle(all_items)
    groups = [all_items[i * GROUP_SIZE:(i + 1) * GROUP_SIZE] for i in range(K)]

    overall_accs = []
    invalid_accs = []
    valid_accs = []

    print(f"{'Group':<8} {'Total':<8} {'Inv':<8} {'Val':<8} {'Overall%':<12} {'Invalid%':<12} {'Valid%':<12}")
    print("-" * 68)

    for i, group in enumerate(groups):
        inv = [it for it in group if it['expected'] == 'invalid']
        val = [it for it in group if it['expected'] == 'valid']

        oa = compute_accuracy(group)
        ia = compute_accuracy(inv)
        va = compute_accuracy(val)

        overall_accs.append(oa)
        invalid_accs.append(ia)
        valid_accs.append(va)

        print(f"{i + 1:<8} {len(group):<8} {len(inv):<8} {len(val):<8} {oa:>8.2f}%    {ia:>8.2f}%    {va:>8.2f}%")

    print("-" * 68)
    print(f"\n{'Metric':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 36)
    print(f"{'Overall':<12} {np.mean(overall_accs):>8.2f}%    {np.std(overall_accs):>8.2f}%")
    print(f"{'Invalid':<12} {np.mean(invalid_accs):>8.2f}%    {np.std(invalid_accs):>8.2f}%")
    print(f"{'Valid':<12} {np.mean(valid_accs):>8.2f}%    {np.std(valid_accs):>8.2f}%")


if __name__ == '__main__':
    main()
