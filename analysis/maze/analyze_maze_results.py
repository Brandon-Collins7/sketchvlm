#!/usr/bin/env python3
"""
Analyze Gemini 2.5 Flash maze validation responses with path length breakdown.
This script evaluates model accuracy on maze path validation tasks, broken down by path length.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def build_maze_to_path_length_mapping() -> Dict[str, int]:
    """
    Build a mapping of maze_id to path_length by scanning the dataset directories.

    Returns:
        Dictionary mapping maze_id to path_length
    """
    maze_to_path_length = {}

    for path_length in range(1, 8):
        path_dir = Path(f'/Users/log/Github/sketchvlm/datasets/maze_v1/path_length_{path_length}')
        if not path_dir.exists():
            continue
        for maze_dir in path_dir.iterdir():
            if maze_dir.is_dir() and maze_dir.name.startswith('maze_'):
                maze_to_path_length[maze_dir.name] = path_length

    return maze_to_path_length


def extract_maze_id(source_image_path: str) -> str:
    """
    Extract maze ID from source image path.

    Args:
        source_image_path: Path like 'datasets/maze_v1/sketch_valid_flattened/maze_10_edf44602.png'

    Returns:
        Maze ID like 'maze_10_edf44602'
    """
    if not source_image_path:
        return None

    # Extract filename without extension
    filename = Path(source_image_path).stem
    return filename


def extract_answer_from_response(response_text: str) -> str:
    """
    Extract answer from model response.
    First tries to extract from <final_answer> tags, then falls back to last 30 chars.

    Args:
        response_text: The full model output

    Returns:
        Extracted answer: 'valid', 'invalid', or 'unknown'
    """
    if not response_text or response_text.strip() == '':
        return 'unknown'

    # Try to extract from <final_answer> tags first
    final_answer_match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', response_text, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        answer_text = final_answer_match.group(1).strip()
        # Look for valid/invalid in the extracted text
        if 'valid' in answer_text.lower():
            if 'invalid' in answer_text.lower():
                return 'invalid'  # "invalid" contains "valid", check invalid first
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


def analyze_directory(dir_path: Path, expected_answer: str, maze_to_path_length: Dict[str, int]) -> Dict:
    """
    Analyze all JSON files in a directory.

    Args:
        dir_path: Path to directory containing JSON files
        expected_answer: The expected answer ('valid' or 'invalid')
        maze_to_path_length: Mapping of maze_id to path_length

    Returns:
        Dictionary with analysis results
    """
    json_files = sorted(dir_path.glob('item_*.json'))

    results = {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'unknown': 0,
        'expected_answer': expected_answer,
        'details': [],
        'by_path_length': defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0, 'unknown': 0})
    }

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Get the model's full output
            model_output = data.get('model_output_full', '')

            # Get the maze ID and path length
            source_image = data.get('source_image', '')
            maze_id = extract_maze_id(source_image)
            path_length = maze_to_path_length.get(maze_id, None)

            # Extract the answer
            extracted_answer = extract_answer_from_response(model_output)

            # Determine correctness
            is_correct = (extracted_answer == expected_answer)
            is_unknown = (extracted_answer == 'unknown')

            # Update overall stats
            results['total'] += 1
            if is_unknown:
                results['unknown'] += 1
            elif is_correct:
                results['correct'] += 1
            else:
                results['incorrect'] += 1

            # Update path length stats
            if path_length is not None:
                results['by_path_length'][path_length]['total'] += 1
                if is_unknown:
                    results['by_path_length'][path_length]['unknown'] += 1
                elif is_correct:
                    results['by_path_length'][path_length]['correct'] += 1
                else:
                    results['by_path_length'][path_length]['incorrect'] += 1

            results['details'].append({
                'file': json_file.name,
                'expected': expected_answer,
                'extracted': extracted_answer,
                'correct': is_correct,
                'unknown': is_unknown,
                'maze_id': maze_id,
                'path_length': path_length
            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            results['total'] += 1
            results['unknown'] += 1
            results['details'].append({
                'file': json_file.name,
                'expected': expected_answer,
                'extracted': 'error',
                'correct': False,
                'unknown': True,
                'error': str(e),
                'maze_id': None,
                'path_length': None
            })

    return results


def print_path_length_breakdown(results: Dict, title: str):
    """Print the breakdown by path length."""
    print(f"\n{title}")
    print("-" * 80)

    by_path_length = results['by_path_length']
    path_lengths = sorted(by_path_length.keys())

    if not path_lengths:
        print("  No path length data available")
        return

    print(f"{'Path Length':<15} {'Total':<10} {'Correct':<10} {'Incorrect':<12} {'Unknown':<10} {'Accuracy':<10}")
    print("-" * 80)

    for pl in path_lengths:
        stats = by_path_length[pl]
        total = stats['total']
        correct = stats['correct']
        incorrect = stats['incorrect']
        unknown = stats['unknown']
        accuracy = (correct / total * 100) if total > 0 else 0.0

        print(f"{pl:<15} {total:<10} {correct:<10} {incorrect:<12} {unknown:<10} {accuracy:>6.2f}%")


def main():
    # Build maze to path length mapping
    print("Building maze to path length mapping...")
    maze_to_path_length = build_maze_to_path_length_mapping()
    print(f"Mapped {len(maze_to_path_length)} mazes to path lengths 1-7\n")

    # Define paths
    base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze/gemini')
    invalid_dir = base_path / 'gemini25_flash_invalid'
    valid_dir = base_path / 'gemini25_flash_valid'

    print("=" * 80)
    print("Gemini 2.5 Flash - Maze Path Validation Analysis")
    print("=" * 80)
    print()

    # Analyze invalid directory (should answer "invalid")
    print("Analyzing INVALID subdirectory...")
    invalid_results = analyze_directory(invalid_dir, 'invalid', maze_to_path_length)

    # Analyze valid directory (should answer "valid")
    print("Analyzing VALID subdirectory...")
    valid_results = analyze_directory(valid_dir, 'valid', maze_to_path_length)

    # Print results
    print()
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print()

    # Invalid directory results
    print(f"INVALID Directory (expected answer: 'invalid')")
    print(f"  Total files:    {invalid_results['total']}")
    print(f"  Correct:        {invalid_results['correct']}")
    print(f"  Incorrect:      {invalid_results['incorrect']}")
    print(f"  Unknown:        {invalid_results['unknown']}")
    if invalid_results['total'] > 0:
        accuracy = (invalid_results['correct'] / invalid_results['total']) * 100
        print(f"  Accuracy:       {accuracy:.2f}%")
    print()

    # Valid directory results
    print(f"VALID Directory (expected answer: 'valid')")
    print(f"  Total files:    {valid_results['total']}")
    print(f"  Correct:        {valid_results['correct']}")
    print(f"  Incorrect:      {valid_results['incorrect']}")
    print(f"  Unknown:        {valid_results['unknown']}")
    if valid_results['total'] > 0:
        accuracy = (valid_results['correct'] / valid_results['total']) * 100
        print(f"  Accuracy:       {accuracy:.2f}%")
    print()

    # Overall accuracy
    total_all = invalid_results['total'] + valid_results['total']
    correct_all = invalid_results['correct'] + valid_results['correct']
    incorrect_all = invalid_results['incorrect'] + valid_results['incorrect']
    unknown_all = invalid_results['unknown'] + valid_results['unknown']

    print(f"OVERALL")
    print(f"  Total files:    {total_all}")
    print(f"  Correct:        {correct_all}")
    print(f"  Incorrect:      {incorrect_all}")
    print(f"  Unknown:        {unknown_all}")
    if total_all > 0:
        overall_accuracy = (correct_all / total_all) * 100
        print(f"  Accuracy:       {overall_accuracy:.2f}%")
    print()

    # Path length breakdown
    print("=" * 80)
    print("BREAKDOWN BY PATH LENGTH")
    print("=" * 80)

    print_path_length_breakdown(invalid_results, "INVALID Paths by Length")
    print_path_length_breakdown(valid_results, "VALID Paths by Length")

    # Combined path length breakdown
    print("\nCOMBINED (Invalid + Valid) by Path Length")
    print("-" * 80)

    combined_by_length = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0, 'unknown': 0})

    for pl in set(list(invalid_results['by_path_length'].keys()) + list(valid_results['by_path_length'].keys())):
        invalid_stats = invalid_results['by_path_length'][pl]
        valid_stats = valid_results['by_path_length'][pl]

        combined_by_length[pl]['total'] = invalid_stats['total'] + valid_stats['total']
        combined_by_length[pl]['correct'] = invalid_stats['correct'] + valid_stats['correct']
        combined_by_length[pl]['incorrect'] = invalid_stats['incorrect'] + valid_stats['incorrect']
        combined_by_length[pl]['unknown'] = invalid_stats['unknown'] + valid_stats['unknown']

    path_lengths = sorted(combined_by_length.keys())
    print(f"{'Path Length':<15} {'Total':<10} {'Correct':<10} {'Incorrect':<12} {'Unknown':<10} {'Accuracy':<10}")
    print("-" * 80)

    for pl in path_lengths:
        stats = combined_by_length[pl]
        total = stats['total']
        correct = stats['correct']
        incorrect = stats['incorrect']
        unknown = stats['unknown']
        accuracy = (correct / total * 100) if total > 0 else 0.0

        print(f"{pl:<15} {total:<10} {correct:<10} {incorrect:<12} {unknown:<10} {accuracy:>6.2f}%")

    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
