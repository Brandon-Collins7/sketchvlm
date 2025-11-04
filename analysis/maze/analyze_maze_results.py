#!/usr/bin/env python3
"""
Analyze Gemini maze validation responses with path length breakdown.
This script evaluates model accuracy on maze path validation tasks, broken down by path length.

Example usage:
    python3 analysis/maze/analyze_maze_results.py \
        "gemini25_flash:gemini25_flash:Flash (Sketch)" \
        "gemini25_pro:gemini25_pro:Pro (Sketch)" \
        "direct_vqa:gemini25_pro:Pro (Direct VQA)"

    For index-based evaluation:
    python3 analysis/maze/analyze_maze_results.py --index-mode \
        "gemini25_pro:gemini25_pro:Pro (Sketch)"
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


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


def extract_index_answer_from_response(response_text: str) -> Union[int, str]:
    """
    Extract answer from model response for index-based evaluation.
    Extracts either a numeric index from $\\boxed{N}$ notation or 'valid' string.

    Args:
        response_text: The full model output

    Returns:
        Extracted answer: integer (0-based index) or 'valid' string, or 'unknown' if can't parse
    """
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
    # Try to extract number from $\boxed{N}$ notation (search from end to get the final answer)
    boxed_matches = list(re.finditer(r'\$\\boxed\{(\d+)\}\$', response_text))
    if boxed_matches:
        # Get the last match (usually the final answer)
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


def analyze_directory(dir_path: Path, expected_answer: Union[str, int], maze_to_path_length: Dict[str, int],
                      index_mode: bool = False) -> Dict:
    """
    Analyze all JSON files in a directory.

    Args:
        dir_path: Path to directory containing JSON files
        expected_answer: The expected answer ('valid', 'invalid', or for index mode: 'valid' or int)
        maze_to_path_length: Mapping of maze_id to path_length
        index_mode: If True, use index-based evaluation (extract numeric indices or 'valid')

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

            # In index mode, get ground truth from the dataset metadata.json
            if index_mode:
                if '_valid' in str(dir_path):
                    # For valid paths, ground truth is 'valid'
                    gt_answer = 'valid'
                else:
                    # For invalid paths, read from metadata.json
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

                # Extract model's answer
                extracted_answer = extract_index_answer_from_response(model_output)
            else:
                gt_answer = expected_answer
                extracted_answer = extract_answer_from_response(model_output)

            # Determine correctness
            is_correct = (extracted_answer == gt_answer)
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
                'expected': gt_answer,
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
                'expected': expected_answer if not index_mode else 'N/A',
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


def create_overall_accuracy_plot(results_list: List[Dict], output_path: Path):
    """
    Create a bar chart showing overall accuracy for different configurations.

    Args:
        results_list: List of tuples (label, invalid_results, valid_results)
        output_path: Path to save the plot
    """
    labels = []
    overall_accuracies = []
    invalid_accuracies = []
    valid_accuracies = []

    for label, invalid_results, valid_results in results_list:
        labels.append(label)

        # Calculate overall accuracy
        total_all = invalid_results['total'] + valid_results['total']
        correct_all = invalid_results['correct'] + valid_results['correct']
        overall_acc = (correct_all / total_all * 100) if total_all > 0 else 0.0
        overall_accuracies.append(overall_acc)

        # Calculate invalid accuracy
        invalid_acc = (invalid_results['correct'] / invalid_results['total'] * 100) if invalid_results['total'] > 0 else 0.0
        invalid_accuracies.append(invalid_acc)

        # Calculate valid accuracy
        valid_acc = (valid_results['correct'] / valid_results['total'] * 100) if valid_results['total'] > 0 else 0.0
        valid_accuracies.append(valid_acc)

    # Determine colors and hatches based on label
    def get_style(label):
        if 'Flash' in label:
            color = '#4472C4'  # Blue for Flash
        elif 'Pro' in label:
            color = '#C55A11'  # Red for Pro
        else:
            color = 'gray'

        # Add hatching for sketch versions
        hatch = '///' if 'Sketch' in label else None

        return color, hatch

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Overall accuracy
    for idx, (label, acc) in enumerate(zip(labels, overall_accuracies)):
        color, hatch = get_style(label)
        axes[0].bar(idx, acc, color=color, alpha=0.8, hatch=hatch, edgecolor='black', linewidth=1.5)
        axes[0].text(idx, acc, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)

    # Invalid accuracy
    for idx, (label, acc) in enumerate(zip(labels, invalid_accuracies)):
        color, hatch = get_style(label)
        axes[1].bar(idx, acc, color=color, alpha=0.8, hatch=hatch, edgecolor='black', linewidth=1.5)
        axes[1].text(idx, acc, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Invalid Paths Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)

    # Valid accuracy
    for idx, (label, acc) in enumerate(zip(labels, valid_accuracies)):
        color, hatch = get_style(label)
        axes[2].bar(idx, acc, color=color, alpha=0.8, hatch=hatch, edgecolor='black', linewidth=1.5)
        axes[2].text(idx, acc, f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Valid Paths Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 105)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved overall accuracy plot to: {output_path}")


def create_path_length_plot(results_list: List[Dict], output_path: Path):
    """
    Create a line graph showing accuracy by path length for different configurations.

    Args:
        results_list: List of tuples (label, invalid_results, valid_results)
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Determine line style based on label
    def get_line_style(label):
        if 'Flash' in label:
            color = '#4472C4'  # Blue for Flash
        elif 'Pro' in label:
            color = '#C55A11'  # Red for Pro
        else:
            color = 'gray'

        # Dashed line for sketch, solid for direct VQA
        linestyle = '--' if 'Sketch' in label else '-'
        linewidth = 3.0 if 'Sketch' in label else 2.5
        marker = 'o'

        return color, linestyle, linewidth, marker

    # Valid paths by path length
    for idx, (label, invalid_results, valid_results) in enumerate(results_list):
        by_path_length = valid_results['by_path_length']
        path_lengths = sorted(by_path_length.keys())
        accuracies = []

        for pl in path_lengths:
            stats = by_path_length[pl]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            accuracies.append(accuracy)

        color, linestyle, linewidth, marker = get_line_style(label)
        axes[0].plot(path_lengths, accuracies,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    markersize=8,
                    label=label,
                    alpha=0.9)

    axes[0].set_xlabel('Path Length', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Valid Paths Accuracy by Path Length', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].set_xticks(range(1, 8))

    # Combined (Invalid + Valid) by path length
    for idx, (label, invalid_results, valid_results) in enumerate(results_list):
        combined_by_length = defaultdict(lambda: {'total': 0, 'correct': 0})

        for pl in set(list(invalid_results['by_path_length'].keys()) + list(valid_results['by_path_length'].keys())):
            invalid_stats = invalid_results['by_path_length'][pl]
            valid_stats = valid_results['by_path_length'][pl]

            combined_by_length[pl]['total'] = invalid_stats['total'] + valid_stats['total']
            combined_by_length[pl]['correct'] = invalid_stats['correct'] + valid_stats['correct']

        path_lengths = sorted(combined_by_length.keys())
        accuracies = []

        for pl in path_lengths:
            stats = combined_by_length[pl]
            total = stats['total']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0.0
            accuracies.append(accuracy)

        color, linestyle, linewidth, marker = get_line_style(label)
        axes[1].plot(path_lengths, accuracies,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    markersize=8,
                    label=label,
                    alpha=0.9)

    axes[1].set_xlabel('Path Length', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Combined (Invalid + Valid) Accuracy by Path Length', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].set_xticks(range(1, 8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved path length plot to: {output_path}")


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_maze_results.py [--index-mode] <config1> [<config2> ...]")
        print("       Each config should be: parent_dir:model_name:label")
        print()
        print("Flags:")
        print("  --index-mode    Enable index-based evaluation (ground truth is numeric index for invalid paths)")
        print()
        print("Examples:")
        print("  Single model:")
        print("    python analyze_maze_results.py gemini25_flash:gemini25_flash:\"Flash (Sketch)\"")
        print()
        print("  Multiple models for comparison:")
        print("    python analyze_maze_results.py \\")
        print("      gemini25_flash:gemini25_flash:\"Flash (Sketch)\" \\")
        print("      gemini25_pro:gemini25_pro:\"Pro (Sketch)\" \\")
        print("      direct_vqa:gemini25_pro:\"Pro (Direct VQA)\"")
        print()
        print("  Index-based evaluation:")
        print("    python analyze_maze_results.py --index-mode \\")
        print("      gemini25_pro:gemini25_pro:\"Pro (Sketch)\"")
        sys.exit(1)

    # Check for --index-mode flag
    index_mode = False
    args = sys.argv[1:]
    if '--index-mode' in args:
        index_mode = True
        args = [arg for arg in args if arg != '--index-mode']

    # Check if we have any configs after removing flags
    if not args:
        print("Error: No model configurations provided")
        print("Usage: python analyze_maze_results.py [--index-mode] <config1> [<config2> ...]")
        sys.exit(1)

    # Parse configurations
    configs = []
    for arg in args:
        parts = arg.split(':')
        if len(parts) != 3:
            print(f"Error: Invalid config format: {arg}")
            print("Expected format: parent_dir:model_name:label")
            sys.exit(1)
        configs.append(tuple(parts))

    # Build maze to path length mapping
    print("Building maze to path length mapping...")
    maze_to_path_length = build_maze_to_path_length_mapping()
    print(f"Mapped {len(maze_to_path_length)} mazes to path lengths 1-7\n")

    # Determine base path and output directory based on mode
    if index_mode:
        print("Running in INDEX MODE: Ground truth for invalid paths is the modified index\n")
        base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze/gemini/index')
        output_dir = Path('/Users/log/Github/sketchvlm/analysis/maze/index')
    else:
        print("Running in BINARY MODE: Ground truth is 'valid' or 'invalid'\n")
        base_path = Path('/Users/log/Github/sketchvlm/results/mix_eval/maze/gemini')
        output_dir = Path('/Users/log/Github/sketchvlm/analysis/maze/binary')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all configurations
    all_results = []

    for parent_dir, model_name, label in configs:
        # Define paths
        invalid_dir = base_path / parent_dir / f'{model_name}_invalid'
        valid_dir = base_path / parent_dir / f'{model_name}_valid'

        # For index mode, check if parent_dir is empty (direct subdirectories of index/)
        if index_mode and parent_dir == '':
            invalid_dir = base_path / f'{model_name}_invalid'
            valid_dir = base_path / f'{model_name}_valid'

        print("=" * 80)
        if index_mode:
            print(f"{label} - Maze Path Index Identification Analysis")
        else:
            print(f"{label} - Maze Path Validation Analysis")
        print("=" * 80)
        print()

        # Analyze invalid directory
        if index_mode:
            print("Analyzing INVALID subdirectory (ground truth: numeric index)...")
        else:
            print("Analyzing INVALID subdirectory (should answer 'invalid')...")
        invalid_results = analyze_directory(invalid_dir, 'invalid' if not index_mode else None,
                                           maze_to_path_length, index_mode=index_mode)

        # Analyze valid directory (should answer "valid")
        print("Analyzing VALID subdirectory (should answer 'valid')...")
        valid_results = analyze_directory(valid_dir, 'valid', maze_to_path_length, index_mode=index_mode)

        # Store results for plotting
        all_results.append((label, invalid_results, valid_results))

        # Print results
        print()
        print("=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print()

        # Invalid directory results
        if index_mode:
            print(f"INVALID Directory (expected answer: numeric index of incorrect move)")
        else:
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
        print()

    # Generate plots if we have results
    if all_results:
        # Generate overall accuracy bar chart
        overall_plot_path = output_dir / 'maze_overall_accuracy.png'
        print(f"\nSaved overall accuracy plot to: {overall_plot_path}")
        create_overall_accuracy_plot(all_results, overall_plot_path)

        # Generate path length line graphs
        path_length_plot_path = output_dir / 'maze_path_length_accuracy.png'
        print(f"Saved path length plot to: {path_length_plot_path}")
        create_path_length_plot(all_results, path_length_plot_path)


if __name__ == '__main__':
    main()
