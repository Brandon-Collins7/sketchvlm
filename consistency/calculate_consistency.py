"""
Calculate consistency scores between original model answers and judge responses.

Usage:
    python calculate_consistency.py --judge-dir consistency/judge_output
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from $\boxed{...}$ format.

    Args:
        text: Text containing boxed answer

    Returns:
        Extracted answer or None if not found
    """
    if not text:
        return None

    # Try to find $\boxed{X}$ pattern
    boxed_match = re.search(r'\$\\boxed\{([^}]+)\}\$', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try without dollar signs
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> Optional[str]:
    """
    Normalize answer to just the number.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (just the number) or None
    """
    if not answer:
        return None

    answer = str(answer).strip().lower()

    # Extract just the number
    number_match = re.search(r'\d+', answer)
    if number_match:
        return number_match.group(0)

    # Handle special cases
    if 'none' in answer:
        return 'none'
    if 'multiple' in answer:
        return 'multiple'

    return answer


def analyze_consistency(judge_file: Path) -> Dict:
    """
    Analyze consistency for a single judge output file.

    Args:
        judge_file: Path to judge output JSON file

    Returns:
        Dictionary with analysis results
    """
    with open(judge_file, 'r') as f:
        data = json.load(f)

    model_name = judge_file.stem

    results = {
        'model': model_name,
        'total': len(data),
        'consistent': 0,
        'inconsistent': 0,
        'judge_extraction_failed': 0,
        'model_extraction_failed': 0,
        'both_extraction_failed': 0,
        'api_failed': 0,
        'warnings': []
    }

    for entry in data:
        index = entry.get('index', 'N/A')
        original_answer = entry.get('original_extracted_answer', '')
        judge_response = entry.get('consistency_check_response', '')
        success = entry.get('success', False)

        # Check if API call failed
        if not success:
            results['api_failed'] += 1
            results['warnings'].append({
                'index': index,
                'type': 'API_FAILED',
                'error': entry.get('error', 'Unknown error')
            })
            continue

        # Extract judge's answer from boxed format
        judge_answer = extract_boxed_answer(judge_response)

        # Normalize both answers
        norm_original = normalize_answer(original_answer)
        norm_judge = normalize_answer(judge_answer)

        # Check extraction failures
        judge_failed = judge_answer is None or norm_judge is None
        model_failed = not original_answer or norm_original is None

        if judge_failed and model_failed:
            results['both_extraction_failed'] += 1
            results['warnings'].append({
                'index': index,
                'type': 'BOTH_EXTRACTION_FAILED',
                'original_answer': original_answer,
                'judge_response': judge_response[:200] + '...' if len(judge_response) > 200 else judge_response
            })
        elif judge_failed:
            results['judge_extraction_failed'] += 1
            results['warnings'].append({
                'index': index,
                'type': 'JUDGE_EXTRACTION_FAILED',
                'original_answer': original_answer,
                'judge_response': judge_response[:200] + '...' if len(judge_response) > 200 else judge_response
            })
        elif model_failed:
            results['model_extraction_failed'] += 1
            results['warnings'].append({
                'index': index,
                'type': 'MODEL_EXTRACTION_FAILED',
                'original_answer': original_answer,
                'judge_answer': judge_answer
            })
        else:
            # Both extracted successfully, compare
            if norm_original == norm_judge:
                results['consistent'] += 1
            else:
                results['inconsistent'] += 1
                results['warnings'].append({
                    'index': index,
                    'type': 'INCONSISTENT',
                    'original_answer': original_answer,
                    'normalized_original': norm_original,
                    'judge_answer': judge_answer,
                    'normalized_judge': norm_judge
                })

    # Calculate consistency score
    valid_comparisons = results['consistent'] + results['inconsistent']
    if valid_comparisons > 0:
        results['consistency_score'] = (results['consistent'] / valid_comparisons) * 100
    else:
        results['consistency_score'] = 0.0

    return results


def print_summary_table(all_results: List[Dict]):
    """
    Print a summary table of all models.

    Args:
        all_results: List of analysis results for all models
    """
    print("\n" + "="*120)
    print("CONSISTENCY SCORE SUMMARY")
    print("="*120)

    # Header
    print(f"{'Model':<25} {'Total':<8} {'Consistent':<12} {'Inconsistent':<14} {'Score':<10} {'API Failed':<12} {'Extract Fail':<15}")
    print("-"*120)

    # Sort by consistency score (descending)
    sorted_results = sorted(all_results, key=lambda x: x['consistency_score'], reverse=True)

    for result in sorted_results:
        model = result['model']
        total = result['total']
        consistent = result['consistent']
        inconsistent = result['inconsistent']
        score = result['consistency_score']
        api_failed = result['api_failed']

        # Total extraction failures
        extract_fail = (result['judge_extraction_failed'] +
                       result['model_extraction_failed'] +
                       result['both_extraction_failed'])

        print(f"{model:<25} {total:<8} {consistent:<12} {inconsistent:<14} {score:>6.1f}%    {api_failed:<12} {extract_fail:<15}")

    print("-"*120)


def print_warnings(all_results: List[Dict], max_warnings: int = 10):
    """
    Print warnings for failed extractions and inconsistencies.

    Args:
        all_results: List of analysis results for all models
        max_warnings: Maximum number of warnings to show per type per model
    """
    print("\n" + "="*120)
    print("WARNINGS AND ISSUES")
    print("="*120)

    for result in all_results:
        model = result['model']
        warnings = result['warnings']

        if not warnings:
            continue

        print(f"\n{model}:")
        print("-"*120)

        # Group warnings by type
        by_type = defaultdict(list)
        for warning in warnings:
            by_type[warning['type']].append(warning)

        for warning_type, items in by_type.items():
            print(f"\n  {warning_type}: {len(items)} occurrences")

            # Show first few examples
            for i, item in enumerate(items[:max_warnings]):
                print(f"    [{item['index']}]", end=" ")

                if warning_type == 'API_FAILED':
                    print(f"Error: {item['error']}")

                elif warning_type == 'JUDGE_EXTRACTION_FAILED':
                    print(f"Original: {item['original_answer']}")
                    print(f"          Judge response: {item['judge_response']}")

                elif warning_type == 'MODEL_EXTRACTION_FAILED':
                    print(f"Original: '{item['original_answer']}' | Judge: '{item['judge_answer']}'")

                elif warning_type == 'INCONSISTENT':
                    print(f"Original: {item['original_answer']} → {item['normalized_original']} | "
                          f"Judge: {item['judge_answer']} → {item['normalized_judge']}")

                elif warning_type == 'BOTH_EXTRACTION_FAILED':
                    print(f"Original: {item['original_answer']}")
                    print(f"          Judge: {item['judge_response']}")

            if len(items) > max_warnings:
                print(f"    ... and {len(items) - max_warnings} more")


def main():
    parser = argparse.ArgumentParser(description='Calculate consistency scores from judge outputs')
    parser.add_argument('--judge-dir', type=str, required=True,
                       help='Directory containing consistency_results_*.json files')
    parser.add_argument('--max-warnings', type=int, default=5,
                       help='Maximum warnings to show per type (default: 5)')
    parser.add_argument('--show-warnings', action='store_true',
                       help='Show detailed warnings')

    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)

    if not judge_dir.exists():
        print(f"Error: Directory not found: {judge_dir}")
        return

    # Find all judge output files (try both patterns)
    judge_files = sorted(judge_dir.glob('consistency_results_*.json'))
    if not judge_files:
        # Try alternative pattern - any .json file
        judge_files = sorted(judge_dir.glob('*.json'))

    if not judge_files:
        print(f"No JSON files found in {judge_dir}")
        return

    print(f"Found {len(judge_files)} judge output files")

    # Analyze each file
    all_results = []
    for judge_file in judge_files:
        print(f"Analyzing {judge_file.name}...")
        results = analyze_consistency(judge_file)
        all_results.append(results)

    # Print summary table
    print_summary_table(all_results)

    # Print warnings if requested
    if args.show_warnings:
        print_warnings(all_results, max_warnings=args.max_warnings)
    else:
        print(f"\nUse --show-warnings to see detailed warnings")


if __name__ == '__main__':
    main()
