#!/usr/bin/env python3
"""
Find all item numbers where the model response is unknown (no answer extracted).
"""

import json
import re
import sys
from pathlib import Path


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


def find_unknown_items(results_dir: Path) -> list:
    """Find all item numbers with unknown responses."""
    unknown_items = []

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist", file=sys.stderr)
        return unknown_items

    json_files = list(results_dir.glob('item_*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            model_output = data.get('model_output_full', '')
            answer = extract_answer_from_response(model_output)

            if answer == 'unknown':
                # Extract item number from filename
                item_num = json_file.stem.replace('item_', '')
                unknown_items.append(int(item_num))

        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)

    return sorted(unknown_items)


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_none_responses.py <results_directory>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python find_none_responses.py results/mix_eval/maze_v2/gpt5/gpt5_low_1000_valid", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    unknown_items = find_unknown_items(results_dir)

    if unknown_items:
        item_list = ','.join(str(n) for n in unknown_items)
        print(f'"{item_list}"')
    else:
        print('""')
        print(f"No unknown responses found in {results_dir}", file=sys.stderr)


if __name__ == '__main__':
    main()
