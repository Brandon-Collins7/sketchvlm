"""
Calculate counting accuracy by comparing judge responses against ground truth.

Compares the judge's extracted answer (from $\\boxed{integer}$) against gt_number
from the source data.

Usage:
    python calculate_stats/calculate_counting_accuracy.py \
        --judge consistency/judge_output/countbench_nano_banana_results.json \
        --source consistency/source_data/countbench_nano_banana.json

    # Show per-item details for wrong answers:
    python calculate_stats/calculate_counting_accuracy.py \
        --judge consistency/judge_output/countbench_nano_banana_results.json \
        --source consistency/source_data/countbench_nano_banana.json \
        --show-errors
"""

import json
import re
import base64
import argparse
from pathlib import Path
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract integer from $\\boxed{...}$ format."""
    if not text:
        return None

    # Try $\boxed{X}$
    match = re.search(r'\$\\boxed\{([^}]+)\}\$', text)
    if not match:
        # Try without dollar signs
        match = re.search(r'\\boxed\{([^}]+)\}', text)

    if match:
        try:
            return int(match.group(1).strip())
        except ValueError:
            return None

    # Fallback: last number in the text
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return None

    return None


def embed_image(image_path: str, max_width: int = 300) -> str:
    """Return an <img> tag with base64-embedded image, or a placeholder if missing."""
    p = Path(image_path)
    if not p.exists():
        return f'<span style="color:#999">Missing: {p.name}</span>'
    ext = p.suffix.lower()
    mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')
    b64 = base64.b64encode(p.read_bytes()).decode('utf-8')
    return f'<img src="data:{mime};base64,{b64}" style="max-width:{max_width}px;max-height:{max_width}px;">'


def generate_html(rows, accuracy, correct, incorrect, failed, total, html_path, judge_name, source_name):
    """Generate an HTML report with a table of all results."""
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Counting Accuracy Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ margin-bottom: 5px; }}
.summary {{ background: #fff; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; display: inline-block; }}
.summary td {{ padding: 2px 12px 2px 0; }}
.accuracy {{ font-size: 1.4em; font-weight: bold; }}
table.results {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; }}
table.results th {{ background: #333; color: #fff; padding: 10px 12px; text-align: left; position: sticky; top: 0; }}
table.results td {{ padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
table.results tr:hover {{ background: #f9f9f9; }}
tr.correct {{ }}
tr.wrong td {{ background: #fff0f0; }}
tr.failed td {{ background: #fff8e0; }}
.response {{ max-width: 400px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; font-size: 0.85em; line-height: 1.4; }}
.match {{ color: #2a7a2a; font-weight: bold; }}
.mismatch {{ color: #cc0000; font-weight: bold; }}
.na {{ color: #999; }}
img {{ border-radius: 4px; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Counting Accuracy Report</h1>
<div class="summary">
<table>
<tr><td>Judge file:</td><td><b>{judge_name}</b></td></tr>
<tr><td>Source file:</td><td><b>{source_name}</b></td></tr>
<tr><td>Total:</td><td>{total}</td></tr>
<tr><td>Correct:</td><td>{correct}</td></tr>
<tr><td>Incorrect:</td><td>{incorrect}</td></tr>
<tr><td>Failed:</td><td>{failed}</td></tr>
<tr><td>Accuracy:</td><td class="accuracy">{accuracy:.1f}%</td></tr>
</table>
</div>
<table class="results">
<thead>
<tr>
<th>#</th>
<th>Question</th>
<th>Original Image</th>
<th>Annotated Image</th>
<th>GT</th>
<th>Predicted</th>
<th>Result</th>
<th>Gemini Response</th>
</tr>
</thead>
<tbody>
"""
    for row in rows:
        if row['status'] == 'correct':
            cls = 'correct'
            result_html = '<span class="match">Correct</span>'
        elif row['status'] == 'wrong':
            cls = 'wrong'
            result_html = '<span class="mismatch">Wrong</span>'
        else:
            cls = 'failed'
            result_html = f'<span class="na">{row["status"]}</span>'

        predicted_str = str(row['predicted']) if row['predicted'] is not None else '<span class="na">N/A</span>'
        response_escaped = (row['response'] or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        html += f"""<tr class="{cls}">
<td>{row['index']}</td>
<td>{row['question']}</td>
<td>{row['original_img']}</td>
<td>{row['annotated_img']}</td>
<td>{row['gt']}</td>
<td>{predicted_str}</td>
<td>{result_html}</td>
<td><div class="response">{response_escaped}</div></td>
</tr>
"""

    html += """</tbody>
</table>
</body>
</html>"""

    with open(html_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='Calculate counting accuracy')
    parser.add_argument('--judge', type=str, required=True,
                       help='Judge output JSON file')
    parser.add_argument('--source', type=str, required=True,
                       help='Source data JSON file with gt_number')
    parser.add_argument('--show-errors', action='store_true',
                       help='Show details for incorrect answers')

    args = parser.parse_args()

    with open(args.judge, 'r') as f:
        judge_data = json.load(f)

    with open(args.source, 'r') as f:
        source_data = json.load(f)

    # Build lookup from image_path to source entry
    source_lookup = {}
    for entry in source_data:
        source_lookup[entry['image_path']] = entry

    correct = 0
    incorrect = 0
    failed = 0
    no_gt = 0
    errors = []
    html_rows = []

    for entry in judge_data:
        image_path = entry.get('image_path', '')
        src = source_lookup.get(image_path, {})
        gt = src.get('gt_number')
        question = src.get('question', entry.get('prompt', '').split('\n')[-1])
        original_image_path = src.get('original_image_path', '')

        if gt is None:
            no_gt += 1
            continue

        response = entry.get('consistency_check_response', '') or ''
        idx = entry.get('index', '?')

        if not entry.get('success', False):
            failed += 1
            error_msg = entry.get('error', 'Unknown error')
            errors.append({'index': idx, 'gt': gt, 'predicted': None, 'reason': 'api_failed', 'response_snippet': error_msg})
            html_rows.append({'index': idx, 'question': question, 'original_img': embed_image(original_image_path), 'annotated_img': embed_image(image_path), 'gt': gt, 'predicted': None, 'status': 'api_failed', 'response': error_msg})
            continue

        predicted = extract_boxed_answer(response)

        if predicted is None:
            failed += 1
            errors.append({'index': idx, 'gt': gt, 'predicted': None, 'reason': 'extraction_failed', 'response_snippet': response[:150]})
            html_rows.append({'index': idx, 'question': question, 'original_img': embed_image(original_image_path), 'annotated_img': embed_image(image_path), 'gt': gt, 'predicted': None, 'status': 'extraction_failed', 'response': response})
            continue

        if predicted == gt:
            correct += 1
            status = 'correct'
        else:
            incorrect += 1
            status = 'wrong'
            errors.append({'index': idx, 'gt': gt, 'predicted': predicted, 'reason': 'wrong', 'question': question})

        html_rows.append({'index': idx, 'question': question, 'original_img': embed_image(original_image_path), 'annotated_img': embed_image(image_path), 'gt': gt, 'predicted': predicted, 'status': status, 'response': response})

    total_evaluated = correct + incorrect
    accuracy = (correct / total_evaluated * 100) if total_evaluated > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"COUNTING ACCURACY")
    print(f"{'='*60}")
    print(f"Judge file:  {Path(args.judge).name}")
    print(f"Source file: {Path(args.source).name}")
    print(f"{'='*60}")
    print(f"Total entries:     {len(judge_data)}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {incorrect}")
    print(f"Extraction failed: {failed}")
    if no_gt > 0:
        print(f"No ground truth:   {no_gt}")
    print(f"{'='*60}")
    print(f"Accuracy:          {accuracy:.1f}% ({correct}/{total_evaluated})")
    print(f"{'='*60}")

    # Generate HTML report
    html_path = Path(args.judge).with_suffix('.html')
    print(f"\nGenerating HTML report...")
    generate_html(html_rows, accuracy, correct, incorrect, failed, len(judge_data), html_path, Path(args.judge).name, Path(args.source).name)
    print(f"HTML report saved to: {html_path}")

    if args.show_errors and errors:
        wrong = [e for e in errors if e['reason'] == 'wrong']
        extract_fail = [e for e in errors if e['reason'] == 'extraction_failed']
        api_fail = [e for e in errors if e['reason'] == 'api_failed']

        if wrong:
            print(f"\nWRONG ANSWERS ({len(wrong)}):")
            print("-"*60)
            for e in wrong:
                print(f"  [{e['index']}] GT={e['gt']} Predicted={e['predicted']}  {e.get('question', '')}")

        if extract_fail:
            print(f"\nEXTRACTION FAILURES ({len(extract_fail)}):")
            print("-"*60)
            for e in extract_fail:
                print(f"  [{e['index']}] GT={e['gt']}  Response: {e['response_snippet']}")

        if api_fail:
            print(f"\nAPI FAILURES ({len(api_fail)}):")
            print("-"*60)
            for e in api_fail:
                print(f"  [{e['index']}] GT={e['gt']}  Error: {e['response_snippet']}")


if __name__ == '__main__':
    main()
