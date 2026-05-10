"""
Calculate maze quality scores from judge outputs.

Usage:
    python calculate_maze_quality.py --judge-dir consistency/judge_output/grid_world_quality
"""

import os
import json
import re
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def extract_quality_score(text: str) -> Optional[int]:
    """
    Extract quality score from "Quality Score: X" format.

    Args:
        text: Text containing the score

    Returns:
        Extracted score (1-5) or None if not found
    """
    if not text:
        return None

    # Try to find "Quality Score: X" pattern
    score_match = re.search(r'Quality\s+Score:\s*(\d+)', text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 5:
            return score

    return None


def analyze_quality_scores(judge_file: Path) -> Dict:
    """
    Analyze quality scores for a single judge output file.

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
        'scores': [],
        'extraction_failed': 0,
        'api_failed': 0,
        'average_score': 0.0,
        'warnings': []
    }

    for entry in data:
        index = entry.get('index', 'N/A')
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

        # Extract quality score
        score = extract_quality_score(judge_response)

        if score is None:
            results['extraction_failed'] += 1
            results['warnings'].append({
                'index': index,
                'type': 'EXTRACTION_FAILED',
                'judge_response': judge_response[:200] + '...' if len(judge_response) > 200 else judge_response
            })
        else:
            results['scores'].append(score)

    # Calculate average score
    if results['scores']:
        results['average_score'] = sum(results['scores']) / len(results['scores'])
    else:
        results['average_score'] = 0.0

    return results


def combine_invalid_valid_results(all_results: List[Dict]) -> Dict[str, Dict]:
    """
    Combine results for models that have both invalid and valid variants.

    Args:
        all_results: List of analysis results for all models

    Returns:
        Dictionary mapping base model name to combined results
    """
    # Group by base model name (without invalid/valid suffix)
    by_base_model = defaultdict(list)

    for result in all_results:
        model_name = result['model']

        # Extract base model name (remove _invalid/_valid suffix)
        if '_invalid' in model_name:
            base_name = model_name.replace('_invalid', '')
        elif '_valid' in model_name:
            base_name = model_name.replace('_valid', '')
        else:
            base_name = model_name

        by_base_model[base_name].append(result)

    # Combine results
    combined = {}
    for base_name, results_list in by_base_model.items():
        if len(results_list) == 1:
            # Only one variant, use as is
            combined[base_name] = {
                'results': results_list,
                'combined_scores': results_list[0]['scores'],
                'combined_average': results_list[0]['average_score']
            }
        else:
            # Multiple variants, combine scores
            all_scores = []
            for r in results_list:
                all_scores.extend(r['scores'])

            combined_average = sum(all_scores) / len(all_scores) if all_scores else 0.0

            combined[base_name] = {
                'results': results_list,
                'combined_scores': all_scores,
                'combined_average': combined_average
            }

    return combined


def print_summary_table(all_results: List[Dict], combined_results: Dict[str, Dict]):
    """
    Print a summary table of all models.

    Args:
        all_results: List of analysis results for all models
        combined_results: Dictionary of combined variant results
    """
    print("\n" + "="*120)
    print("MAZE QUALITY SCORE SUMMARY")
    print("="*120)

    # Header
    print(f"{'Model':<30} {'Total':<8} {'Valid':<8} {'Avg Score':<12} {'API Failed':<12} {'Extract Fail':<15}")
    print("-"*120)

    # Sort combined results by average score (descending)
    sorted_models = sorted(combined_results.items(),
                          key=lambda x: x[1]['combined_average'],
                          reverse=True)

    for base_name, combined in sorted_models:
        results_list = combined['results']

        if len(results_list) == 1:
            # Single variant
            result = results_list[0]
            model = result['model']
            total = result['total']
            valid = len(result['scores'])
            avg_score = result['average_score']
            api_failed = result['api_failed']
            extract_fail = result['extraction_failed']

            print(f"{model:<30} {total:<8} {valid:<8} {avg_score:>6.2f}       {api_failed:<12} {extract_fail:<15}")
        else:
            # Multiple variants - show combined first, then individual
            total_entries = sum(r['total'] for r in results_list)
            total_valid = len(combined['combined_scores'])
            combined_avg = combined['combined_average']
            total_api_failed = sum(r['api_failed'] for r in results_list)
            total_extract_fail = sum(r['extraction_failed'] for r in results_list)

            print(f"{base_name + ' (combined)':<30} {total_entries:<8} {total_valid:<8} {combined_avg:>6.2f}       {total_api_failed:<12} {total_extract_fail:<15}")

            # Show individual variants indented
            for result in sorted(results_list, key=lambda x: x['model']):
                model = '  ' + result['model']
                total = result['total']
                valid = len(result['scores'])
                avg_score = result['average_score']
                api_failed = result['api_failed']
                extract_fail = result['extraction_failed']

                print(f"{model:<30} {total:<8} {valid:<8} {avg_score:>6.2f}       {api_failed:<12} {extract_fail:<15}")

    print("-"*120)


def print_score_distribution(combined_results: Dict[str, Dict]):
    """
    Print score distribution for each model.

    Args:
        combined_results: Dictionary of combined variant results
    """
    print("\n" + "="*120)
    print("SCORE DISTRIBUTION (1-5)")
    print("="*120)

    # Sort by average score
    sorted_models = sorted(combined_results.items(),
                          key=lambda x: x[1]['combined_average'],
                          reverse=True)

    for base_name, combined in sorted_models:
        scores = combined['combined_scores']
        if not scores:
            continue

        # Count distribution
        distribution = {i: 0 for i in range(1, 6)}
        for score in scores:
            distribution[score] += 1

        # Calculate percentages
        total = len(scores)
        percentages = {i: (count / total * 100) for i, count in distribution.items()}

        print(f"\n{base_name}:")
        print(f"  Average: {combined['combined_average']:.2f}")
        print(f"  Distribution: ", end="")
        for i in range(1, 6):
            print(f"{i}: {distribution[i]:>3} ({percentages[i]:>5.1f}%)  ", end="")
        print()


def print_warnings(all_results: List[Dict], max_warnings: int = 10):
    """
    Print warnings for failed extractions and API errors.

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

                elif warning_type == 'EXTRACTION_FAILED':
                    print(f"Judge response: {item['judge_response']}")

            if len(items) > max_warnings:
                print(f"    ... and {len(items) - max_warnings} more")


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_mime_type(image_path: str) -> str:
    """Get image MIME type."""
    ext = Path(image_path).suffix.lower()
    return {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/png')


def load_all_entries(judge_dir: Path) -> List[Dict]:
    """Load all entries from all JSON files."""
    judge_files = sorted(judge_dir.glob('*.json'))
    all_entries = []

    for judge_file in judge_files:
        model_name = judge_file.stem

        try:
            with open(judge_file, 'r') as f:
                data = json.load(f)

            # Add model name and extract score for each entry
            for entry in data:
                entry['model_name'] = model_name
                score = extract_quality_score(entry.get('consistency_check_response', ''))
                entry['quality_score'] = score

            all_entries.extend(data)

        except Exception as e:
            print(f"Warning: Could not load {judge_file}: {e}")

    return all_entries


def generate_html(all_entries: List[Dict], combined_results: Dict[str, Dict], output_file: str):
    """Generate HTML visualization of maze quality results."""

    # Sort by score (lowest first to highlight problems), then by model
    all_entries.sort(key=lambda x: (x.get('quality_score') or 0, x.get('model_name', '')))

    rows = []
    for entry in all_entries:
        index = entry.get('index', 'N/A')
        image_path = entry.get('image_path', '')
        original_image_path = entry.get('original_image_path', '')
        model_name = entry.get('model_name', 'N/A')
        judge_model = entry.get('consistency_check_model', 'N/A')
        consistency_response = entry.get('consistency_check_response', 'N/A')
        quality_score = entry.get('quality_score')
        success = entry.get('success', False)

        # Encode images
        img_tags = []
        for img_path, label in [(original_image_path, 'Original'), (image_path, 'Annotated')]:
            if img_path and os.path.exists(img_path):
                try:
                    base64_img = encode_image_to_base64(img_path)
                    mime = get_mime_type(img_path)
                    img_tag = f'<div style="text-align: center; margin-bottom: 5px;"><div style="font-size: 11px; color: #6b7280; margin-bottom: 3px;">{label}</div><img src="data:{mime};base64,{base64_img}" style="max-width: 250px; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;"></div>'
                    img_tags.append(img_tag)
                except:
                    img_tags.append(f'<p style="font-size: 12px; color: #ef4444;">Error loading {label}</p>')
            elif img_path:
                img_tags.append(f'<p style="font-size: 12px; color: #9ca3af;">{label} not found</p>')

        images_html = '<div style="display: flex; gap: 10px; flex-wrap: wrap;">' + ''.join(img_tags) + '</div>'

        # Format text
        if consistency_response:
            consistency_text = consistency_response.replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
        else:
            consistency_text = '<em>No response</em>'

        # Score styling
        if not success:
            score_display = '✗ Failed'
            score_color = '#ef4444'
            row_bg = '#fee2e2'
        elif quality_score is None:
            score_display = '? Unknown'
            score_color = '#f59e0b'
            row_bg = '#fef3c7'
        else:
            score_display = str(quality_score)
            # Color gradient: 1=red, 3=yellow, 5=green
            if quality_score <= 2:
                score_color = '#ef4444'
                row_bg = '#fee2e2'
            elif quality_score == 3:
                score_color = '#f59e0b'
                row_bg = '#fef3c7'
            else:
                score_color = '#10b981'
                row_bg = '#d1fae5'

        rows.append(f"""
        <tr style="background: {row_bg};">
            <td style="text-align: center; font-weight: bold; font-size: 14px;">{model_name}</td>
            <td style="text-align: center; font-weight: bold;">{index}</td>
            <td>{images_html}</td>
            <td style="text-align: center; font-size: 32px; font-weight: bold; color: {score_color};">{score_display}</td>
            <td style="font-size: 13px; max-width: 400px; line-height: 1.5;">{consistency_text}</td>
        </tr>
        """)

    # Generate summary stats
    total_entries = len(all_entries)
    scored_entries = sum(1 for e in all_entries if e.get('quality_score') is not None)
    failed_entries = sum(1 for e in all_entries if not e.get('success', False))

    # Score distribution across all entries
    score_dist = {i: 0 for i in range(1, 6)}
    for entry in all_entries:
        score = entry.get('quality_score')
        if score and 1 <= score <= 5:
            score_dist[score] += 1

    # Generate summary table HTML
    summary_rows = []
    sorted_models = sorted(combined_results.items(),
                          key=lambda x: x[1]['combined_average'],
                          reverse=True)

    for base_name, combined in sorted_models:
        avg = combined['combined_average']
        count = len(combined['combined_scores'])

        # Color based on average
        if avg >= 4:
            color = '#10b981'
        elif avg >= 3:
            color = '#f59e0b'
        else:
            color = '#ef4444'

        summary_rows.append(f"""
        <tr>
            <td style="font-weight: 600;">{base_name}</td>
            <td style="text-align: center; font-size: 18px; font-weight: bold; color: {color};">{avg:.2f}</td>
        </tr>
        """)

    summary_table = f"""
    <table style="width: 100%; max-width: 600px; margin: 20px auto;">
        <thead>
            <tr>
                <th>Model</th>
                <th style="width: 120px;">Avg Score</th>
            </tr>
        </thead>
        <tbody>
            {''.join(summary_rows)}
        </tbody>
    </table>
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Maze Quality Scores</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1f2937;
            text-align: center;
        }}
        .stats {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1200px;
            margin: 0 auto 20px auto;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: #f9fafb;
            border-radius: 6px;
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 13px;
            color: #6b7280;
            margin-top: 5px;
            font-weight: 500;
        }}
        .score-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #f9fafb;
            border-radius: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px 10px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
            font-weight: 600;
        }}
        td {{
            padding: 15px 10px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: top;
        }}
        tr:hover {{
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <h1>Maze Quality Score Analysis</h1>

    <div class="stats">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number">{total_entries}</div>
                <div class="stat-label">Total Entries</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #10b981;">{scored_entries}</div>
                <div class="stat-label">Successfully Scored</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #ef4444;">{failed_entries}</div>
                <div class="stat-label">API Failures</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #f59e0b;">{score_dist[1]}</div>
                <div class="stat-label">Score 1</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #f59e0b;">{score_dist[2]}</div>
                <div class="stat-label">Score 2</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #f59e0b;">{score_dist[3]}</div>
                <div class="stat-label">Score 3</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #10b981;">{score_dist[4]}</div>
                <div class="stat-label">Score 4</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #10b981;">{score_dist[5]}</div>
                <div class="stat-label">Score 5</div>
            </div>
        </div>

        <div class="score-legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444;"></div>
                <span>Poor (1-2)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f59e0b;"></div>
                <span>Fair (3)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981;"></div>
                <span>Good (4-5)</span>
            </div>
        </div>

        <h2 style="text-align: center; color: #1f2937; margin-top: 30px; margin-bottom: 10px;">Model Rankings</h2>
        {summary_table}
    </div>

    <table>
        <thead>
            <tr>
                <th style="width: 180px;">Model</th>
                <th style="width: 60px;">Index</th>
                <th style="width: 530px;">Images</th>
                <th style="width: 80px;">Score</th>
                <th>Judge Response</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Generated HTML: {output_file}")
    print(f"  Open in browser: file://{os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description='Calculate maze quality scores from judge outputs')
    parser.add_argument('--judge-dir', type=str, required=True,
                       help='Directory containing maze quality judge output files')
    parser.add_argument('--max-warnings', type=int, default=5,
                       help='Maximum warnings to show per type (default: 5)')
    parser.add_argument('--show-warnings', action='store_true',
                       help='Show detailed warnings')

    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)

    if not judge_dir.exists():
        print(f"Error: Directory not found: {judge_dir}")
        return

    # Find all judge output files
    judge_files = sorted(judge_dir.glob('*.json'))

    if not judge_files:
        print(f"No JSON files found in {judge_dir}")
        return

    print(f"Found {len(judge_files)} judge output files")

    # Analyze each file
    all_results = []
    for judge_file in judge_files:
        print(f"Analyzing {judge_file.name}...")
        results = analyze_quality_scores(judge_file)
        all_results.append(results)

    # Combine invalid/valid variants
    combined = combine_invalid_valid_results(all_results)

    # Print summary table
    print_summary_table(all_results, combined)

    # Print score distribution
    print_score_distribution(combined)

    # Print warnings if requested
    if args.show_warnings:
        print_warnings(all_results, max_warnings=args.max_warnings)
    else:
        print(f"\nUse --show-warnings to see detailed warnings")

    # Load all entries and generate HTML
    print("\nGenerating HTML visualization...")
    all_entries = load_all_entries(judge_dir)

    output_file = 'consistency/html_output/maze_quality_all_samples.html'
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_html(all_entries, combined, output_file)


if __name__ == '__main__':
    main()
