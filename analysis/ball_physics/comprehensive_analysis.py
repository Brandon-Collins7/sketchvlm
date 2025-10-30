#!/usr/bin/env python3
"""
Comprehensive Ball Physics Analysis
Loads all results (direct_vqa CSVs and paths JSONs) and computes accuracy by type and reasoning level.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Regex patterns for parsing bucket numbers
_BOXED_RE = re.compile(r"\$\\boxed\{\s*(1|2|3|4|none)\s*\}\$", re.IGNORECASE)
_LOOSE_RE = re.compile(r"\\boxed\{\s*(1|2|3|4|none)\s*\}", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"\b(1|2|3|4|none)\b", re.IGNORECASE)


def parse_bucket(text: Optional[str]) -> Optional[int]:
    """Parse bucket number from text, return as integer."""
    if not text:
        return None

    # Try strict $\boxed{} format first
    m = _BOXED_RE.search(text)
    if m:
        val = m.group(1)
        if val.isdigit():
            return int(val)
        return None

    # Try without $ signs
    m = _LOOSE_RE.search(text)
    if m:
        val = m.group(1)
        if val.isdigit():
            return int(val)
        return None

    # Fallback to any digit
    m = _FALLBACK_RE.search(text)
    if m:
        val = m.group(1)
        if val.isdigit():
            return int(val)
        return None

    return None


def load_ground_truth(dataset_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Load ground truth labels from dataset metadata.

    Args:
        dataset_dir: Path to datasets/large_run_split

    Returns:
        Tuple of (bucket_hit_dict, num_lines_dict)
        - bucket_hit_dict: Dictionary mapping image_name -> bucket_hit (1-4)
        - num_lines_dict: Dictionary mapping image_name -> num_lines (1-3)
    """
    bucket_dict = {}
    lines_dict = {}

    # Iterate through all subdirectories
    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue

        image_name = subdir.name  # e.g., "run_001_1"
        metadata_path = subdir / "random_scene_metadata.json"

        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            simulation = data.get('simulation', {})
            bucket_hit = simulation.get('bucket_hit')
            num_lines = simulation.get('num_lines')

            img_key = f"{image_name}.png"

            if bucket_hit is not None and isinstance(bucket_hit, int):
                bucket_dict[img_key] = bucket_hit

            if num_lines is not None and isinstance(num_lines, int):
                lines_dict[img_key] = num_lines

        except Exception as e:
            print(f"Error loading {metadata_path}: {e}")
            continue

    return bucket_dict, lines_dict


def load_direct_vqa_results(results_dir: Path, reasoning_level: str, gt_dict: Dict[str, int], lines_dict: Dict[str, int]) -> pd.DataFrame:
    """
    Load direct VQA results from CSV file.

    Args:
        results_dir: Path to results/mix_eval/ball_paths/gpt5/direct_vqa
        reasoning_level: One of 'low', 'med', 'high'
        gt_dict: Ground truth dictionary
        lines_dict: Number of lines dictionary

    Returns:
        DataFrame with columns: image, type, reasoning, prediction, gold, num_lines, correct
    """
    csv_path = results_dir / f"ball_number_gpt5_{reasoning_level}.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()

    # Read CSV
    df = pd.read_csv(csv_path)

    # Add ground truth
    df['gold'] = df['image'].map(gt_dict)
    df['num_lines'] = df['image'].map(lines_dict)

    # Compute correct if not already present or if empty
    if 'parsed_int' in df.columns:
        df['prediction'] = df['parsed_int']
    else:
        # Parse from raw_text if needed
        df['prediction'] = df.get('raw_text', '').apply(parse_bucket)

    df['correct'] = (df['prediction'] == df['gold']) & df['prediction'].notna() & df['gold'].notna()

    # Add metadata
    df['type'] = 'direct_vqa'
    df['reasoning'] = reasoning_level

    return df[['image', 'type', 'reasoning', 'prediction', 'gold', 'num_lines', 'correct']]


def load_paths_results(results_dir: Path, reasoning_level: str, gt_dict: Dict[str, int], lines_dict: Dict[str, int]) -> pd.DataFrame:
    """
    Load path-based results from JSON files.

    Args:
        results_dir: Path to results/mix_eval/ball_paths/gpt5
        reasoning_level: One of 'low', 'med', 'high'
        gt_dict: Ground truth dictionary
        lines_dict: Number of lines dictionary

    Returns:
        DataFrame with columns: image, type, reasoning, prediction, gold, num_lines, correct
    """
    # Map reasoning level to directory name
    dir_map = {
        'low': 'gpt5_low_ball_paths',
        'med': 'gpt5_med_ball_paths',
        'high': 'gpt5_high_ball_paths'
    }

    paths_dir = results_dir / dir_map[reasoning_level]

    if not paths_dir.exists():
        print(f"Warning: {paths_dir} not found")
        return pd.DataFrame()

    rows = []

    # Load all item_*.json files
    for json_file in sorted(paths_dir.glob("item_*.json")):
        if json_file.name == "summary.json":
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract image name from source_image field
            source_image = data.get('source_image', '')
            if source_image:
                # Handle both forward and backward slashes
                source_image = source_image.replace('\\', '/')
                image_name = Path(source_image).name
            else:
                # Skip if no source image
                continue

            # Extract prediction from 'answer' field
            prediction = data.get('answer')

            # Get ground truth and num_lines
            gold = gt_dict.get(image_name)
            num_lines = lines_dict.get(image_name)

            # Compute correctness
            correct = (prediction == gold) if (prediction is not None and gold is not None) else False

            rows.append({
                'image': image_name,
                'type': 'paths',
                'reasoning': reasoning_level,
                'prediction': prediction,
                'gold': gold,
                'num_lines': num_lines,
                'correct': correct
            })

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return pd.DataFrame(rows)


def compute_accuracy_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy breakdown by type and reasoning level.

    Args:
        df: Combined dataframe with all results

    Returns:
        DataFrame with accuracy statistics
    """
    # Filter out rows where gold is missing
    df_valid = df[df['gold'].notna()].copy()

    # Overall accuracy
    overall = pd.DataFrame([{
        'type': 'ALL',
        'reasoning': 'ALL',
        'total': len(df_valid),
        'correct': df_valid['correct'].sum(),
        'accuracy': df_valid['correct'].mean() if len(df_valid) > 0 else 0.0
    }])

    # By type
    by_type = df_valid.groupby('type').agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()
    by_type['reasoning'] = 'ALL'

    # By reasoning
    by_reasoning = df_valid.groupby('reasoning').agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()
    by_reasoning['type'] = 'ALL'

    # By type and reasoning
    by_both = df_valid.groupby(['type', 'reasoning']).agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()

    # Combine all breakdowns
    result = pd.concat([overall, by_type, by_reasoning, by_both], ignore_index=True)

    # Reorder columns
    result = result[['type', 'reasoning', 'total', 'correct', 'accuracy']]

    # Sort for better readability
    result = result.sort_values(['type', 'reasoning'])

    return result


def plot_reasoning_comparison(accuracy_df: pd.DataFrame, output_dir: Path):
    """
    Create a bar plot comparing direct_vqa vs paths across reasoning levels.

    Args:
        accuracy_df: DataFrame with accuracy breakdown
        output_dir: Directory to save the plot
    """
    # Filter to only the rows we care about
    plot_data = accuracy_df[
        (accuracy_df['type'].isin(['direct_vqa', 'paths'])) &
        (accuracy_df['reasoning'].isin(['low', 'med', 'high']))
    ].copy()

    # Pivot for easier plotting
    pivot = plot_data.pivot(index='reasoning', columns='type', values='accuracy')

    # Reorder to low, med, high
    pivot = pivot.reindex(['low', 'med', 'high'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up bar positions
    x = np.arange(len(pivot.index))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, pivot['direct_vqa'] * 100, width,
                   label='Direct VQA', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot['paths'] * 100, width,
                   label='SketchVLM', color='#e74c3c', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Reasoning Effort', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ball Physics: GPT-5 Direct VQA vs SketchVLM Across Reasoning Levels',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "reasoning_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")

    # Also save as PDF
    plot_path_pdf = output_dir / "reasoning_comparison.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved plot to: {plot_path_pdf}")

    plt.close()


def plot_per_line_breakdown(df: pd.DataFrame, output_dir: Path):
    """
    Create a plot showing accuracy breakdown by number of lines for each method.

    Args:
        df: Combined dataframe with all results
        output_dir: Directory to save the plot
    """
    # Filter to only valid data
    df_valid = df[df['gold'].notna() & df['num_lines'].notna()].copy()

    # Group by type, reasoning, and num_lines
    breakdown = df_valid.groupby(['type', 'reasoning', 'num_lines']).agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()

    # Create a figure with 3 subplots (one for each reasoning level)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    reasoning_levels = ['low', 'med', 'high']
    reasoning_labels = ['Low', 'Medium', 'High']

    for idx, (reasoning, label) in enumerate(zip(reasoning_levels, reasoning_labels)):
        ax = axes[idx]

        # Filter data for this reasoning level
        data = breakdown[breakdown['reasoning'] == reasoning]

        # Pivot to get direct_vqa and paths side by side
        pivot = data.pivot(index='num_lines', columns='type', values='accuracy')

        # Make sure we have all line counts (1, 2, 3)
        pivot = pivot.reindex([1, 2, 3])

        # Create bars
        x = np.arange(len(pivot.index))
        width = 0.35

        if 'direct_vqa' in pivot.columns:
            bars1 = ax.bar(x - width/2, pivot['direct_vqa'] * 100, width,
                          label='Direct VQA', color='#3498db', alpha=0.8)
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        if 'paths' in pivot.columns:
            bars2 = ax.bar(x + width/2, pivot['paths'] * 100, width,
                          label='SketchVLM', color='#e74c3c', alpha=0.8)
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        # Customize subplot
        ax.set_xlabel('Number of Lines', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Reasoning', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['1', '2', '3'])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if idx == 2:  # Only show legend on the rightmost plot
            ax.legend(fontsize=10)

    plt.suptitle('Accuracy Breakdown by Number of Lines (GPT-5)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "per_line_breakdown.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved per-line breakdown plot to: {plot_path}")

    # Also save as PDF
    plot_path_pdf = output_dir / "per_line_breakdown.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved per-line breakdown plot to: {plot_path_pdf}")

    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("ACCURACY BY NUMBER OF LINES")
    print("="*80)

    for reasoning in reasoning_levels:
        print(f"\n{reasoning.upper()} REASONING:")
        data = breakdown[breakdown['reasoning'] == reasoning]
        for _, row in data.iterrows():
            print(f"  {row['type']:12s} | {int(row['num_lines'])} line(s): "
                  f"{row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']*100:5.2f}%")


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to sketchvlm root
    results_dir = base_dir / "results" / "mix_eval" / "ball_paths" / "gpt5"
    direct_vqa_dir = results_dir / "direct_vqa"
    dataset_dir = base_dir / "datasets" / "large_run_split"

    print("Loading ground truth from dataset metadata...")
    gt_dict, lines_dict = load_ground_truth(dataset_dir)
    print(f"Loaded {len(gt_dict)} ground truth labels")
    print(f"Loaded {len(lines_dict)} num_lines labels")

    # Load all results
    all_dfs = []

    print("\nLoading direct_vqa results...")
    for reasoning in ['low', 'med', 'high']:
        print(f"  - {reasoning}")
        df = load_direct_vqa_results(direct_vqa_dir, reasoning, gt_dict, lines_dict)
        if not df.empty:
            all_dfs.append(df)
            print(f"    Loaded {len(df)} samples")

    print("\nLoading paths results...")
    for reasoning in ['low', 'med', 'high']:
        print(f"  - {reasoning}")
        df = load_paths_results(results_dir, reasoning, gt_dict, lines_dict)
        if not df.empty:
            all_dfs.append(df)
            print(f"    Loaded {len(df)} samples")

    # Combine all results
    if not all_dfs:
        print("No data loaded!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined_df)}")
    print(f"Samples with ground truth: {combined_df['gold'].notna().sum()}")

    # Save combined results
    output_dir = base_dir / "analysis" / "ball_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "combined_results.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined results to: {combined_path}")

    # Compute accuracy breakdown
    print("\n" + "="*80)
    print("ACCURACY BREAKDOWN")
    print("="*80)

    accuracy_df = compute_accuracy_breakdown(combined_df)

    # Save accuracy breakdown
    accuracy_path = output_dir / "accuracy_breakdown.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"\nSaved accuracy breakdown to: {accuracy_path}")

    # Print accuracy breakdown
    print("\n" + accuracy_df.to_string(index=False))

    # Print formatted summary
    print("\n" + "="*80)
    print("SUMMARY BY TYPE AND REASONING")
    print("="*80)

    for type_name in ['direct_vqa', 'paths']:
        print(f"\n{type_name.upper()}:")
        type_df = accuracy_df[(accuracy_df['type'] == type_name) & (accuracy_df['reasoning'] != 'ALL')]
        if not type_df.empty:
            for _, row in type_df.iterrows():
                print(f"  {row['reasoning']:6s}: {row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']*100:5.2f}%")

    print("\nOVERALL:")
    overall_row = accuracy_df[(accuracy_df['type'] == 'ALL') & (accuracy_df['reasoning'] == 'ALL')].iloc[0]
    print(f"  {overall_row['correct']:.0f}/{overall_row['total']:.0f} = {overall_row['accuracy']*100:.2f}%")

    # Generate comparison plot
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_reasoning_comparison(accuracy_df, output_dir)

    # Generate per-line breakdown plot
    plot_per_line_breakdown(combined_df, output_dir)

    return combined_df, accuracy_df


if __name__ == "__main__":
    combined_df, accuracy_df = main()
