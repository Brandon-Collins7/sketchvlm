#!/usr/bin/env python3
"""
Qwen3 Ball Physics Analysis
Loads all Qwen3 results (direct_vqa CSVs and paths JSONs) and computes accuracy by model size.
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
    """
    bucket_dict = {}
    lines_dict = {}

    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue

        image_name = subdir.name
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


def load_qwen_direct_vqa(csv_path: Path, model_name: str, gt_dict: Dict[str, int], lines_dict: Dict[str, int]) -> pd.DataFrame:
    """Load Qwen direct VQA results from CSV."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Add ground truth
    df['gold'] = df['image'].map(gt_dict)
    df['num_lines'] = df['image'].map(lines_dict)

    # Parse prediction
    if 'parsed_int' in df.columns:
        df['prediction'] = df['parsed_int']
    else:
        df['prediction'] = df.get('raw_text', '').apply(parse_bucket)

    df['correct'] = (df['prediction'] == df['gold']) & df['prediction'].notna() & df['gold'].notna()

    # Add metadata
    df['type'] = 'direct_vqa'
    df['model'] = model_name

    return df[['image', 'type', 'model', 'prediction', 'gold', 'num_lines', 'correct']]


def load_qwen_paths(paths_dir: Path, model_name: str, gt_dict: Dict[str, int], lines_dict: Dict[str, int]) -> pd.DataFrame:
    """Load Qwen path results from JSON files."""
    if not paths_dir.exists():
        print(f"Warning: {paths_dir} not found")
        return pd.DataFrame()

    rows = []

    for json_file in sorted(paths_dir.glob("item_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            source_image = data.get('source_image', '')
            if source_image:
                source_image = source_image.replace('\\', '/')
                image_name = Path(source_image).name
            else:
                continue

            prediction = data.get('answer')
            gold = gt_dict.get(image_name)
            num_lines = lines_dict.get(image_name)

            correct = (prediction == gold) if (prediction is not None and gold is not None) else False

            rows.append({
                'image': image_name,
                'type': 'paths',
                'model': model_name,
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
    """Compute accuracy breakdown by type and model."""
    df_valid = df[df['gold'].notna()].copy()

    # Overall
    overall = pd.DataFrame([{
        'type': 'ALL',
        'model': 'ALL',
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
    by_type['model'] = 'ALL'

    # By model
    by_model = df_valid.groupby('model').agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()
    by_model['type'] = 'ALL'

    # By type and model
    by_both = df_valid.groupby(['type', 'model']).agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()

    result = pd.concat([overall, by_type, by_model, by_both], ignore_index=True)
    result = result[['type', 'model', 'total', 'correct', 'accuracy']]
    result = result.sort_values(['type', 'model'])

    return result


def plot_model_comparison(accuracy_df: pd.DataFrame, output_dir: Path):
    """Create a bar plot comparing direct_vqa vs SketchVLM across model sizes."""
    plot_data = accuracy_df[
        (accuracy_df['type'].isin(['direct_vqa', 'paths'])) &
        (accuracy_df['model'].isin(['qwen8b', 'qwen235b']))
    ].copy()

    pivot = plot_data.pivot(index='model', columns='type', values='accuracy')
    pivot = pivot.reindex(['qwen8b', 'qwen235b'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(pivot.index))
    width = 0.35

    bars1 = ax.bar(x - width/2, pivot['direct_vqa'] * 100, width,
                   label='Direct VQA', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot['paths'] * 100, width,
                   label='SketchVLM', color='#e74c3c', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ball Physics: Qwen3 Direct VQA vs SketchVLM',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Qwen-8B', 'Qwen-235B'])
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    plot_path = output_dir / "qwen3_model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")

    plot_path_pdf = output_dir / "qwen3_model_comparison.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved plot to: {plot_path_pdf}")

    plt.close()


def plot_per_line_breakdown_qwen(df: pd.DataFrame, output_dir: Path):
    """Create plot showing accuracy breakdown by number of lines for Qwen models."""
    df_valid = df[df['gold'].notna() & df['num_lines'].notna()].copy()

    breakdown = df_valid.groupby(['type', 'model', 'num_lines']).agg(
        total=('correct', 'count'),
        correct=('correct', 'sum'),
        accuracy=('correct', 'mean')
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = ['qwen8b', 'qwen235b']
    model_labels = ['Qwen-8B', 'Qwen-235B']

    for idx, (model, label) in enumerate(zip(models, model_labels)):
        ax = axes[idx]

        data = breakdown[breakdown['model'] == model]
        pivot = data.pivot(index='num_lines', columns='type', values='accuracy')
        pivot = pivot.reindex([1, 2, 3])

        x = np.arange(len(pivot.index))
        width = 0.35

        if 'direct_vqa' in pivot.columns:
            bars1 = ax.bar(x - width/2, pivot['direct_vqa'] * 100, width,
                          label='Direct VQA', color='#3498db', alpha=0.8)
            for bar in bars1:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        if 'paths' in pivot.columns:
            bars2 = ax.bar(x + width/2, pivot['paths'] * 100, width,
                          label='SketchVLM', color='#e74c3c', alpha=0.8)
            for bar in bars2:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Number of Lines', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['1', '2', '3'])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if idx == 1:
            ax.legend(fontsize=10)

    plt.suptitle('Accuracy Breakdown by Number of Lines (Qwen3)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = output_dir / "qwen3_per_line_breakdown.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved per-line breakdown plot to: {plot_path}")

    plot_path_pdf = output_dir / "qwen3_per_line_breakdown.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved per-line breakdown plot to: {plot_path_pdf}")

    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("ACCURACY BY NUMBER OF LINES (QWEN3)")
    print("="*80)

    for model in models:
        print(f"\n{model.upper()}:")
        data = breakdown[breakdown['model'] == model]
        for _, row in data.iterrows():
            print(f"  {row['type']:12s} | {int(row['num_lines'])} line(s): "
                  f"{row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']*100:5.2f}%")


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / "results" / "mix_eval" / "ball_paths" / "qwen3"
    direct_vqa_dir = results_dir / "direct_vqa"
    dataset_dir = base_dir / "datasets" / "large_run_split"

    print("Loading ground truth from dataset metadata...")
    gt_dict, lines_dict = load_ground_truth(dataset_dir)
    print(f"Loaded {len(gt_dict)} ground truth labels")
    print(f"Loaded {len(lines_dict)} num_lines labels")

    all_dfs = []

    print("\nLoading Qwen3 direct_vqa results...")
    print("  - qwen8b")
    df = load_qwen_direct_vqa(direct_vqa_dir / "qwen8b_results.csv", "qwen8b", gt_dict, lines_dict)
    if not df.empty:
        all_dfs.append(df)
        print(f"    Loaded {len(df)} samples")

    print("  - qwen235b")
    df = load_qwen_direct_vqa(direct_vqa_dir / "qwen235b_results.csv", "qwen235b", gt_dict, lines_dict)
    if not df.empty:
        all_dfs.append(df)
        print(f"    Loaded {len(df)} samples")

    print("\nLoading Qwen3 paths results...")
    print("  - qwen8b")
    df = load_qwen_paths(results_dir / "qwen3_8b_thinking_ball_paths", "qwen8b", gt_dict, lines_dict)
    if not df.empty:
        all_dfs.append(df)
        print(f"    Loaded {len(df)} samples")

    print("  - qwen235b")
    df = load_qwen_paths(results_dir / "qwen3_235b_thinking_ball_paths", "qwen235b", gt_dict, lines_dict)
    if not df.empty:
        all_dfs.append(df)
        print(f"    Loaded {len(df)} samples")

    if not all_dfs:
        print("No data loaded!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined_df)}")
    print(f"Samples with ground truth: {combined_df['gold'].notna().sum()}")

    output_dir = base_dir / "analysis" / "ball_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "qwen3_combined_results.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined results to: {combined_path}")

    print("\n" + "="*80)
    print("ACCURACY BREAKDOWN")
    print("="*80)

    accuracy_df = compute_accuracy_breakdown(combined_df)
    accuracy_path = output_dir / "qwen3_accuracy_breakdown.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"\nSaved accuracy breakdown to: {accuracy_path}")

    print("\n" + accuracy_df.to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY BY TYPE AND MODEL")
    print("="*80)

    for type_name in ['direct_vqa', 'paths']:
        print(f"\n{type_name.upper()}:")
        type_df = accuracy_df[(accuracy_df['type'] == type_name) & (accuracy_df['model'] != 'ALL')]
        if not type_df.empty:
            for _, row in type_df.iterrows():
                print(f"  {row['model']:10s}: {row['correct']:3.0f}/{row['total']:3.0f} = {row['accuracy']*100:5.2f}%")

    print("\nOVERALL:")
    overall_row = accuracy_df[(accuracy_df['type'] == 'ALL') & (accuracy_df['model'] == 'ALL')].iloc[0]
    print(f"  {overall_row['correct']:.0f}/{overall_row['total']:.0f} = {overall_row['accuracy']*100:.2f}%")

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_model_comparison(accuracy_df, output_dir)
    plot_per_line_breakdown_qwen(combined_df, output_dir)

    return combined_df, accuracy_df


if __name__ == "__main__":
    combined_df, accuracy_df = main()
