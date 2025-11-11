#!/usr/bin/env python3
"""
Evaluate two-turn ball path results from batch1 and batch2.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Regex patterns for parsing bucket numbers from $\boxed{}$ format
_BOXED_RE = re.compile(r"\$\\boxed\{\s*(\d+)\s*\}\$", re.IGNORECASE)
_LOOSE_RE = re.compile(r"\\boxed\{\s*(\d+)\s*\}", re.IGNORECASE)


def parse_bucket_from_answer(text: Optional[str]) -> Optional[int]:
    """Parse bucket number from answer field."""
    if not text:
        return None

    # Try strict $\boxed{} format first
    m = _BOXED_RE.search(text)
    if m:
        return int(m.group(1))

    # Try without $ signs
    m = _LOOSE_RE.search(text)
    if m:
        return int(m.group(1))

    return None


def load_ground_truth(dataset_dir: Path) -> Dict[str, int]:
    """
    Load ground truth labels from dataset metadata.

    Args:
        dataset_dir: Path to datasets/large_run_split

    Returns:
        Dictionary mapping image_name -> bucket_hit (1-4)
    """
    bucket_dict = {}

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

            # Store with .png extension to match source_image format
            img_key = f"{image_name}.png"

            if bucket_hit is not None and isinstance(bucket_hit, int):
                bucket_dict[img_key] = bucket_hit

        except Exception as e:
            print(f"Error loading {metadata_path}: {e}")
            continue

    return bucket_dict


def evaluate_batch(batch_dir: Path, batch_name: str, gt_dict: Dict[str, int]) -> pd.DataFrame:
    """
    Evaluate a single batch of results.

    Args:
        batch_dir: Path to the batch results directory
        batch_name: Name of the batch (for reporting)
        gt_dict: Ground truth dictionary

    Returns:
        DataFrame with evaluation results
    """
    rows = []

    # Find all JSON files
    json_files = sorted(batch_dir.glob("item_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract source image name
            source_image = data.get('source_image', '')
            if source_image:
                source_image = source_image.replace('\\', '/')
                image_name = Path(source_image).name
            else:
                continue

            # Extract model answer from 'answer' field
            answer_text = data.get('answer', '')
            prediction = parse_bucket_from_answer(answer_text)

            # Get ground truth
            gold = gt_dict.get(image_name)

            # Compute correctness
            correct = (prediction == gold) if (prediction is not None and gold is not None) else False

            rows.append({
                'batch': batch_name,
                'item': json_file.stem,
                'image': image_name,
                'answer_text': answer_text,
                'prediction': prediction,
                'gold': gold,
                'correct': correct
            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return pd.DataFrame(rows)


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    results_base = base_dir / "results" / "mix_eval" / "ball_paths"
    dataset_dir1 = base_dir / "datasets" / "large_run_split"
    dataset_dir2 = base_dir / "datasets" / "large_second_batch"
    output_dir = base_dir / "analysis" / "ball_physics"

    print("="*80)
    print("TWO-TURN BALL PATH EVALUATION - ALL MODELS")
    print("="*80)

    # Load ground truth from both datasets
    print("\nLoading ground truth from dataset metadata...")
    print("  Loading from large_run_split...")
    gt_dict1 = load_ground_truth(dataset_dir1)
    print(f"  Loaded {len(gt_dict1)} labels from batch 1 dataset")

    print("  Loading from large_second_batch...")
    gt_dict2 = load_ground_truth(dataset_dir2)
    print(f"  Loaded {len(gt_dict2)} labels from batch 2 dataset")

    # Combine both ground truth dictionaries
    gt_dict = {**gt_dict1, **gt_dict2}
    print(f"Total ground truth labels: {len(gt_dict)}")

    # Define all models to evaluate
    models = [
        ('gemini25_flash_ball_paths', 'gemini25_flash_ball_paths_batch2', 'Gemini 2.5 Flash'),
        ('gemini25_pro_ball_paths', 'gemini25_pro_ball_paths_batch2', 'Gemini 2.5 Pro'),
        ('gpt5_low_paths', 'gpt5_low_ball_paths_batch2', 'GPT-5 Low'),
        ('gpt5_med_ball_paths', 'gpt5_med_ball_paths_batch2', 'GPT-5 Med'),
    ]

    all_dfs = []

    # Evaluate each model
    for batch1_name, batch2_name, model_label in models:
        print("\n" + "="*80)
        print(f"EVALUATING {model_label.upper()}")
        print("="*80)

        # Batch 1
        batch1_dir = results_base / "batch1" / "two_turn" / batch1_name
        if batch1_dir.exists():
            print(f"Batch 1: Loading from {batch1_name}...")
            df = evaluate_batch(batch1_dir, f"batch1", gt_dict)
            df['model'] = model_label
            all_dfs.append(df)
            print(f"  Loaded {len(df)} samples")
        else:
            print(f"Warning: {batch1_dir} not found")

        # Batch 2
        batch2_dir = results_base / "batch2" / "two_turn" / batch2_name
        if batch2_dir.exists():
            print(f"Batch 2: Loading from {batch2_name}...")
            df = evaluate_batch(batch2_dir, f"batch2", gt_dict)
            df['model'] = model_label
            all_dfs.append(df)
            print(f"  Loaded {len(df)} samples")
        else:
            print(f"Warning: {batch2_dir} not found")

    # Combine all results
    if not all_dfs:
        print("Error: No data loaded!")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save detailed results
    output_path = output_dir / "two_turn_detailed_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")

    # Compute accuracy statistics
    print("\n" + "="*80)
    print("ACCURACY RESULTS")
    print("="*80)

    # Overall accuracy
    valid_df = combined_df[combined_df['gold'].notna()].copy()

    overall_total = len(valid_df)
    overall_correct = valid_df['correct'].sum()
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0

    print(f"\nOVERALL (All Models, Both Batches):")
    print(f"  Total:    {overall_total}")
    print(f"  Correct:  {overall_correct}")
    print(f"  Accuracy: {overall_accuracy:.2f}%")

    # Per-model accuracy
    print("\n" + "="*80)
    print("PER-MODEL RESULTS (Both Batches Combined)")
    print("="*80)
    for model_name in valid_df['model'].unique():
        model_df = valid_df[valid_df['model'] == model_name]
        model_total = len(model_df)
        model_correct = model_df['correct'].sum()
        model_accuracy = (model_correct / model_total * 100) if model_total > 0 else 0.0

        print(f"\n{model_name}:")
        print(f"  Total:    {model_total}")
        print(f"  Correct:  {int(model_correct)}")
        print(f"  Accuracy: {model_accuracy:.2f}%")

    # Per-model per-batch breakdown
    print("\n" + "="*80)
    print("PER-MODEL PER-BATCH BREAKDOWN")
    print("="*80)
    for model_name in valid_df['model'].unique():
        print(f"\n{model_name}:")
        for batch_name in ['batch1', 'batch2']:
            batch_df = valid_df[(valid_df['model'] == model_name) & (valid_df['batch'] == batch_name)]
            batch_total = len(batch_df)
            batch_correct = batch_df['correct'].sum()
            batch_accuracy = (batch_correct / batch_total * 100) if batch_total > 0 else 0.0

            print(f"  {batch_name}: {int(batch_correct)}/{batch_total} = {batch_accuracy:.2f}%")

    # Show some examples of errors
    print("\n" + "="*80)
    print("SAMPLE ERRORS (First 10)")
    print("="*80)

    errors = valid_df[~valid_df['correct']].head(10)
    if len(errors) > 0:
        for idx, row in errors.iterrows():
            print(f"\nItem: {row['item']} ({row['batch']})")
            print(f"  Image:      {row['image']}")
            print(f"  Predicted:  {row['prediction']}")
            print(f"  Gold:       {row['gold']}")
            print(f"  Answer:     {row['answer_text']}")
    else:
        print("\nNo errors found!")

    # Create summary CSV with per-model per-batch breakdown
    summary_data = []

    # Per-model per-batch
    for model_name in sorted(valid_df['model'].unique()):
        for batch_name in ['batch1', 'batch2']:
            batch_df = valid_df[(valid_df['model'] == model_name) & (valid_df['batch'] == batch_name)]
            if len(batch_df) > 0:
                summary_data.append({
                    'model': model_name,
                    'batch': batch_name,
                    'total': len(batch_df),
                    'correct': int(batch_df['correct'].sum()),
                    'incorrect': int((~batch_df['correct']).sum()),
                    'accuracy': (batch_df['correct'].sum() / len(batch_df) * 100)
                })

        # Per-model overall (both batches)
        model_df = valid_df[valid_df['model'] == model_name]
        summary_data.append({
            'model': model_name,
            'batch': 'both',
            'total': len(model_df),
            'correct': int(model_df['correct'].sum()),
            'incorrect': int((~model_df['correct']).sum()),
            'accuracy': (model_df['correct'].sum() / len(model_df) * 100)
        })

    # Add overall (all models, all batches)
    summary_data.append({
        'model': 'ALL',
        'batch': 'both',
        'total': overall_total,
        'correct': int(overall_correct),
        'incorrect': int(overall_total - overall_correct),
        'accuracy': overall_accuracy
    })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "two_turn_accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    return combined_df, summary_df


if __name__ == "__main__":
    combined_df, summary_df = main()
