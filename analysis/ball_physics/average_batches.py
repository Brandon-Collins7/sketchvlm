#!/usr/bin/env python3
"""
Average batch1 and batch2 accuracy results together.
"""

import pandas as pd
from pathlib import Path


def average_batches():
    """Average accuracy results from batch1 and batch2."""

    # Load the pivot tables
    batch1 = pd.read_csv('analysis/ball_physics/batch1_accuracy_pivot.csv', index_col=0)
    batch2 = pd.read_csv('analysis/ball_physics/batch2_accuracy_pivot.csv', index_col=0)

    print("Batch 1 Accuracy:")
    print(batch1)
    print("\n" + "="*80 + "\n")

    print("Batch 2 Accuracy:")
    print(batch2)
    print("\n" + "="*80 + "\n")

    # Average the two batches
    # Only average models that appear in both batches
    common_models = batch1.index.intersection(batch2.index)

    averaged_data = []
    for model in common_models:
        averaged_data.append((batch1.loc[model] + batch2.loc[model]) / 2)

    averaged = pd.DataFrame(averaged_data, index=common_models, columns=batch1.columns)

    # Add delta column (SketchVLM - Direct VQA)
    if 'paths' in averaged.columns and 'direct_vqa' in averaged.columns:
        averaged['delta'] = (averaged['paths'] - averaged['direct_vqa']).round(1)

    print("Averaged Accuracy (%):")
    print(averaged)

    # Save the averaged results
    output_file = Path('analysis/ball_physics/averaged_accuracy.csv')
    averaged.to_csv(output_file)
    print(f"\n✓ Saved averaged results to: {output_file}")

    # Print formatted summary
    print("\n" + "="*80)
    print("AVERAGED ACCURACY SUMMARY")
    print("="*80)
    for model in averaged.index:
        print(f"\n{model}:")
        if 'direct_vqa' in averaged.columns:
            print(f"  Direct VQA : {averaged.loc[model, 'direct_vqa']:5.1f}%")
        if 'paths' in averaged.columns:
            print(f"  SketchVLM  : {averaged.loc[model, 'paths']:5.1f}%")
        if 'delta' in averaged.columns:
            print(f"  Delta      : {averaged.loc[model, 'delta']:+5.1f}%")

    return averaged


if __name__ == '__main__':
    averaged = average_batches()
