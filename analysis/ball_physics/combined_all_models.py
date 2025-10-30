#!/usr/bin/env python3
"""
Combined GPT-5 and Qwen3 Analysis Plot
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_combined_plot():
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "analysis" / "ball_physics"

    # Load GPT-5 accuracy breakdown
    gpt5_acc = pd.read_csv(output_dir / "accuracy_breakdown.csv")

    # Load Qwen3 accuracy breakdown
    qwen3_acc = pd.read_csv(output_dir / "qwen3_accuracy_breakdown.csv")

    # Prepare data for plotting
    data = []

    # GPT-5 data
    for reasoning in ['low', 'med', 'high']:
        for type_name in ['direct_vqa', 'paths']:
            row = gpt5_acc[
                (gpt5_acc['type'] == type_name) &
                (gpt5_acc['reasoning'] == reasoning)
            ]
            if not row.empty:
                label = f"GPT-5-{reasoning}"
                data.append({
                    'model': label,
                    'type': 'Direct VQA' if type_name == 'direct_vqa' else 'SketchVLM',
                    'accuracy': row.iloc[0]['accuracy'] * 100,
                    'reasoning': reasoning
                })

    # Qwen3 data
    for model in ['qwen8b', 'qwen235b']:
        for type_name in ['direct_vqa', 'paths']:
            row = qwen3_acc[
                (qwen3_acc['type'] == type_name) &
                (qwen3_acc['model'] == model)
            ]
            if not row.empty:
                model_label = 'Qwen-8B' if model == 'qwen8b' else 'Qwen-235B'
                data.append({
                    'model': model_label,
                    'type': 'Direct VQA' if type_name == 'direct_vqa' else 'SketchVLM',
                    'accuracy': row.iloc[0]['accuracy'] * 100,
                    'reasoning': None
                })

    df = pd.DataFrame(data)

    # Create pivot table
    pivot = df.pivot(index='model', columns='type', values='accuracy')

    # Reorder for better visualization
    model_order = ['GPT-5-low', 'GPT-5-med', 'GPT-5-high', 'Qwen-8B', 'Qwen-235B']
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(pivot.index))
    width = 0.35

    bars1 = ax.bar(x - width/2, pivot['Direct VQA'], width,
                   label='Direct VQA', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, pivot['SketchVLM'], width,
                   label='SketchVLM', color='#e74c3c', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Ball Physics: Direct VQA vs SketchVLM Across All Models',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0, ha='center')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add vertical line to separate GPT-5 from Qwen
    ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # Add text labels for sections
    ax.text(1, 95, 'GPT-5', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(3.5, 95, 'Qwen3', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "combined_all_models.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {plot_path}")

    plot_path_pdf = output_dir / "combined_all_models.pdf"
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Saved combined plot to: {plot_path_pdf}")

    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("COMBINED ACCURACY SUMMARY")
    print("="*80)
    print("\nGPT-5:")
    for idx in pivot.index:
        if 'GPT-5' in idx:
            print(f"  {idx:15s}: Direct VQA={pivot.loc[idx, 'Direct VQA']:.1f}%, "
                  f"SketchVLM={pivot.loc[idx, 'SketchVLM']:.1f}%")

    print("\nQwen3:")
    for idx in pivot.index:
        if 'Qwen' in idx:
            sketch = pivot.loc[idx, 'SketchVLM']
            sketch_str = f"{sketch:.1f}%" if not np.isnan(sketch) else "N/A"
            print(f"  {idx:15s}: Direct VQA={pivot.loc[idx, 'Direct VQA']:.1f}%, "
                  f"SketchVLM={sketch_str}")

if __name__ == "__main__":
    create_combined_plot()
