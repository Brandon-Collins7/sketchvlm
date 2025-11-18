#!/usr/bin/env python3
"""
Plot comparison of trajectory metrics across different models.

Usage:
    python3 plot_model_comparison.py --models gpt5_med gemini_25_pro gpt5_low gemini_25_flash --output all_models_comparison.png
    python3 plot_model_comparison.py --models gpt5_low gpt5_med
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def format_model_name(model_name):
    """
    Format model name for display.

    Parameters:
    -----------
    model_name : str
        Raw model name (e.g., 'gpt5_med')

    Returns:
    --------
    str
        Formatted model name (e.g., 'GPT-5 (medium)')
    """
    name_map = {
        'gpt5_low': 'GPT-5 (low)',
        'gpt5_med': 'GPT-5 (medium)',
        'gpt5_high': 'GPT-5 (high)',
        'gemini_25_flash': 'Gemini 2.5 Flash',
        'gemini_25_pro': 'Gemini 2.5 Pro',
        'qwen3_8b_thinking': 'Qwen3 8B Thinking',
        'qwen3_235b_thinking': 'Qwen3 235B Thinking'
    }
    return name_map.get(model_name, model_name)

def load_model_metrics(comparisons_dir, model_name):
    """
    Load metrics summary CSV for a given model.

    Parameters:
    -----------
    comparisons_dir : str
        Path to comparisons directory
    model_name : str
        Name of model subdirectory

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with metrics, or None if file doesn't exist
    """
    csv_path = os.path.join(comparisons_dir, model_name, "metrics_summary.csv")

    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for {model_name} at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    return df

def plot_model_comparison(models, comparisons_dir, output_path=None):
    """
    Create bar plot comparing average min distance and MSE across models.

    Parameters:
    -----------
    models : list of str
        List of model names to compare
    comparisons_dir : str
        Path to comparisons directory
    output_path : str, optional
        Path to save the plot
    """
    # Load data for all models
    model_data = {}
    for model in models:
        df = load_model_metrics(comparisons_dir, model)
        if df is not None:
            model_data[model] = df

    if not model_data:
        print("Error: No valid model data found!")
        return

    # Calculate statistics for each model
    stats = []
    for model_name, df in model_data.items():
        stats.append({
            'model': format_model_name(model_name),
            'avg_dist_mean': df['avg_min_distance'].mean(),
            'avg_dist_std': df['avg_min_distance'].std(),
            'mse_mean': df['mse_min_distance'].mean(),
            'mse_std': df['mse_min_distance'].std(),
            'n_files': len(df)
        })

    stats_df = pd.DataFrame(stats)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(stats_df))
    width = 0.6

    # Plot 1: Average Minimum Distance
    bars1 = ax1.bar(x, stats_df['avg_dist_mean'], width,
                    yerr=stats_df['avg_dist_std'],
                    capsize=5, alpha=0.8, color='steelblue',
                    edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Minimum Distance (pixels)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Minimum Distance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['model'], rotation=45, ha='right')
    ax1.set_ylim(0, 70)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, stats_df['avg_dist_mean'], stats_df['avg_dist_std'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{val:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: MSE Minimum Distance
    bars2 = ax2.bar(x, stats_df['mse_mean'], width,
                    yerr=stats_df['mse_std'],
                    capsize=5, alpha=0.8, color='coral',
                    edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE Minimum Distance (pixels²)', fontsize=12, fontweight='bold')
    ax2.set_title('MSE Minimum Distance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df['model'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars2, stats_df['mse_mean'], stats_df['mse_std'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 50,
                f'{val:.0f}±{std:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Print summary table
    print("\n" + "="*70)
    print("Model Comparison Summary")
    print("="*70)
    print(f"{'Model':<20} {'Avg Dist (px)':<20} {'MSE (px²)':<20} {'N Files':<10}")
    print("-"*70)
    for _, row in stats_df.iterrows():
        print(f"{row['model']:<20} {row['avg_dist_mean']:>8.2f} ± {row['avg_dist_std']:<8.2f} "
              f"{row['mse_mean']:>9.0f} ± {row['mse_std']:<8.0f} {row['n_files']:<10.0f}")
    print("="*70)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Compare trajectory metrics across different models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three models
  python3 plot_model_comparison.py --models gpt5_low gpt5_med gpt5_high

  # Compare and save to file
  python3 plot_model_comparison.py --models gpt5_low gpt5_med --output my_comparison.png

  # Use custom comparisons directory
  python3 plot_model_comparison.py --models model1 model2 --comparisons_dir /path/to/comparisons
        """
    )

    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to compare (subdirectories in comparisons/)')
    parser.add_argument('--comparisons_dir', type=str, default=None,
                       help='Path to comparisons directory (default: same dir as this script)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot image (if not provided, displays plot)')

    args = parser.parse_args()

    # Default to comparisons directory in same location as script
    if args.comparisons_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.comparisons_dir = os.path.join(script_dir, "comparisons")

    print(f"Comparisons directory: {args.comparisons_dir}")
    print(f"Models to compare: {', '.join(args.models)}")
    print()

    plot_model_comparison(args.models, args.comparisons_dir, args.output)

if __name__ == "__main__":
    main()
