# Line Visualization Comparison Tool

Compares SVG ball trajectory predictions with ground truth physics simulations.

## Features

- Samples points uniformly by arc-length from both SVG and ground truth trajectories
- Computes average minimum distance and MSE metrics
- Generates comparison visualizations with background scene images
- Organizes outputs by model name for easy comparison

## Usage

### Basic Usage (Default: gpt5_low)

```bash
python3 analysis/line_visualizations/compare_lines.py
```

### Compare Different Models

```bash
# GPT-5 Medium
python3 analysis/line_visualizations/compare_lines.py \
  --svg_dir /Users/log/Github/sketchvlm/results/mix_eval/ball_paths/gpt5/gpt5_med_ball_paths

# Custom output name
python3 analysis/line_visualizations/compare_lines.py \
  --svg_dir /path/to/your/svg/files \
  --output_name my_model_name \
  --n_points 200
```

### Arguments

- `--svg_dir`: Directory containing SVG files and their JSON metadata (default: gpt5_low_ball_paths)
- `--n_points`: Number of points to sample per trajectory (default: 100)
- `--output_name`: Custom name for output subdirectory (auto-detected if not provided)

## Output Structure

```
analysis/line_visualizations/
├── comparisons/
│   ├── gpt5_low/
│   │   ├── item_00000_comparison.png
│   │   ├── item_00001_comparison.png
│   │   ├── ...
│   │   └── metrics_summary.csv
│   ├── gpt5_med/
│   │   └── ...
│   └── your_model/
│       └── ...
└── compare_lines.py
```

## Output Files

Each comparison generates:
- **PNG images**: Visual comparison of trajectories overlaid on source scene
- **metrics_summary.csv**: Summary of all distance metrics per file

## Metrics

- **Average Minimum Distance**: Mean of the minimum distance from each SVG point to nearest GT point
- **MSE Minimum Distance**: Mean squared error of minimum distances (penalizes large errors)

## Git Ignore

All comparison outputs are automatically git-ignored (see .gitignore in project root).
