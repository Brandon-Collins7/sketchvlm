import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def process_directory(results_file, labels_file, title, output_path):
    """Process one directory and generate its plot"""

    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    # Categorize by source
    source_categories = {
        'random_source': [],
        'outlines_source': [],
        'worksheets_source': []
    }

    # Process each result
    for result in results:
        filename = result['original_filename']
        original_path = result['original_image_path']

        # Determine source category
        if 'random_source' in original_path:
            source = 'random_source'
        elif 'outlines_source' in original_path:
            source = 'outlines_source'
        elif 'worksheets_source' in original_path:
            source = 'worksheets_source'
        else:
            continue

        # Get labels for this image
        image_labels = labels.get(filename, [])
        if isinstance(image_labels, str):
            image_labels = [image_labels]

        source_categories[source].append({
            'filename': filename,
            'labels': image_labels
        })

    # Count label occurrences
    def count_labels(items):
        counts = {
            'good': 0,
            'no-change': 0,
            'hallucinated-points': 0,
            'improper-lines': 0
        }

        for item in items:
            for label in item['labels']:
                if label in counts:
                    counts[label] += 1

        return counts

    # Calculate overall counts
    all_items = []
    for source_items in source_categories.values():
        all_items.extend(source_items)

    overall_counts = count_labels(all_items)

    # Calculate percentages
    def counts_to_percentages(counts, total):
        if total == 0:
            return {k: 0 for k in counts}
        return {k: (v / total) * 100 for k, v in counts.items()}

    total_images = len(all_items)
    overall_pct = counts_to_percentages(overall_counts, total_images)

    # Print tables
    print("=" * 80)
    print(f"LABEL DISTRIBUTION ANALYSIS - {title}")
    print("=" * 80)
    print()

    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"{'Label':<35} {'Count':<10} {'Percentage':<15}")
    print("-" * 80)
    label_display = {
        'good': 'Success',
        'no-change': 'No Change',
        'hallucinated-points': 'Hallucinated Change in Image',
        'improper-lines': 'Improper Line Order'
    }
    for label in ['good', 'no-change', 'hallucinated-points', 'improper-lines']:
        count = overall_counts[label]
        pct = overall_pct[label]
        print(f"{label_display[label]:<35} {count:<10} {pct:>6.2f}%")
    print("-" * 80)
    total_labeled = sum(overall_counts[k] for k in ['good', 'no-change', 'hallucinated-points', 'improper-lines'])
    print(f"{'TOTAL LABELED':<25} {total_labeled:<10}")
    print()

    # Create plot - single overall distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Connect-the-Dots - {title}', fontsize=16, fontweight='bold')

    label_names = ['Success', 'No Change', 'Hallucinated\nChange in Image', 'Improper\nLine Order']
    label_keys = ['good', 'no-change', 'hallucinated-points', 'improper-lines']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

    # Overall distribution
    values = [overall_pct[k] for k in label_keys]
    bars = ax.bar(label_names, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add vertical separator after "Good" column
    ax.axvline(x=0.5, color='black', linestyle=':', linewidth=2, alpha=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}%\n({overall_counts[label_keys[values.index(val)]]})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")
    print()

    plt.close()


# Process both directories
print("\n" + "="*80)
print("GENERATING PLOTS FOR BOTH MODELS")
print("="*80 + "\n")

# Process nano (gemini-3-pro-image-preview)
process_directory(
    results_file="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/results.json",
    labels_file="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/labels.json",
    title="gemini-3-pro-image-preview",
    output_path="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/label_analysis.png"
)

# Process flash (gemini-2.5-flash-image)
process_directory(
    results_file="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano_25_flash/results.json",
    labels_file="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano_25_flash/labels.json",
    title="gemini-2.5-flash-image",
    output_path="/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano_25_flash/label_analysis.png"
)

print("="*80)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("="*80)
