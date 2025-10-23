import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load results and labels
results_file = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/results.json"
labels_file = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/labels.json"

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
random_counts = count_labels(source_categories['random_source'])
outlines_counts = count_labels(source_categories['outlines_source'])
worksheets_counts = count_labels(source_categories['worksheets_source'])

# Calculate percentages
def counts_to_percentages(counts, total):
    if total == 0:
        return {k: 0 for k in counts}
    return {k: (v / total) * 100 for k, v in counts.items()}

total_images = len(all_items)
overall_pct = counts_to_percentages(overall_counts, total_images)
random_pct = counts_to_percentages(random_counts, len(source_categories['random_source']))
outlines_pct = counts_to_percentages(outlines_counts, len(source_categories['outlines_source']))
worksheets_pct = counts_to_percentages(worksheets_counts, len(source_categories['worksheets_source']))

# Print tables
print("=" * 80)
print("LABEL DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

print("OVERALL STATISTICS")
print("-" * 80)
print(f"{'Label':<25} {'Count':<10} {'Percentage':<15}")
print("-" * 80)
for label in ['good', 'no-change', 'hallucinated-points', 'improper-lines']:
    count = overall_counts[label]
    pct = overall_pct[label]
    print(f"{label:<25} {count:<10} {pct:>6.2f}%")
print("-" * 80)
total_labeled = sum(overall_counts[k] for k in ['good', 'no-change', 'hallucinated-points', 'improper-lines'])
print(f"{'TOTAL LABELED':<25} {total_labeled:<10}")
print()

print("BY SOURCE TYPE")
print("-" * 80)
print(f"{'Label':<25} {'Random':<12} {'Outlines':<12} {'Worksheets':<12}")
print("-" * 80)
for label in ['good', 'no-change', 'hallucinated-points', 'improper-lines']:
    random_str = f"{random_counts[label]} ({random_pct[label]:.1f}%)"
    outlines_str = f"{outlines_counts[label]} ({outlines_pct[label]:.1f}%)"
    worksheets_str = f"{worksheets_counts[label]} ({worksheets_pct[label]:.1f}%)"
    print(f"{label:<25} {random_str:<12} {outlines_str:<12} {worksheets_str:<12}")
print("-" * 80)
print()

# Get totals for plot titles
random_total = len(source_categories['random_source'])
outlines_total = len(source_categories['outlines_source'])
worksheets_total = len(source_categories['worksheets_source'])

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Label Distribution Analysis', fontsize=16, fontweight='bold')

label_names = ['Good', 'No Change', 'Hallucinated\nPoints', 'Improper\nLines']
label_keys = ['good', 'no-change', 'hallucinated-points', 'improper-lines']
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

# Plot 1: Overall distribution
ax1 = axes[0, 0]
values = [overall_pct[k] for k in label_keys]
bars1 = ax1.bar(label_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Distribution', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add vertical separator after "Good" column
ax1.axvline(x=0.5, color='black', linestyle=':', linewidth=2, alpha=0.7)

# Add value labels on bars
for bar, val in zip(bars1, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%\n({overall_counts[label_keys[values.index(val)]]})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Random source
ax2 = axes[0, 1]
values = [random_pct[k] for k in label_keys]
bars2 = ax2.bar(label_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title(f'Random Source (n={random_total})', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add vertical separator after "Good" column
ax2.axvline(x=0.5, color='black', linestyle=':', linewidth=2, alpha=0.7)

for bar, val in zip(bars2, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%\n({random_counts[label_keys[values.index(val)]]})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Outlines source
ax3 = axes[1, 0]
values = [outlines_pct[k] for k in label_keys]
bars3 = ax3.bar(label_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_title(f'Outlines Source (n={outlines_total})', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add vertical separator after "Good" column
ax3.axvline(x=0.5, color='black', linestyle=':', linewidth=2, alpha=0.7)

for bar, val in zip(bars3, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%\n({outlines_counts[label_keys[values.index(val)]]})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Worksheets source
ax4 = axes[1, 1]
values = [worksheets_pct[k] for k in label_keys]
bars4 = ax4.bar(label_names, values, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title(f'Worksheets Source (n={worksheets_total})', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 100)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add vertical separator after "Good" column
ax4.axvline(x=0.5, color='black', linestyle=':', linewidth=2, alpha=0.7)

for bar, val in zip(bars4, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%\n({worksheets_counts[label_keys[values.index(val)]]})',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/label_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
print()

plt.show()
