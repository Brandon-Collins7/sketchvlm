import numpy as np
from svgpathtools import parse_path
import xml.etree.ElementTree as ET
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
def extract_points_from_svg(svg_path, n_points=500, target_size=512):
    """
    Extract n_points uniformly sampled along the entire SVG path.

    Scales coordinates from SVG canvas space to target_size x target_size space
    to match ground truth coordinate system.

    Parameters:
    -----------
    svg_path : str
        Path to the SVG file
    n_points : int
        Total number of points to sample uniformly along entire trajectory
    target_size : int
        Target canvas size (e.g., 512 for 512x512 ground truth space)

    Returns:
    --------
    np.ndarray
        Array of shape (n_points, 2) containing [x, y] coordinates scaled to target space
    """
    import xml.etree.ElementTree as ET
    from svgpathtools import parse_path, Path
    import numpy as np

    # Parse the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Get SVG canvas dimensions for scaling
    svg_width = float(root.get('width', 2091))
    svg_height = float(root.get('height', 2091))

    # Calculate scale factors to convert to target space (e.g., 512x512)
    scale_x = target_size / svg_width
    scale_y = target_size / svg_height

    # Handle XML namespaces
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    paths = root.findall('.//path') + root.findall('.//svg:path', namespaces)

    if not paths:
        raise ValueError("No path elements found in SVG")

    # Parse and concatenate all paths into one continuous path
    all_segments = []
    for path_elem in paths:
        d = path_elem.get('d')
        if d:
            path = parse_path(d)
            all_segments.extend(path)  # Add all segments from this path

    if not all_segments:
        raise ValueError("No valid path data found")

    # Create one continuous path from all segments
    continuous_path = Path(*all_segments)

    # Get total length
    total_length = continuous_path.length()

    # Sample uniformly by arc length
    points = []
    for i in range(n_points):
        # Distance along the path
        distance = (i / (n_points - 1)) * total_length if n_points > 1 else 0

        # Convert distance to parameter t
        t = continuous_path.ilength(distance)

        # Get point at parameter t
        pt = continuous_path.point(t)

        # Scale to target coordinate space
        scaled_x = pt.real * scale_x
        scaled_y = pt.imag * scale_y
        points.append([scaled_x, scaled_y])

    return np.array(points)

def extract_points_from_gt_json(json_path, n_points=100, ball_body_index=None):
    """
    Extract n_points uniformly sampled by arc-length from ground truth trajectory JSON.

    Samples uniformly along the physical path distance (arc length), not by time/index.
    This matches the SVG sampling method for fair comparison.

    Transforms coordinates from physics coordinate system (origin bottom-left, Y up)
    to SVG coordinate system (origin top-left, Y down).

    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing trajectories
    n_points : int
        Number of points to sample uniformly along arc length
    ball_body_index : str or None
        The key in the trajectories dict for the ball. If None, automatically finds the ball.

    Returns:
    --------
    np.ndarray
        Array of shape (n_points, 2) containing [x, y] coordinates in SVG space
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get scene height for coordinate transformation
    scene_height = data['scene']['height']

    # Find the ball trajectory
    # If ball_body_index not specified, find it automatically
    if ball_body_index is None:
        trajectories = data.get('trajectories', {})
        if not trajectories:
            raise ValueError("No trajectories found in GT JSON")

        # Look for common ball body indices
        for possible_key in ['10', '9', '8', '11', '7']:
            if possible_key in trajectories:
                ball_body_index = possible_key
                break

        # If still not found, just take the first trajectory
        if ball_body_index is None:
            ball_body_index = list(trajectories.keys())[0]

    # Extract the trajectory points for the ball
    trajectory = data['trajectories'][ball_body_index]

    # Convert to numpy array and transform Y coordinates to SVG space
    # Physics: origin at bottom-left, Y increases upward
    # SVG: origin at top-left, Y increases downward
    # Transformation: svg_y = scene_height - physics_y
    all_points = np.array([[pt['x'], scene_height - pt['y']] for pt in trajectory])

    # Total number of points in ground truth
    total_points = len(all_points)

    if n_points >= total_points:
        # If requesting more points than available, return all points
        return all_points

    # Calculate cumulative arc length along the trajectory
    # distances[i] = distance between point i and point i+1
    distances = np.linalg.norm(np.diff(all_points, axis=0), axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_distances[-1]

    # Sample uniformly along arc length
    target_distances = np.linspace(0, total_length, n_points)

    # Interpolate to find points at target distances
    sampled_points = np.zeros((n_points, 2))
    for i, target_dist in enumerate(target_distances):
        # Find the segment containing this target distance
        idx = np.searchsorted(cumulative_distances, target_dist)

        if idx == 0:
            sampled_points[i] = all_points[0]
        elif idx >= len(all_points):
            sampled_points[i] = all_points[-1]
        else:
            # Linear interpolation between points idx-1 and idx
            seg_start_dist = cumulative_distances[idx - 1]
            seg_end_dist = cumulative_distances[idx]
            seg_length = seg_end_dist - seg_start_dist

            if seg_length > 0:
                # How far along this segment?
                t = (target_dist - seg_start_dist) / seg_length
                sampled_points[i] = (1 - t) * all_points[idx - 1] + t * all_points[idx]
            else:
                sampled_points[i] = all_points[idx - 1]

    return sampled_points

def visualize_trajectory(points, title="Sampled Trajectory", save_path=None):
    """Visualize the sampled trajectory points"""
    import matplotlib.pyplot as plt
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot 1: Full trajectory with all points
    ax1.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(points[0, 0], points[0, 1], c='blue', s=200, zorder=5, 
                marker='o', edgecolors='black', linewidths=2, label='Start')
    ax1.scatter(points[-1, 0], points[-1, 1], c='red', s=200, zorder=5, 
                marker='s', edgecolors='black', linewidths=2, label='End')
    ax1.scatter(points[::1, 0], points[::1, 1], c='orange', s=50, zorder=4, 
                alpha=0.6, label='Sample points')
    ax1.invert_yaxis()  # SVG Y-axis points down
    ax1.axis('equal')
    ax1.set_title(f'{title}\n({len(points)} points total)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"Saved visualization to {save_path}")
    
    plt.show()

def compute_average_min_distance(svg_points, gt_points):
    """
    Compute average minimum distance and MSE from SVG points to GT points.

    For each SVG point, finds the closest GT point and computes the distance.
    Then calculates both the average and MSE (Mean Squared Error) of these minimum distances.

    Parameters:
    -----------
    svg_points : np.ndarray
        Array of shape (n, 2) containing SVG trajectory points
    gt_points : np.ndarray
        Array of shape (m, 2) containing ground truth trajectory points

    Returns:
    --------
    tuple of (float, float)
        (average_min_distance, mse_min_distance)
    """
    min_distances = []

    for svg_pt in svg_points:
        # Compute distances from this SVG point to all GT points
        distances = np.linalg.norm(gt_points - svg_pt, axis=1)
        # Find the minimum distance
        min_dist = np.min(distances)
        min_distances.append(min_dist)

    # Convert to numpy array for easier computation
    min_distances = np.array(min_distances)

    # Average all minimum distances
    avg_min_distance = np.mean(min_distances)

    # MSE of minimum distances
    mse_min_distance = np.mean(min_distances ** 2)

    return avg_min_distance, mse_min_distance

def visualize_comparison(svg_points, gt_points, title="SVG vs Ground Truth Comparison", save_path=None, background_image_path=None):
    """
    Visualize both SVG trajectory and ground truth trajectory for comparison

    Parameters:
    -----------
    svg_points : np.ndarray
        Points sampled from SVG path
    gt_points : np.ndarray
        Points sampled from ground truth trajectory
    title : str
        Title for the plot
    save_path : str
        Optional path to save the figure
    background_image_path : str
        Optional path to background image to overlay trajectories on
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    fig, ax = plt.subplots(figsize=(16, 10))

    # Display background image if provided
    if background_image_path and os.path.exists(background_image_path):
        try:
            img = Image.open(background_image_path)
            # Display image in 512x512 coordinate space (0,0 at top-left)
            ax.imshow(img, extent=[0, 512, 512, 0], aspect='auto', alpha=0.5, zorder=1)
        except Exception as e:
            print(f"Warning: Could not load background image: {e}")

    # Plot ground truth trajectory
    ax.plot(gt_points[:, 0], gt_points[:, 1], 'b-', linewidth=3, alpha=0.8, label='GT Trajectory', zorder=3)
    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=50, zorder=4,
                alpha=0.6, label=f'GT points (n={len(gt_points)})')
    ax.scatter(gt_points[0, 0], gt_points[0, 1], c='blue', s=200, zorder=5,
                marker='o', edgecolors='black', linewidths=2, label='GT Start')
    ax.scatter(gt_points[-1, 0], gt_points[-1, 1], c='darkblue', s=200, zorder=5,
                marker='s', edgecolors='black', linewidths=2, label='GT End')

    # Plot SVG trajectory
    ax.plot(svg_points[:, 0], svg_points[:, 1], 'g-', linewidth=3, alpha=0.8, label='SVG Trajectory', zorder=3)
    ax.scatter(svg_points[:, 0], svg_points[:, 1], c='green', s=50, zorder=4,
                alpha=0.6, label=f'SVG points (n={len(svg_points)})')
    ax.scatter(svg_points[0, 0], svg_points[0, 1], c='green', s=200, zorder=5,
                marker='o', edgecolors='black', linewidths=2, label='SVG Start')
    ax.scatter(svg_points[-1, 0], svg_points[-1, 1], c='darkgreen', s=200, zorder=5,
                marker='s', edgecolors='black', linewidths=2, label='SVG End')

    # Set axis limits and properties
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # Inverted Y-axis to match SVG coordinate system
    ax.axis('equal')
    ax.set_title(f'{title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()

def process_all_svg_files(svg_dir, output_dir, n_points=100):
    """
    Process all SVG files in a directory, finding their GT pairs and computing metrics.

    Parameters:
    -----------
    svg_dir : str
        Directory containing SVG files and their corresponding JSON metadata
    output_dir : str
        Directory to save comparison visualizations and metrics
    n_points : int
        Number of points to sample from each trajectory

    Returns:
    --------
    list of dict
        List of results for each SVG-GT pair
    """
    import os
    import glob
    import csv

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all SVG files
    svg_files = sorted(glob.glob(os.path.join(svg_dir, "*.svg")))
    print(f"Found {len(svg_files)} SVG files to process\n")

    results = []

    for svg_file in svg_files:
        # Get base name (e.g., "item_00000")
        base_name = os.path.splitext(os.path.basename(svg_file))[0]
        json_file = os.path.join(svg_dir, f"{base_name}.json")

        # Read source_image from JSON to find GT file
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {base_name}, skipping...")
            continue

        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            source_image = metadata.get('source_image', '')
            if not source_image:
                print(f"Warning: No source_image in {base_name}.json, skipping...")
                continue

            # Extract GT identifier (e.g., "run_001_1" from "datasets/ball_path/run_001_1.png")
            gt_identifier = os.path.splitext(os.path.basename(source_image))[0]

            # Find GT JSON file in datasets/large_run_split/
            gt_json_path = os.path.join("/Users/log/Github/sketchvlm/datasets/large_run_split",
                                       gt_identifier, "random_scene_metadata.json")

            if not os.path.exists(gt_json_path):
                print(f"Warning: GT file not found at {gt_json_path}, skipping...")
                continue

            print(f"Processing {base_name}...")
            print(f"  SVG: {svg_file}")
            print(f"  GT:  {gt_json_path}")

            # Extract points
            svg_points = extract_points_from_svg(svg_file, n_points=n_points, target_size=512)
            gt_points = extract_points_from_gt_json(gt_json_path, n_points=n_points)

            # Compute metrics
            avg_min_dist, mse_min_dist = compute_average_min_distance(svg_points, gt_points)

            print(f"  Average min distance: {avg_min_dist:.2f} pixels")
            print(f"  MSE min distance: {mse_min_dist:.2f} pixels²")

            # Build path to source image
            source_image_path = os.path.join("/Users/log/Github/sketchvlm", source_image)
            if not os.path.exists(source_image_path):
                print(f"  Warning: Source image not found at {source_image_path}")
                source_image_path = None

            # Save visualization
            vis_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            visualize_comparison(svg_points, gt_points,
                               title=f"{base_name} - SVG vs GT (Avg Dist: {avg_min_dist:.2f}, MSE Dist: {mse_min_dist:.2f})",
                               save_path=vis_path,
                               background_image_path=source_image_path)

            # Store results
            results.append({
                'svg_file': base_name,
                'gt_identifier': gt_identifier,
                'avg_min_distance': avg_min_dist,
                'mse_min_distance': mse_min_dist,
                'n_points': n_points
            })

            print()

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            print()
            continue

    # Save summary metrics to CSV
    if results:
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['svg_file', 'gt_identifier', 'avg_min_distance', 'mse_min_distance', 'n_points'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved summary metrics to {csv_path}")

        # Print summary statistics
        avg_distances = [r['avg_min_distance'] for r in results]
        mse_distances = [r['mse_min_distance'] for r in results]
        print(f"\nSummary Statistics ({len(results)} files):")
        print(f"  Average min distance: {np.mean(avg_distances):.2f} ± {np.std(avg_distances):.2f} pixels")
        print(f"  MSE min distance: {np.mean(mse_distances):.2f} ± {np.std(mse_distances):.2f} pixels²")

    return results

# Usage
if __name__ == "__main__":
    import numpy as np
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Compare SVG ball trajectories with ground truth')
    parser.add_argument('--svg_dir', type=str,
                       default='/Users/log/Github/sketchvlm/results/mix_eval/ball_paths/gpt5/gpt5_low_ball_paths',
                       help='Directory containing SVG files to compare')
    parser.add_argument('--n_points', type=int, default=100,
                       help='Number of points to sample from each trajectory')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Name for output subdirectory (auto-detected from path if not provided)')

    args = parser.parse_args()

    svg_dir = args.svg_dir

    # Auto-detect model name from path if not provided
    if args.output_name:
        output_name = args.output_name
    else:
        # Extract name from path (e.g., "gpt5_low_ball_paths" -> "gpt5_low")
        dir_name = os.path.basename(svg_dir)
        output_name = dir_name.replace('_ball_paths', '')

    # Create output directory organized by model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "comparisons", output_name)

    print("="*60)
    print("Batch Processing: SVG vs Ground Truth Trajectory Comparison")
    print("="*60)
    print(f"SVG Directory: {svg_dir}")
    print(f"Output Name: {output_name}")
    print(f"Output Directory: {output_dir}")
    print(f"Sampling {args.n_points} points per trajectory")
    print("="*60)
    print()

    results = process_all_svg_files(svg_dir, output_dir, n_points=args.n_points)