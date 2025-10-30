import numpy as np
from svgpathtools import parse_path
import xml.etree.ElementTree as ET
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
def extract_points_from_svg(svg_path, n_points=500):
    """
    Extract n_points uniformly sampled along the entire SVG path.
    
    Parameters:
    -----------
    svg_path : str
        Path to the SVG file
    n_points : int
        Total number of points to sample uniformly along entire trajectory
    
    Returns:
    --------
    np.ndarray
        Array of shape (n_points, 2) containing [x, y] coordinates
    """
    import xml.etree.ElementTree as ET
    from svgpathtools import parse_path, Path
    import numpy as np
    
    # Parse the SVG file
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
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
        points.append([pt.real, pt.imag])
    
    return np.array(points)

def extract_points_from_gt_json(json_path, n_points=100, ball_body_index="10"):
    """
    Extract n_points uniformly sampled from ground truth trajectory JSON.

    Transforms coordinates from physics coordinate system (origin bottom-left, Y up)
    to SVG coordinate system (origin top-left, Y down).

    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing trajectories
    n_points : int
        Number of points to sample uniformly from the trajectory
    ball_body_index : str
        The key in the trajectories dict for the ball (default "10")

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

    # Extract the trajectory points for the ball
    trajectory = data['trajectories'][ball_body_index]

    # Convert to numpy array and transform Y coordinates to SVG space
    # Physics: origin at bottom-left, Y increases upward
    # SVG: origin at top-left, Y increases downward
    # Transformation: svg_y = scene_height - physics_y
    all_points = np.array([[pt['x'], scene_height - pt['y']] for pt in trajectory])

    # Total number of points in ground truth
    total_points = len(all_points)

    # Uniformly sample n_points
    if n_points >= total_points:
        # If requesting more points than available, return all points
        return all_points

    # Calculate indices for uniform sampling
    indices = np.linspace(0, total_points - 1, n_points, dtype=int)
    sampled_points = all_points[indices]

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

def visualize_comparison(svg_points, gt_points, title="SVG vs Ground Truth Comparison", save_path=None):
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
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot ground truth trajectory
    ax.plot(gt_points[:, 0], gt_points[:, 1], 'b-', linewidth=2, alpha=0.5, label='GT Trajectory')
    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='blue', s=50, zorder=4,
                alpha=0.6, label=f'GT points (n={len(gt_points)})')
    ax.scatter(gt_points[0, 0], gt_points[0, 1], c='blue', s=200, zorder=5,
                marker='o', edgecolors='black', linewidths=2, label='GT Start')
    ax.scatter(gt_points[-1, 0], gt_points[-1, 1], c='darkblue', s=200, zorder=5,
                marker='s', edgecolors='black', linewidths=2, label='GT End')

    # Plot SVG trajectory
    ax.plot(svg_points[:, 0], svg_points[:, 1], 'g-', linewidth=2, alpha=0.5, label='SVG Trajectory')
    ax.scatter(svg_points[:, 0], svg_points[:, 1], c='green', s=50, zorder=4,
                alpha=0.6, label=f'SVG points (n={len(svg_points)})')
    ax.scatter(svg_points[0, 0], svg_points[0, 1], c='green', s=200, zorder=5,
                marker='o', edgecolors='black', linewidths=2, label='SVG Start')
    ax.scatter(svg_points[-1, 0], svg_points[-1, 1], c='darkgreen', s=200, zorder=5,
                marker='s', edgecolors='black', linewidths=2, label='SVG End')

    # Invert Y-axis to match SVG coordinate system
    ax.invert_yaxis()
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

# Usage
if __name__ == "__main__":
    import numpy as np

    svg_file = "/Users/log/Github/sketchvlm/results/mix_eval/ball_paths/gpt5/gpt5_low_ball_paths/item_00000.svg"
    gt_json_file = "/Users/log/Github/sketchvlm/datasets/large_run_split/run_001_1/random_scene_metadata.json"

    # Extract 100 points from SVG
    svg_points = extract_points_from_svg(svg_file, n_points=100)
    print(f"SVG: Extracted {len(svg_points)} points")
    print(f"  First point: {svg_points[0]}")
    print(f"  Last point: {svg_points[-1]}")

    # Extract 100 points from ground truth JSON
    gt_points = extract_points_from_gt_json(gt_json_file, n_points=100)
    print(f"\nGround Truth: Extracted {len(gt_points)} points")
    print(f"  First point: {gt_points[0]}")
    print(f"  Last point: {gt_points[-1]}")

    # Verify uniform spacing for SVG
    svg_distances = np.linalg.norm(np.diff(svg_points, axis=0), axis=1)
    print(f"\nSVG Spacing Statistics:")
    print(f"  Average spacing: {np.mean(svg_distances):.2f} pixels")
    print(f"  Std dev: {np.std(svg_distances):.2f} pixels")
    print(f"  Min spacing: {np.min(svg_distances):.2f} pixels")
    print(f"  Max spacing: {np.max(svg_distances):.2f} pixels")
    print(f"  Coefficient of variation: {(np.std(svg_distances)/np.mean(svg_distances)*100):.1f}%")

    # Verify uniform spacing for GT
    gt_distances = np.linalg.norm(np.diff(gt_points, axis=0), axis=1)
    print(f"\nGround Truth Spacing Statistics:")
    print(f"  Average spacing: {np.mean(gt_distances):.2f} pixels")
    print(f"  Std dev: {np.std(gt_distances):.2f} pixels")
    print(f"  Min spacing: {np.min(gt_distances):.2f} pixels")
    print(f"  Max spacing: {np.max(gt_distances):.2f} pixels")
    print(f"  Coefficient of variation: {(np.std(gt_distances)/np.mean(gt_distances)*100):.1f}%")

    # Visualize comparison
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "trajectory_comparison.png")
    visualize_comparison(svg_points, gt_points,
                        title="SVG Model Output vs Ground Truth Trajectory (100 points each)",
                        save_path=save_path)