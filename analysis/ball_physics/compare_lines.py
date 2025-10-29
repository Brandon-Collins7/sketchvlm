import numpy as np
from svgpathtools import parse_path
import xml.etree.ElementTree as ET
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

# Usage
if __name__ == "__main__":
    import numpy as np
    
    svg_file = "/Users/log/Github/sketchvlm/results/mix_eval/gpt5_low_ball_paths/item_00000.svg"
    
    # Extract exactly 100 points uniformly along entire trajectory
    points = extract_points_from_svg(svg_file, n_points=100)
    print(f"Extracted {len(points)} points")
    print(f"First point: {points[0]}")
    print(f"Last point: {points[-1]}")
    
    # Verify uniform spacing
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    print(f"\nSpacing Statistics:")
    print(f"  Average spacing: {np.mean(distances):.2f} pixels")
    print(f"  Std dev: {np.std(distances):.2f} pixels")
    print(f"  Min spacing: {np.min(distances):.2f} pixels")
    print(f"  Max spacing: {np.max(distances):.2f} pixels")
    print(f"  Coefficient of variation: {(np.std(distances)/np.mean(distances)*100):.1f}%")
    
    # Visualize
    visualize_trajectory(points, title="Ball Trajectory", 
                        save_path="trajectory_visualization.png")