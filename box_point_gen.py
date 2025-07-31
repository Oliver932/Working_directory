import matplotlib.pyplot as plt
import numpy as np

def generate_points_from_bottom_left(box_width, box_height, normalized_distance, num_points):
    """
    Generates points within a 2D bounding box at a fixed distance from the bottom-left corner.

    Args:
        box_width (float): The width of the bounding box.
        box_height (float): The height of the bounding box.
        normalized_distance (float): A value from 0 to 1 representing the distance. 
                                     1.0 is the max distance to the top-right corner.
        num_points (int): The number of points to generate for the arc.

    Returns:
        tuple: A tuple containing:
               - list: A list of (x, y) points.
               - float: The actual distance used for calculations.
    """
    if not (0 <= normalized_distance <= 1):
        raise ValueError("normalized_distance must be between 0 and 1.")

    # Calculate the maximum possible distance (to the top-right corner)
    max_possible_dist = np.sqrt(box_width**2 + box_height**2)
    
    # Calculate the actual distance from the normalized value
    actual_distance = normalized_distance * max_possible_dist
    
    points = []
    
    # --- Bottom-Left Corner Logic ---
    x_c, y_c = (0, 0)
    
    # If distance is zero, there's only one point at the corner
    if actual_distance == 0:
        return ([(x_c, y_c)], 0)

    # --- Corrected Angle Clipping Logic ---
    
    # Calculate the minimum valid angle to stay within the box width
    min_valid_angle = 0.0
    if actual_distance > box_width:
        # The angle must be large enough so that cos(angle) is small enough
        # cos(angle) <= box_width / actual_distance  =>  angle >= arccos(...)
        # Clamp argument to handle potential floating point inaccuracies
        min_valid_angle = np.arccos(min(1.0, box_width / actual_distance))

    # Calculate the maximum valid angle to stay within the box height
    max_valid_angle = np.pi / 2
    if actual_distance > box_height:
        # The angle must be small enough so that sin(angle) is small enough
        # sin(angle) <= box_height / actual_distance  =>  angle <= arcsin(...)
        max_valid_angle = np.arcsin(min(1.0, box_height / actual_distance))
    
    # If the valid angle range is impossible, return no points.
    # This can happen with floating point errors if min > max slightly.
    if min_valid_angle > max_valid_angle:
        # Check if they are very close, which means a single point
        if np.isclose(min_valid_angle, max_valid_angle):
             max_valid_angle = min_valid_angle
        else:
             return [], actual_distance
    
    # Generate points along the valid arc
    # If min == max, linspace will correctly generate a single point.
    for angle in np.linspace(min_valid_angle, max_valid_angle, num_points):
        x = x_c + actual_distance * np.cos(angle)
        y = y_c + actual_distance * np.sin(angle)
        points.append((x, y))
        
    return points, actual_distance

def plot_points(ax, box_width, box_height, normalized_distance, actual_distance, points):
    """Helper function to plot the results."""
    # Plot the bounding box
    rect = plt.Rectangle((0, 0), box_width, box_height, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    
    if points:
        x_vals, y_vals = zip(*points)
        ax.scatter(x_vals, y_vals, c='r', label='Generated Points', s=20)

    title = (f'Normalized Distance: {normalized_distance:.2f}\n'
             f'Actual Distance: {actual_distance:.2f}')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

# --- Main Test Execution ---
if __name__ == "__main__":
    # Define bounding box dimensions
    BOX_WIDTH = 10.0
    BOX_HEIGHT = 5.0
    NUM_POINTS = 50

    # Create a figure to display the tests
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('Generating Points from Bottom-Left Corner with Normalized Distance', fontsize=16)

    # Test Case 1: Small normalized distance
    norm_dist1 = 0.3
    points1, actual_dist1 = generate_points_from_bottom_left(BOX_WIDTH, BOX_HEIGHT, norm_dist1, NUM_POINTS)
    plot_points(axs[0], BOX_WIDTH, BOX_HEIGHT, norm_dist1, actual_dist1, points1)

    # Test Case 2: Medium normalized distance (actual distance > height)
    norm_dist2 = 0.7
    points2, actual_dist2 = generate_points_from_bottom_left(BOX_WIDTH, BOX_HEIGHT, norm_dist2, NUM_POINTS)
    plot_points(axs[1], BOX_WIDTH, BOX_HEIGHT, norm_dist2, actual_dist2, points2)

    # Test Case 3: Max normalized distance
    norm_dist3 = 0.95
    points3, actual_dist3 = generate_points_from_bottom_left(BOX_WIDTH, BOX_HEIGHT, norm_dist3, NUM_POINTS)
    plot_points(axs[2], BOX_WIDTH, BOX_HEIGHT, norm_dist3, actual_dist3, points3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
