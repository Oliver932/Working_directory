import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import imageio
import glob
from collections import deque
import json
import matplotlib.pyplot as plt
import time


# Import custom modules for the robot simulation
from arm_ik_model import RobotKinematics, Ring
from collision_and_render_management import CollisionAndRenderManager
from ring_projector import RingProjector
from overview_render_manager import OverviewRenderManager
from system_plot_functions import visualize_system

def test_ring_placement(robot, ring, ring_projector, collision_render_manager, random_pose_func):
    """
    Tests the ring placement across a range of difficulty levels, analyzing various metrics
    like collision, success, visibility, and setup success rates. It also visualizes
    the results, including a scatter plot of E1 positions colored by difficulty.
    """
    # --- Testing over difficulty levels ---
    n_runs = 200  # Number of runs per difficulty level
    difficulty_points = np.linspace(0, 1, 31)  # Test 31 points between 0 and 1

    collision_proportions = []
    success_proportions = []
    visible_proportions = []
    calculable_proportions = []
    invalid_setup_proportions = []
    avg_setup_times = []
    
    e1_positions = []
    e1_difficulties = []
    e1_outcomes = []
    e1_visibility = []
    e1_calculability = []

    for difficulty in difficulty_points:
        collisions = 0
        successes = 0
        visible_count = 0
        calculable_count = 0
        invalid_setups = 0
        setup_times = []
        
        print(f"Testing difficulty: {difficulty:.2f}")

        for _ in range(n_runs):
            start_time = time.time()
            # Set a random pose for the ring
            if 'collision_render_manager' in random_pose_func.__code__.co_varnames:
                success, pose, actual_difficulty = random_pose_func(
                    robot, 
                    ring=ring, 
                    collision_render_manager=collision_render_manager,
                    difficulty=difficulty
                )
            else:
                success, pose, actual_difficulty = random_pose_func(
                    robot, 
                    difficulty=difficulty
                )
            
            e1_positions.append([pose['x'], pose['y']])  # Store x,y position for all attempts
            e1_difficulties.append(actual_difficulty)

            if not success:
                invalid_setups += 1
                e1_outcomes.append('Invalid Setup')
                e1_visibility.append(False)
                e1_calculability.append(False)
                continue

            setup_times.append(time.time() - start_time)
            
            # Robot is already at home and ring is positioned correctly by pose function
            # But let's double-check the configuration as a safety net
            robot.go_home()

            collision_render_manager.update_poses(robot, ring)
            ring_projector.update()

            # Check for collision and successful grip (should match pose function validation)
            is_collision = collision_render_manager.check_collision()
            is_success = robot.evaluate_grip(ring)

            ellipse_details = ring_projector.projected_properties
            is_visible = ellipse_details.get('visible', False)
            is_calculable = ellipse_details.get('calculable', False)
            
            # Store visibility and calculability for this position
            e1_visibility.append(is_visible)
            e1_calculability.append(is_calculable)

            if is_collision:
                collisions += 1
                e1_outcomes.append('Collision')
            elif is_success:
                successes += 1
                e1_outcomes.append('Success')
            else:
                e1_outcomes.append('Other')

            if is_visible:
                visible_count += 1
            if is_calculable:
                calculable_count += 1

        valid_runs = n_runs - invalid_setups
        collision_proportions.append(collisions / valid_runs if valid_runs > 0 else 0)
        success_proportions.append(successes / valid_runs if valid_runs > 0 else 0)
        visible_proportions.append(visible_count / valid_runs if valid_runs > 0 else 0)
        calculable_proportions.append(calculable_count / valid_runs if valid_runs > 0 else 0)
        invalid_setup_proportions.append(invalid_setups / n_runs)
        avg_setup_times.append(np.mean(setup_times) if setup_times else 0)

    difficulty_labels = [f"{p:.2f}" for p in difficulty_points]

    # --- Plotting the results ---
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(difficulty_labels, collision_proportions, marker='o', linestyle='-', label='Collisions')
    ax.plot(difficulty_labels, success_proportions, marker='o', linestyle='-', label='Successes')
    ax.plot(difficulty_labels, visible_proportions, marker='o', linestyle='-', label='Visible')
    ax.plot(difficulty_labels, calculable_proportions, marker='o', linestyle='-', label='Calculable')
    ax.plot(difficulty_labels, invalid_setup_proportions, marker='o', linestyle='-', label='Invalid Setups')

    ax.set_ylabel('Proportion')
    ax.set_xlabel('Difficulty Level')
    ax.set_title('Proportions by Difficulty Level')
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.show()

    # --- Plotting setup times ---
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    ax2.bar(difficulty_labels, avg_setup_times, color='skyblue')
    ax2.set_ylabel('Average Setup Time (s)')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_title('Average Setup Time by Difficulty Level')
    plt.xticks(rotation=45, ha="right")
    fig2.tight_layout()
    plt.show()

    # --- Aesthetically Pleasing Scatter plots of E1 positions ---
    plt.style.use('seaborn-v0_8-whitegrid')
    e1_positions = np.array(e1_positions)
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(22, 10))
    fig3.suptitle('Analysis of E1 End-Effector Positions', fontsize=20)

    # Filter data for the first plot to exclude failed setups
    successful_indices = [i for i, o in enumerate(e1_outcomes) if o != 'Invalid Setup']
    successful_positions = e1_positions[successful_indices]
    successful_difficulties = np.array(e1_difficulties)[successful_indices]

    # Determine shared plot limits from all positions
    x_min, x_max = e1_positions[:, 0].min() - 10, e1_positions[:, 0].max() + 10
    y_min, y_max = e1_positions[:, 1].min() - 10, e1_positions[:, 1].max() + 10

    # Plot 1: E1 positions by difficulty (successful setups only)
    scatter = ax3.scatter(successful_positions[:, 0], successful_positions[:, 1], c=successful_difficulties, cmap='coolwarm', alpha=0.7, s=15)
    ax3.set_xlabel('E1 X Position (mm)', fontsize=12)
    ax3.set_ylabel('E1 Y Position (mm)', fontsize=12)
    ax3.set_title('Successful Positions by Difficulty', fontsize=16)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    
    # Plot 2: E1 positions by outcome with visibility/calculability indicators
    # Marker meanings: Circle (o) = Visible+Calculable, Square (s) = Visible only, 
    # Triangle (^) = Calculable only, X = Neither visible nor calculable
    outcome_colors = {
        'Success': '#2ca02c',       # Forest Green
        'Collision': '#d62728',      # Brick Red
        'Invalid Setup': '#1f77b4',  # Steel Blue
        'Other': '#7f7f7f'           # Medium Grey
    }
    
    # Create different marker styles for visibility/calculability
    for outcome, color in outcome_colors.items():
        outcome_indices = [i for i, o in enumerate(e1_outcomes) if o == outcome]
        if not outcome_indices:
            continue
            
        # Separate by visibility and calculability status
        for vis_status, calc_status, marker, alpha, size in [
            (True, True, 'o', 0.8, 20),      # Visible and calculable: solid circle, larger
            (True, False, 's', 0.7, 15),     # Visible but not calculable: square
            (False, True, '^', 0.7, 15),     # Not visible but calculable: triangle
            (False, False, 'x', 0.5, 12)     # Neither visible nor calculable: X, smaller
        ]:
            indices = [i for i in outcome_indices 
                      if e1_visibility[i] == vis_status and e1_calculability[i] == calc_status]
            if indices:
                label_suffix = ""
                if outcome != 'Invalid Setup':  # Don't add visibility info for invalid setups
                    if vis_status and calc_status:
                        label_suffix = " (Vis+Calc)"
                    elif vis_status:
                        label_suffix = " (Vis only)"
                    elif calc_status:
                        label_suffix = " (Calc only)"
                    else:
                        label_suffix = " (Neither)"
                
                ax4.scatter(e1_positions[indices, 0], e1_positions[indices, 1], 
                           c=color, marker=marker, label=f"{outcome}{label_suffix}", 
                           alpha=alpha, s=size, edgecolors='black', linewidth=0.5)

    ax4.set_xlabel('E1 X Position (mm)', fontsize=12)
    ax4.set_ylabel('E1 Y Position (mm)', fontsize=12)
    ax4.set_title('All Positions by Outcome & Visibility/Calculability', fontsize=16)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend(title='Outcome & Ellipse Properties', fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)

    # Add a shared colorbar for the difficulty plot
    fig3.subplots_adjust(right=0.75)  # Make more room for the legend
    cbar_ax = fig3.add_axes([0.77, 0.15, 0.02, 0.7])
    cbar = fig3.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Difficulty', fontsize=12)
    
    plt.show()

    
def set_random_e1_pose_circular_with_repeats(robot, ring, collision_render_manager, difficulty=0.5, max_attempts=20):
    """
    Tries to set a random, valid, and collision-free E1 pose by making repeated attempts.
    Validates robot can reach ring position, then checks collision from home position.
    """
    # Cache limits and home position from the robot's parameters
    rx_lim = robot.params.get('tilt_rx_limit_deg', 0.0)
    rz_lim = robot.params.get('tilt_rz_limit_deg', 0.0)
    x_home = robot.E1_home_x
    y_home = robot.E1_home_y
    max_radius = 350  # Max radius for circular placement

    for _ in range(max_attempts):
        # Generate a random difficulty within the specified range for each parameter
        d_radius = difficulty  # Deterministic radius difficulty
        d_rx = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max
        d_rz = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max

        # Generate a random position in a circular pattern around the home position
        if d_radius > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = d_radius * max_radius
            x = x_home + radius * np.cos(angle)
            y = y_home + radius * np.sin(angle)
        else:
            x, y = x_home, y_home

        # Generate random tilts based on difficulty, with a random sign
        rx = np.random.choice([-1, 1]) * d_rx * rx_lim
        rz = np.random.choice([-1, 1]) * d_rz * rz_lim

        # Create the pose dictionary and test if it's kinematically valid
        e1_position = np.array([x, y, 0], dtype=np.float32)
        pose = {'x': x, 'y': y, 'rx': rx, 'rz': rz}
        
        if robot.update_from_e1_pose(e1_position, rx, rz):
            # If the pose is kinematically valid, create ring at this pose
            robot.create_ring(ring=ring)
            
            # Now return robot to home and check collision/calculability from home
            robot.go_home()
            collision_render_manager.update_poses(robot, ring)
            
            if not collision_render_manager.check_collision():
                # If no collision from home, return robot to ring position and return success
                robot.update_from_e1_pose(e1_position, rx, rz)
                actual_difficulty = _calculate_pose_difficulty_circular(robot, x, y, rx, rz, x_home, y_home, max_radius, rx_lim, rz_lim)
                return True, pose, actual_difficulty
            # If collision from home, continue to next attempt
        # If pose is invalid, robot state is unchanged, continue to next attempt

    # If all attempts fail, reset the robot to home and return failure
    robot.go_home()
    # Return a default home pose dictionary, but with a failure status
    home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
    return False, home_pose, 0.0

def set_random_e1_pose_circular_angle_limit(robot, ring, collision_render_manager, difficulty=0.5, 
                                          angle_min_deg=-20, angle_max_deg=95, max_attempts=20):
    """
    Tries to set a random, valid, and collision-free E1 pose by making repeated attempts.
    Only generates positions within the specified angular sector around the home position.
    
    Args:
        robot: Robot kinematics object
        ring: Ring object
        collision_render_manager: Collision detection manager
        difficulty: Difficulty level (0-1)
        angle_min_deg: Minimum angle in degrees (-180 to 180, where 0 is +X direction from home)
        angle_max_deg: Maximum angle in degrees (-180 to 180, where 0 is +X direction from home)
        max_attempts: Maximum number of attempts to find a valid pose
    
    Returns:
        success (bool): Whether a valid pose was found
        pose (dict): The pose dictionary with x, y, rx, rz
        actual_difficulty (float): The calculated difficulty of the pose
    """
    # Cache limits and home position from the robot's parameters
    rx_lim = robot.params.get('tilt_rx_limit_deg', 0.0)
    rz_lim = robot.params.get('tilt_rz_limit_deg', 0.0)
    x_home = robot.E1_home_x
    y_home = robot.E1_home_y
    max_radius = 250  # Max radius for circular placement
    
    # Convert angle limits to radians
    angle_min = np.deg2rad(angle_min_deg)
    angle_max = np.deg2rad(angle_max_deg)
    
    # Handle angle wrap-around (e.g., if angle_min > angle_max, we're crossing 0°)
    angle_wraps = angle_min > angle_max

    for _ in range(max_attempts):
        # Generate a random difficulty within the specified range for each parameter
        d_radius = difficulty  # Deterministic radius difficulty
        d_rx = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max
        d_rz = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max

        # Generate a random position in a circular pattern around the home position
        if d_radius > 0:
            # Generate angle within the specified limits
            if angle_wraps:
                # Handle wrap-around case (e.g., 150° to -150°)
                if np.random.rand() < 0.5:
                    angle = np.random.uniform(angle_min, np.pi)
                else:
                    angle = np.random.uniform(-np.pi, angle_max)
            else:
                # Normal case
                angle = np.random.uniform(angle_min, angle_max)
            
            radius = d_radius * max_radius
            x = x_home + radius * np.cos(angle)
            y = y_home + radius * np.sin(angle)
        else:
            x, y = x_home, y_home

        # Generate random tilts based on difficulty, with a random sign
        rx = np.random.choice([-1, 1]) * d_rx * rx_lim
        rz = np.random.choice([-1, 1]) * d_rz * rz_lim

        # Create the pose dictionary and test if it's kinematically valid
        e1_position = np.array([x, y, 0], dtype=np.float32)
        pose = {'x': x, 'y': y, 'rx': rx, 'rz': rz}
        
        if robot.update_from_e1_pose(e1_position, rx, rz):
            # If the pose is kinematically valid, create ring at this pose
            robot.create_ring(ring=ring)
            
            # Now return robot to home and check collision/calculability from home
            robot.go_home()
            collision_render_manager.update_poses(robot, ring)
            
            if not collision_render_manager.check_collision():
                # If no collision from home, return robot to ring position and return success
                robot.update_from_e1_pose(e1_position, rx, rz)
                actual_difficulty = _calculate_pose_difficulty_circular(robot, x, y, rx, rz, x_home, y_home, max_radius, rx_lim, rz_lim)
                return True, pose, actual_difficulty
            # If collision from home, continue to next attempt
        # If pose is invalid, robot state is unchanged, continue to next attempt

    # If all attempts fail, reset the robot to home and return failure
    robot.go_home()
    # Return a default home pose dictionary, but with a failure status
    home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
    return False, home_pose, 0.0

def _calculate_pose_difficulty_circular(robot, x, y, rx, rz, x_home, y_home, max_radius, rx_lim, rz_lim):
    """Calculate normalized difficulty of a pose using circular distance and independent tilts."""
    # Calculate circular distance from home position
    distance = np.sqrt((x - x_home)**2 + (y - y_home)**2)
    d_radius_norm = distance / max_radius if max_radius > 0 else 0.0
    
    # Calculate normalized tilt difficulties
    d_rx_norm = abs(rx) / rx_lim if rx_lim > 0 else 0.0
    d_rz_norm = abs(rz) / rz_lim if rz_lim > 0 else 0.0
    
    # Average of the three independent difficulty components
    return (d_radius_norm + d_rx_norm + d_rz_norm) / 3.0

def set_random_pose_box_constraint_fixed_rotation(robot, ring, collision_render_manager, difficulty=0.5, 
                                                box_width=190, box_height=120, max_attempts=100):
    """
    Sets a random E1 pose within a box constraint with fixed rotation difficulty.
    The rotations are set to exactly the difficulty level, not random up to difficulty.
    Returns failure if the exact difficulty rotation results in an invalid pose.
    
    Args:
        robot: Robot kinematics object
        ring: Ring object
        collision_render_manager: Collision detection manager
        difficulty: Difficulty level (0-1) - controls distance from home and exact rotation magnitude
        box_width: Width of the constraint box in mm
        box_height: Height of the constraint box in mm
        max_attempts: Maximum number of attempts to find a valid pose
    
    Returns:
        success (bool): Whether a valid pose was found
        pose (dict): The pose dictionary with x, y, rx, rz
        actual_difficulty (float): The calculated difficulty of the pose
    """
    # Cache limits and home position from the robot's parameters
    rx_lim = robot.params.get('tilt_rx_limit_deg', 0.0)
    rz_lim = robot.params.get('tilt_rz_limit_deg', 0.0)
    x_home = robot.E1_home_x
    y_home = robot.E1_home_y
    
    # Get the actual home rotation values (not necessarily 0,0)
    rx_home = getattr(robot, 'E1_home_rx', 0.0)  # Default to 0 if not available
    rz_home = getattr(robot, 'E1_home_rz', 0.0)  # Default to 0 if not available
    
    # Calculate the maximum possible distance (to the top-right corner of the box)
    max_possible_dist = np.sqrt(box_width**2 + box_height**2)
    
    # Calculate the actual distance from the normalized difficulty value
    actual_distance = difficulty * max_possible_dist
    
    # Calculate available rotation ranges in both directions from home
    rx_pos_range = rx_lim - rx_home  # Available positive rotation from home
    rx_neg_range = rx_home + rx_lim  # Available negative rotation from home  
    rz_pos_range = rz_lim - rz_home  # Available positive rotation from home
    rz_neg_range = rz_home + rz_lim  # Available negative rotation from home
    
    # Generate position at the specified distance from home (bottom-left corner) - done once
    if actual_distance == 0:
        # Stay at home position
        x, y = x_home, y_home
    else:
        # Calculate valid angle range to stay within the box
        min_valid_angle = 0.0
        if actual_distance > box_width:
            # Ensure we don't exceed box width
            min_valid_angle = np.arccos(min(1.0, box_width / actual_distance))
        
        max_valid_angle = np.pi / 2
        if actual_distance > box_height:
            # Ensure we don't exceed box height
            max_valid_angle = np.arcsin(min(1.0, box_height / actual_distance))
        
        # If the valid angle range is impossible, return failure immediately
        if min_valid_angle > max_valid_angle:
            if not np.isclose(min_valid_angle, max_valid_angle):
                robot.go_home()
                home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
                return False, home_pose, 0.0
            max_valid_angle = min_valid_angle
        
        # Generate random angle within valid range
        angle = np.random.uniform(min_valid_angle, max_valid_angle)
        
        # Calculate position
        x = x_home + actual_distance * np.cos(angle)
        y = y_home + actual_distance * np.sin(angle)
    
    # Generate fixed tilts based on exact difficulty from home position - done once
    # Choose random direction and calculate rotation based on available range in that direction
    rx_direction = np.random.choice([-1, 1])
    rz_direction = np.random.choice([-1, 1])
    
    if rx_direction > 0:
        # Positive rotation from home
        base_rx = rx_home + difficulty * rx_pos_range
    else:
        # Negative rotation from home
        base_rx = rx_home - difficulty * rx_neg_range
        
    if rz_direction > 0:
        # Positive rotation from home
        base_rz = rz_home + difficulty * rz_pos_range
    else:
        # Negative rotation from home
        base_rz = rz_home - difficulty * rz_neg_range
    
    # Try with progressively halved rotations and different positions
    current_rotation_scale = 1.0
    
    for attempt in range(max_attempts):
        # Alternate between halving rotations and trying new positions
        if attempt % 2 == 0:
            # Even attempts: use current rotation scale with original position
            x_current, y_current = x, y
        else:
            # Odd attempts: halve rotations and try a new position at the same difficulty
            current_rotation_scale *= 0.5
            
            # Generate a new position at the same distance
            if actual_distance == 0:
                x_current, y_current = x_home, y_home
            else:
                # Generate new random angle within valid range
                angle_new = np.random.uniform(min_valid_angle, max_valid_angle)
                x_current = x_home + actual_distance * np.cos(angle_new)
                y_current = y_home + actual_distance * np.sin(angle_new)
        
        # Apply current rotation scaling to the base rotations
        rx = rx_home + (base_rx - rx_home) * current_rotation_scale
        rz = rz_home + (base_rz - rz_home) * current_rotation_scale
        
        # Create the pose dictionary and test if it's kinematically valid
        e1_position = np.array([x_current, y_current, 0], dtype=np.float32)
        pose = {'x': x_current, 'y': y_current, 'rx': rx, 'rz': rz}
        
        if robot.update_from_e1_pose(e1_position, rx, rz):
            # If the pose is kinematically valid, create ring at this pose
            robot.create_ring(ring=ring)
            
            # Now return robot to home and check collision/calculability from home
            robot.go_home()
            collision_render_manager.update_poses(robot, ring)
            
            if not collision_render_manager.check_collision():
                # If no collision from home, return robot to ring position and return success
                robot.update_from_e1_pose(e1_position, rx, rz)
                actual_difficulty = _calculate_pose_difficulty_box_from_home(robot, x_current, y_current, rx, rz, x_home, y_home, 
                                                                           rx_home, rz_home, max_possible_dist, 
                                                                           rx_pos_range, rx_neg_range, rz_pos_range, rz_neg_range)
                return True, pose, actual_difficulty
            # If collision from home, continue to next attempt
        # If pose is kinematically invalid or collision occurred, continue with next attempt
    
    # If all attempts fail, reset the robot to home and return failure
    robot.go_home()
    # Return a default home pose dictionary, but with a failure status
    home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
    return False, home_pose, 0.0

# current best performance!
def set_random_pose_box_constraint(robot, ring, collision_render_manager, difficulty=0.5, 
                                 box_width=190, box_height=120, max_attempts=100):
    """
    Sets a random E1 pose within a box constraint using difficulty to control distance from home position.
    The home position is treated as the bottom-left corner of the box.
    
    Args:
        robot: Robot kinematics object
        ring: Ring object
        collision_render_manager: Collision detection manager
        difficulty: Difficulty level (0-1) - controls distance from home and rotation limits
        box_width: Width of the constraint box in mm
        box_height: Height of the constraint box in mm
        max_attempts: Maximum number of attempts to find a valid pose
    
    Returns:
        success (bool): Whether a valid pose was found
        pose (dict): The pose dictionary with x, y, rx, rz
        actual_difficulty (float): The calculated difficulty of the pose
    """
    # Cache limits and home position from the robot's parameters
    rx_lim = robot.params.get('tilt_rx_limit_deg', 0.0)
    rz_lim = robot.params.get('tilt_rz_limit_deg', 0.0)
    x_home = robot.E1_home_x
    y_home = robot.E1_home_y
    
    # Calculate the maximum possible distance (to the top-right corner of the box)
    max_possible_dist = np.sqrt(box_width**2 + box_height**2)
    
    # Calculate the actual distance from the normalized difficulty value
    actual_distance = difficulty * max_possible_dist
    
    for _ in range(max_attempts):
        # Generate random tilt difficulties up to the specified difficulty
        d_rx = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max
        d_rz = np.random.uniform(0, difficulty)  # Random tilt difficulty up to max

        
        # Generate position at the specified distance from home (bottom-left corner)
        if actual_distance == 0:
            # Stay at home position
            x, y = x_home, y_home
        else:
            # Calculate valid angle range to stay within the box
            min_valid_angle = 0.0
            if actual_distance > box_width:
                # Ensure we don't exceed box width
                min_valid_angle = np.arccos(min(1.0, box_width / actual_distance))
            
            max_valid_angle = np.pi / 2
            if actual_distance > box_height:
                # Ensure we don't exceed box height
                max_valid_angle = np.arcsin(min(1.0, box_height / actual_distance))
            
            # If the valid angle range is impossible, skip this attempt
            if min_valid_angle > max_valid_angle:
                if not np.isclose(min_valid_angle, max_valid_angle):
                    continue
                max_valid_angle = min_valid_angle
            
            # Generate random angle within valid range
            angle = np.random.uniform(min_valid_angle, max_valid_angle)
            
            # Calculate position
            x = x_home + actual_distance * np.cos(angle)
            y = y_home + actual_distance * np.sin(angle)
        
        # Generate random tilts based on difficulty, with random signs
        rx = np.random.choice([-1, 1]) * d_rx * rx_lim
        rz = np.random.choice([-1, 1]) * d_rz * rz_lim
        
        # Create the pose dictionary and test if it's kinematically valid
        e1_position = np.array([x, y, 0], dtype=np.float32)
        pose = {'x': x, 'y': y, 'rx': rx, 'rz': rz}
        
        if robot.update_from_e1_pose(e1_position, rx, rz):
            # If the pose is kinematically valid, create ring at this pose
            robot.create_ring(ring=ring)
            
            # Now return robot to home and check collision/calculability from home
            robot.go_home()
            collision_render_manager.update_poses(robot, ring)
            
            if not collision_render_manager.check_collision():
                # If no collision from home, return robot to ring position and return success
                robot.update_from_e1_pose(e1_position, rx, rz)
                actual_difficulty = _calculate_pose_difficulty_box(robot, x, y, rx, rz, x_home, y_home, 
                                                                 max_possible_dist, rx_lim, rz_lim)
                return True, pose, actual_difficulty
            # If collision from home, continue to next attempt
        # If pose is invalid, robot state is unchanged, continue to next attempt
    
    # If all attempts fail, reset the robot to home and return failure
    robot.go_home()
    # Return a default home pose dictionary, but with a failure status
    home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
    return False, home_pose, 0.0

def generate_pose_difficulty_from_box_angle_free(robot, ring, collision_render_manager, difficulty=0.5, 
                                               box_width=190, box_height=120, max_attempts=100):
    """
    Generates a pose using difficulty to fix distance from corner, then binary searches rotation space.
    Uses binary search to find valid rotations without assuming zero is ideal.
    
    Args:
        robot: Robot kinematics object
        ring: Ring object
        collision_render_manager: Collision detection manager
        difficulty: Difficulty level (0-1) - controls distance from home corner
        box_width: Width of the constraint box in mm
        box_height: Height of the constraint box in mm
        max_attempts: Maximum number of binary search attempts
    
    Returns:
        success (bool): Whether a valid pose was found
        pose (dict): The pose dictionary with x, y, rx, rz
        actual_difficulty (float): The calculated difficulty of the pose
    """
    # Cache limits and home position from the robot's parameters
    rx_lim = robot.params.get('tilt_rx_limit_deg', 0.0)
    rz_lim = robot.params.get('tilt_rz_limit_deg', 0.0)
    x_home = robot.E1_home_x
    y_home = robot.E1_home_y
    
    # Get the actual home rotation values (not necessarily 0,0)
    rx_home = getattr(robot, 'E1_home_rx', 0.0)  # Default to 0 if not available
    rz_home = getattr(robot, 'E1_home_rz', 0.0)  # Default to 0 if not available
    
    # Calculate the maximum possible distance (to the top-right corner of the box)
    max_possible_dist = np.sqrt(box_width**2 + box_height**2)
    
    # Fix the distance from the normalized difficulty value
    actual_distance = difficulty * max_possible_dist
    
    # Generate position at the specified distance from home (bottom-left corner) - done once
    if actual_distance == 0:
        # Stay at home position
        x, y = x_home, y_home
    else:
        # Calculate valid angle range to stay within the box
        min_valid_angle = 0.0
        if actual_distance > box_width:
            # Ensure we don't exceed box width
            min_valid_angle = np.arccos(min(1.0, box_width / actual_distance))
        
        max_valid_angle = np.pi / 2
        if actual_distance > box_height:
            # Ensure we don't exceed box height
            max_valid_angle = np.arcsin(min(1.0, box_height / actual_distance))
        
        # If the valid angle range is impossible, return failure immediately
        if min_valid_angle > max_valid_angle:
            if not np.isclose(min_valid_angle, max_valid_angle):
                robot.go_home()
                home_pose = {'x': x_home, 'y': y_home, 'rx': rx_home, 'rz': rz_home}
                return False, home_pose, 0.0
            max_valid_angle = min_valid_angle
        
        # Generate random angle within valid range
        angle = np.random.uniform(min_valid_angle, max_valid_angle)
        
        # Calculate position
        x = x_home + actual_distance * np.cos(angle)
        y = y_home + actual_distance * np.sin(angle)
    
    # Define full rotation ranges (not relative to home)
    rx_min, rx_max = -rx_lim, rx_lim
    rz_min, rz_max = -rz_lim, rz_lim
    
    # Binary search for valid rotations
    for attempt in range(max_attempts):
        # Randomly choose rotations within current search ranges
        rx = np.random.uniform(rx_min, rx_max)
        rz = np.random.uniform(rz_min, rz_max)
        
        # Create the pose dictionary and test if it's kinematically valid
        e1_position = np.array([x, y, 0], dtype=np.float32)
        pose = {'x': x, 'y': y, 'rx': rx, 'rz': rz}
        
        # Test kinematics and collision
        valid_kinematics = robot.update_from_e1_pose(e1_position, rx, rz)
        collision_detected = False
        
        if valid_kinematics:
            # If kinematically valid, create ring at this pose
            robot.create_ring(ring=ring)
            
            # Now return robot to home and check collision/calculability from home
            robot.go_home()
            collision_render_manager.update_poses(robot, ring)
            collision_detected = collision_render_manager.check_collision()
            
            if not collision_detected:
                # Success! Return robot to ring position, calculate actual difficulty and return
                robot.update_from_e1_pose(e1_position, rx, rz)
                actual_difficulty = _calculate_pose_difficulty_box(robot, x, y, rx, rz, x_home, y_home, 
                                                                 max_possible_dist, rx_lim, rz_lim)
                return True, pose, actual_difficulty
            # If collision from home, continue binary search
        
        # Failure (either invalid kinematics or collision) - adjust search ranges
        # Binary search strategy: narrow the range by eliminating the half that contains the failed point
        
        # For rx: determine which half of the current range contains the failed point
        rx_mid = (rx_min + rx_max) / 2
        if rx >= rx_mid:
            # Failed point is in upper half, search lower half next
            rx_max = rx_mid
        else:
            # Failed point is in lower half, search upper half next
            rx_min = rx_mid
            
        # For rz: determine which half of the current range contains the failed point
        rz_mid = (rz_min + rz_max) / 2
        if rz >= rz_mid:
            # Failed point is in upper half, search lower half next
            rz_max = rz_mid
        else:
            # Failed point is in lower half, search upper half next
            rz_min = rz_mid
        
        # If ranges become too small, break
        if (rx_max - rx_min) < 0.1 and (rz_max - rz_min) < 0.1:
            break
    
    # If all attempts fail, reset the robot to home and return failure
    robot.go_home()
    # Return a default home pose dictionary, but with a failure status
    home_pose = {'x': x_home, 'y': y_home, 'rx': rx_home, 'rz': rz_home}
    return False, home_pose, 0.0

def _calculate_pose_difficulty_box_from_home(robot, x, y, rx, rz, x_home, y_home, rx_home, rz_home, 
                                           max_distance, rx_pos_range, rx_neg_range, rz_pos_range, rz_neg_range):
    """Calculate normalized difficulty of a pose using box distance and rotation relative to home position."""
    # Calculate distance from home position
    distance = np.sqrt((x - x_home)**2 + (y - y_home)**2)
    d_distance_norm = distance / max_distance if max_distance > 0 else 0.0
    
    # Calculate normalized tilt difficulties relative to home position
    if rx >= rx_home:
        # Positive rotation from home
        d_rx_norm = (rx - rx_home) / rx_pos_range if rx_pos_range > 0 else 0.0
    else:
        # Negative rotation from home
        d_rx_norm = (rx_home - rx) / rx_neg_range if rx_neg_range > 0 else 0.0
        
    if rz >= rz_home:
        # Positive rotation from home
        d_rz_norm = (rz - rz_home) / rz_pos_range if rz_pos_range > 0 else 0.0
    else:
        # Negative rotation from home
        d_rz_norm = (rz_home - rz) / rz_neg_range if rz_neg_range > 0 else 0.0
    
    # Average of the three independent difficulty components
    return (d_distance_norm + d_rx_norm + d_rz_norm) / 3.0

def _calculate_pose_difficulty_box(robot, x, y, rx, rz, x_home, y_home, max_distance, rx_lim, rz_lim):
    """Calculate normalized difficulty of a pose using box distance and independent tilts."""
    # Calculate distance from home position
    distance = np.sqrt((x - x_home)**2 + (y - y_home)**2)
    d_distance_norm = distance / max_distance if max_distance > 0 else 0.0
    
    # Calculate normalized tilt difficulties
    d_rx_norm = abs(rx) / rx_lim if rx_lim > 0 else 0.0
    d_rz_norm = abs(rz) / rz_lim if rz_lim > 0 else 0.0
    
    # Average of the three independent difficulty components
    return (d_distance_norm + d_rx_norm + d_rz_norm) / 3.0

if __name__ == '__main__':
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    env_config = config['environment']
    camera_settings = env_config['camera']
    paths = env_config['paths']

    robot = RobotKinematics(verbosity=0)

    home_success = robot.go_home()
    ring = robot.create_ring()

    ring_projector = RingProjector(robot, ring, vertical_fov_deg=camera_settings["fov"], image_width=camera_settings["width"], image_height=camera_settings["height"], method='custom')
    collision_render_manager = CollisionAndRenderManager(paths["gripper_col"], paths["gripper_col"], paths["ring_render"], paths["ring_col"], vertical_FOV=camera_settings["fov"], render_width=camera_settings["width"], render_height=camera_settings["height"])
    
    test_ring_placement(robot, ring, ring_projector, collision_render_manager, set_random_pose_box_constraint)

    set_random_pose_box_constraint(robot, ring, collision_render_manager, difficulty=1)
    visualize_system(robot, ring=ring)
    set_random_pose_box_constraint(robot, ring, collision_render_manager, difficulty=1)
    visualize_system(robot, ring=ring)



    



