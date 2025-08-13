import numpy as np
import matplotlib.pyplot as plt


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
from scipy.spatial.transform import Rotation as R


# Import custom modules for the robot simulation
from arm_ik_model import RobotKinematics, Ring
from collision_and_render_management import CollisionAndRenderManager
from ring_projector import RingProjector
from overview_render_manager import OverviewRenderManager
from system_plot_functions import visualize_system

from ring_placement import set_random_pose_box_limit



def evaluate_movement_at_point(robot, config, n_samples=100, independent=False):
    """
    Evaluate movement characteristics at the current robot position by sampling
    small movements and recording various metrics.
    
    Args:
        robot (RobotKinematics): Robot instance at the position to evaluate
        config (dict): Configuration containing multipliers
        n_samples (int): Number of movement samples to take
        
    Returns:
        dict: Dictionary containing mean movement metrics:
            - mean_delta_E1: Mean E1 position change [x, y, z]
            - mean_delta_E1_quaternion: Mean E1 quaternion change
            - mean_delta_extensions: Mean actuator extension changes
            - mean_G1_approach: Mean G1 movement along approach vector
            - mean_G1_radial: Mean G1 movement along radial vector  
            - mean_G1_tangential: Mean G1 movement along tangential vector
            - mean_gripper_rot_about_approach: Mean gripper rotation about original approach vector
            - mean_gripper_rot_about_radial: Mean gripper rotation about original radial vector
            - mean_gripper_rot_about_tangential: Mean gripper rotation about original tangential vector
            - mean_delta_rx: Mean rx rotation change
            - mean_delta_rz: Mean rz rotation change
            - mean_delta_E1_x: Mean E1 x-coordinate change
            - mean_delta_E1_y: Mean E1 y-coordinate change
            - successful_moves: Number of successful moves out of total samples
            - mean_attempted_dx: Mean attempted dx
            - mean_attempted_dy: Mean attempted dy
            - mean_attempted_drx: Mean attempted drx
            - mean_attempted_drz: Mean attempted drz
    """
    
    # Get multipliers from config
    multipliers = config['environment']['multipliers']
    linear_mult = multipliers['linear']
    rx_mult = multipliers['rx'] 
    rz_mult = multipliers['rz']
    
    # Store original robot state
    if not robot.last_solve_successful:
        raise ValueError("Robot must be in a valid solved state before evaluation")
    
    original_E1 = robot.E1.copy()
    original_G1 = robot.G1.copy()
    original_approach = robot.approach_vec.copy()
    original_radial = robot.radial_vec.copy()
    original_tangential = robot.tangential_vec.copy()
    original_extensions = robot.extensions.copy()
    original_E1_quaternion = robot.E1_quaternion.copy()
    original_rx = robot.rx
    original_rz = robot.rz
    
    # Lists to store deltas for each successful movement
    # Removed delta_E1_quaternion_list logging
    delta_extensions_list = []
    G1_approach_list = []
    G1_radial_list = []
    G1_tangential_list = []
    gripper_rot_about_approach_list = []
    gripper_rot_about_radial_list = []
    gripper_rot_about_tangential_list = []
    delta_rx_list = []
    delta_rz_list = []
    delta_E1_x_list = []
    delta_E1_y_list = []
    
    successful_moves = 0
    
    axes = ['dx', 'dy', 'drx', 'drz']
    for i in range(n_samples):
        if independent:
            # Only move in one axis at a time
            axis = np.random.choice(axes)
            dx, dy, drx, drz = 0.0, 0.0, 0.0, 0.0
            if axis == 'dx':
                dx = np.random.uniform(-linear_mult, linear_mult)
            elif axis == 'dy':
                dy = np.random.uniform(-linear_mult, linear_mult)
            elif axis == 'drx':
                drx = np.random.uniform(-rx_mult, rx_mult)
            elif axis == 'drz':
                drz = np.random.uniform(-rz_mult, rz_mult)
        else:
            # Move in all axes simultaneously
            dx = np.random.uniform(-linear_mult, linear_mult)
            dy = np.random.uniform(-linear_mult, linear_mult)
            drx = np.random.uniform(-rx_mult, rx_mult)
            drz = np.random.uniform(-rz_mult, rz_mult)

        # Try the movement
        success, reason, actuator_delta = robot.move_E1(dx=dx, dy=dy, drx=drx, drz=drz)
        
        if success:
            successful_moves += 1
            
            # Record E1 position change
            delta_E1 = robot.E1 - original_E1
            delta_extensions_list.append(actuator_delta)
            
            # Calculate G1 movement components along original orientation vectors
            delta_G1 = robot.G1 - original_G1
            G1_approach_movement = np.dot(delta_G1, original_approach)
            G1_radial_movement = np.dot(delta_G1, original_radial)
            G1_tangential_movement = np.dot(delta_G1, original_tangential)
            
            G1_approach_list.append(G1_approach_movement)
            G1_radial_list.append(G1_radial_movement)
            G1_tangential_list.append(G1_tangential_movement)
            
            # Calculate gripper orientation changes about original axes
            # Use scipy Rotation to calculate precise rotations like in find_gripper_tolerances.py
            current_approach = robot.approach_vec
            current_radial = robot.radial_vec
            current_tangential = robot.tangential_vec
            
            # Create rotation matrices from original and current orientation frames
            original_frame = np.column_stack([original_approach, original_radial, original_tangential])
            current_frame = np.column_stack([current_approach, current_radial, current_tangential])
            
            # Calculate the rotation from original to current frame
            # R_current = R_delta * R_original
            # Therefore: R_delta = R_current * R_original^T
            
            try:
                R_original = R.from_matrix(original_frame)
                R_current = R.from_matrix(current_frame)
                R_delta = R_current * R_original.inv()
                
                # Get rotation vector (axis-angle representation)
                rotvec = R_delta.as_rotvec()
                
                # Project rotation vector onto each original axis to get rotation components
                rot_about_approach = np.dot(rotvec, original_approach)
                rot_about_radial = np.dot(rotvec, original_radial)
                rot_about_tangential = np.dot(rotvec, original_tangential)
                
            except ValueError:
                # Fallback to zero if rotation calculation fails (degenerate case)
                rot_about_approach = 0.0
                rot_about_radial = 0.0
                rot_about_tangential = 0.0
            
            gripper_rot_about_approach_list.append(rot_about_approach)
            gripper_rot_about_radial_list.append(rot_about_radial)
            gripper_rot_about_tangential_list.append(rot_about_tangential)
            
            # Record rotation changes
            delta_rx = robot.rx - original_rx
            delta_rz = robot.rz - original_rz
            
            delta_rx_list.append(delta_rx)
            delta_rz_list.append(delta_rz)
            
            # Record E1 x and y changes specifically
            delta_E1_x_list.append(delta_E1[0])
            delta_E1_y_list.append(delta_E1[1])
        
        # Reset robot to original state for next sample
        robot.update_from_e1_pose(original_E1, original_rx, original_rz)
    
    # Return raw lists of numpy arrays and successful_moves
    results = {
        'delta_extensions_list': np.array(delta_extensions_list),
        'G1_approach_list': np.array(G1_approach_list),
        'G1_radial_list': np.array(G1_radial_list),
        'G1_tangential_list': np.array(G1_tangential_list),
        'gripper_rot_about_approach_list': np.array(gripper_rot_about_approach_list),
        'gripper_rot_about_radial_list': np.array(gripper_rot_about_radial_list),
        'gripper_rot_about_tangential_list': np.array(gripper_rot_about_tangential_list),
        'delta_rx_list': np.array(delta_rx_list),
        'delta_rz_list': np.array(delta_rz_list),
        'delta_E1_x_list': np.array(delta_E1_x_list),
        'delta_E1_y_list': np.array(delta_E1_y_list),
        'successful_moves': successful_moves,
    }
    return results

def evaluate_gripper_tolerances(robot, ring, collision_render_manager, max_range=100, angle_range_deg=40, n_search=300):
    """
    Evaluate gripper tolerances by sampling translation and rotation along/about each axis
    and checking for collisions using the collision_render_manager.
    
    Args:
        robot (RobotKinematics): Robot instance at the position to evaluate
        ring (Ring): Ring object at current position
        collision_render_manager (CollisionAndRenderManager): Collision detection manager
        max_range (float): Maximum translation range to test in mm
        angle_range_deg (float): Maximum rotation range to test in degrees
        n_search (int): Number of binary search iterations
        
    Returns:
        dict: Dictionary containing tolerance ranges for each axis:
            - trans_approach: (neg_limit, pos_limit) in mm
            - trans_radial: (neg_limit, pos_limit) in mm
            - trans_tangential: (neg_limit, pos_limit) in mm
            - rot_approach: (neg_limit, pos_limit) in degrees
            - rot_radial: (neg_limit, pos_limit) in degrees
            - rot_tangential: (neg_limit, pos_limit) in degrees
    """
    
    def sample_tolerance(axis_vec, mode):
        """
        Sample tolerance along/about a given axis using binary search.
        mode: 'trans' or 'rot'
        axis_vec: axis in world coordinates (robot's orientation vectors)
        """
        # Store the original pose to always start from it
        orig_pos = ring.position.copy()
        orig_approach = ring.approach_vec.copy()
        orig_tangential = ring.tangential_vec.copy()
        orig_radial = ring.radial_vec.copy()
        
        # Use robot's G1 point as pivot for rotations
        pivot = robot.G1.copy()

        def test_value(v):
            if mode == 'trans':
                new_pos = orig_pos - v * axis_vec
                test_ring = Ring(ring.outer_radius, ring.middle_radius, ring.inner_radius, 
                               ring.outer_thickness, ring.inner_thickness,
                               new_pos, orig_approach, orig_tangential, orig_radial)
            else:  # mode == 'rot'
                # Rotate about robot's G1 point
                rot = R.from_rotvec(-np.deg2rad(v) * axis_vec)
                
                # Move ring so pivot is at origin
                rel_pos = orig_pos - pivot
                
                # Rotate orientation vectors
                new_approach = rot.apply(orig_approach)
                new_tangential = rot.apply(orig_tangential)
                new_radial = rot.apply(orig_radial)
                
                # Rotate position around pivot
                new_rel_pos = rot.apply(rel_pos)
                new_pos = new_rel_pos + pivot
                
                test_ring = Ring(ring.outer_radius, ring.middle_radius, ring.inner_radius,
                               ring.outer_thickness, ring.inner_thickness,
                               new_pos, new_approach, new_tangential, new_radial)
            
            # Check if robot can grip the ring at this pose
            can_grip = robot.evaluate_grip(test_ring)
            if not can_grip:
                return False
                
            # Check for collisions using collision_render_manager
            collision_render_manager.update_poses(robot, test_ring)
            has_collision = collision_render_manager.check_collision()

            return not has_collision  # Return True if no collision

        # Binary search for positive limit
        low, high = 0.0, max_range if mode == 'trans' else angle_range_deg
        pos_limit = 0.0
        for _ in range(n_search):
            mid = (low + high) / 2
            if test_value(mid):
                pos_limit = mid
                low = mid
            else:
                high = mid
                
        # Binary search for negative limit
        low, high = -max_range if mode == 'trans' else -angle_range_deg, 0.0
        neg_limit = 0.0
        for _ in range(n_search):
            mid = (low + high) / 2
            if test_value(mid):
                neg_limit = mid
                high = mid
            else:
                low = mid
                
        return neg_limit, pos_limit

    # Define the axes to test (use robot's orientation vectors)
    axes = [
        (robot.approach_vec, 'approach'),
        (robot.radial_vec, 'radial'),
        (robot.tangential_vec, 'tangential')
    ]
    
    results = {}
    
    # Test translations along each axis
    for axis, name in axes:
        neg, pos = sample_tolerance(axis, 'trans')
        results[f'trans_{name}'] = (neg, pos)
    
    # Test rotations about each axis
    for axis, name in axes:
        neg, pos = sample_tolerance(axis, 'rot')
        results[f'rot_{name}'] = (neg, pos)
    
    # Prepare readable dict for saving and plotting
    readable = {}
    for key, (neg, pos) in results.items():
        range_val = pos - neg
        readable[key] = {
            "neg_limit": float(neg),
            "pos_limit": float(pos),
            "range": float(range_val)
        }
    # Bar chart: show lower to upper limit for each axis
    labels = []
    lowers = []
    uppers = []
    for key in results:
        labels.append(key)
        lowers.append(results[key][0])
        uppers.append(results[key][1])
    plt.figure(figsize=(10,5))
    # Plot as horizontal bars from lower to upper
    for i, (low, up) in enumerate(zip(lowers, uppers)):
        plt.barh(i, up-low, left=low, color="#1f77b4" if "trans" in labels[i] else "#ff7f0e")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Tolerance (mm or deg)")
    plt.title("Gripper Tolerances by Axis (Lower to Upper Limit)")
    plt.tight_layout()
    plt.show()
    # Save JSON
    with open("gripper_tolerances.json", "w") as f:
        json.dump(readable, f, indent=2)
    return results

def evaluate_movement_scales(robot, ring, collision_render_manager, config, n_positions=100, n_samples_per_pos=50):
    """
    Samples random robot positions across difficulty using set_random_pose_box_limit,
    evaluates movement at each using evaluate_movement_at_point, concatenates all movement lists,
    and plots bar charts showing the range of each movement metric.

    Args:
        robot (RobotKinematics): Robot instance
        ring (Ring): Ring instance
        collision_render_manager (CollisionAndRenderManager): Collision manager
        config (dict): Configuration dictionary
        n_positions (int): Number of random positions to sample
        n_samples_per_pos (int): Number of movement samples per position

    Returns:
        dict: Concatenated movement results for all positions
    """
    # Prepare lists to concatenate results
    all_results = {
        'delta_extensions_list': [],
        'G1_approach_list': [],
        'G1_radial_list': [],
        'G1_tangential_list': [],
        'gripper_rot_about_approach_list': [],
        'gripper_rot_about_radial_list': [],
        'gripper_rot_about_tangential_list': [],
        'delta_rx_list': [],
        'delta_rz_list': [],
        'delta_E1_x_list': [],
        'delta_E1_y_list': [],
    }
    difficulties = np.linspace(0, 1, n_positions)
    successful_positions = 0

    for difficulty in difficulties:
        # Try to set a random pose
        result = set_random_pose_box_limit(robot, ring, collision_render_manager, difficulty=difficulty)
        # Accept both (success, pose, actual_difficulty) or just success
        if isinstance(result, tuple):
            success = result[0]
        else:
            success = result
        if not success:
            continue
        successful_positions += 1
        # Evaluate movement at this pose
        movement_results = evaluate_movement_at_point(robot, config, n_samples=n_samples_per_pos)
        # Concatenate lists
        for key in all_results:
            all_results[key].append(movement_results[key])

    # Concatenate all arrays
    for key in all_results:
        if all_results[key]:
            all_results[key] = np.concatenate(all_results[key], axis=0)
        else:
            all_results[key] = np.array([])

    # Unnormalize delta_extensions_list using actuator length of 400mm
    # NOTE: Actuator length is 400mm, so delta_extensions_list is multiplied by 400 to convert to mm
    if all_results['delta_extensions_list'].size > 0:
        all_results['delta_extensions_list'] = all_results['delta_extensions_list'] * 400


    # Split metrics into angles and distances, convert angles to degrees
    distance_metrics = [
        ('delta_extensions_list', 'ΔExtensions[0] (mm)', 0),
        ('delta_extensions_list', 'ΔExtensions[1] (mm)', 1),
        ('delta_extensions_list', 'ΔExtensions[2] (mm)', 2),
        ('delta_extensions_list', 'ΔExtensions[3] (mm)', 3),
        ('G1_approach_list', 'G1 Approach (mm)', None),
        ('G1_radial_list', 'G1 Radial (mm)', None),
        ('G1_tangential_list', 'G1 Tangential (mm)', None),
        ('delta_E1_x_list', 'ΔE1.x (mm)', None),
        ('delta_E1_y_list', 'ΔE1.y (mm)', None),
    ]
    angle_metrics = [
        ('gripper_rot_about_approach_list', 'Gripper Rot About Approach (deg)', None),
        ('gripper_rot_about_radial_list', 'Gripper Rot About Radial (deg)', None),
        ('gripper_rot_about_tangential_list', 'Gripper Rot About Tangential (deg)', None),
        ('delta_rx_list', 'Δrx (deg)', None),
        ('delta_rz_list', 'Δrz (deg)', None),
    ]

    # Prepare box plot data
    dist_data = []
    dist_labels = []
    for key, label, idx in distance_metrics:
        arr = all_results[key]
        if arr.size == 0:
            continue
        if idx is None:
            dist_data.append(arr)
            dist_labels.append(label)
        else:
            dist_data.append(arr[:, idx])
            dist_labels.append(label)

    ang_data = []
    ang_labels = []
    for key, label, idx in angle_metrics:
        arr = all_results[key]
        if arr.size == 0:
            continue
        # Convert radians to degrees for rotation metrics
        if 'Gripper Rot' in label:
            arr = np.degrees(arr)
        ang_data.append(arr)
        ang_labels.append(label)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, max(len(dist_labels), len(ang_labels))*0.7), 10))
    # Distance metrics (hide outliers)
    axes[0].boxplot(dist_data, vert=True, patch_artist=True, showfliers=False)
    axes[0].set_xticks(range(1, len(dist_labels)+1))
    axes[0].set_xticklabels(dist_labels, rotation=45, ha='right')
    axes[0].set_ylabel("Distance (mm)")
    axes[0].set_title("Movement Distance Metrics Across All Positions (Box Plots)")
    # Angle metrics (hide outliers)
    axes[1].boxplot(ang_data, vert=True, patch_artist=True, showfliers=False)
    axes[1].set_xticks(range(1, len(ang_labels)+1))
    axes[1].set_xticklabels(ang_labels, rotation=45, ha='right')
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Movement Angle Metrics Across All Positions (Box Plots)")

    plt.tight_layout()
    plt.show()

    print(f"Evaluated movement at {successful_positions}/{n_positions} random positions.")
    return all_results

def evaluate_movement_relationships(robot, ring, collision_render_manager, config, n_positions=100, n_samples_per_pos=50):
    """
    Evaluates relationships between commanded changes and resulting movements by sampling
    independent axis movements. Creates scatter plots showing relationships between
    commanded changes and resulting G1 translations, G1 rotations, and extension changes.

    Args:
        robot (RobotKinematics): Robot instance
        ring (Ring): Ring instance
        collision_render_manager (CollisionAndRenderManager): Collision manager
        config (dict): Configuration dictionary
        n_positions (int): Number of random positions to sample
        n_samples_per_pos (int): Number of movement samples per position

    Returns:
        dict: Movement results organized by commanded axis
    """
    # Prepare results organized by commanded axis
    axis_results = {
        'dx': {'commanded': [], 'G1_translations': [], 'G1_rotations': [], 'extensions': []},
        'dy': {'commanded': [], 'G1_translations': [], 'G1_rotations': [], 'extensions': []},
        'drx': {'commanded': [], 'G1_translations': [], 'G1_rotations': [], 'extensions': []},
        'drz': {'commanded': [], 'G1_translations': [], 'G1_rotations': [], 'extensions': []}
    }
    
    difficulties = np.linspace(0, 1, n_positions)
    successful_positions = 0

    for difficulty in difficulties:
        # Try to set a random pose
        result = set_random_pose_box_limit(robot, ring, collision_render_manager, difficulty=difficulty)
        if isinstance(result, tuple):
            success = result[0]
        else:
            success = result
        if not success:
            continue
        successful_positions += 1
        
        # Evaluate movement at this pose using independent axis movements
        movement_results = evaluate_movement_at_point(robot, config, n_samples=n_samples_per_pos, independent=True)
        
        # For each successful movement, determine which axis was commanded by looking at 
        # which of delta_E1_x, delta_E1_y, delta_rx, delta_rz is non-zero
        for i in range(len(movement_results['G1_approach_list'])):
            # Get the actual changes for this sample
            delta_x = movement_results['delta_E1_x_list'][i]
            delta_y = movement_results['delta_E1_y_list'][i]
            delta_rx = movement_results['delta_rx_list'][i]
            delta_rz = movement_results['delta_rz_list'][i]
            
            # Determine which axis was moved by finding the largest absolute change
            # (accounting for different scales)
            abs_changes = {
                'dx': abs(delta_x),
                'dy': abs(delta_y), 
                'drx': abs(delta_rx),
                'drz': abs(delta_rz)
            }
            
            # Find the axis with the largest change
            commanded_axis = max(abs_changes, key=abs_changes.get)
            
            # Get the commanded value (which is the actual change that occurred)
            if commanded_axis == 'dx':
                commanded_value = delta_x
            elif commanded_axis == 'dy':
                commanded_value = delta_y
            elif commanded_axis == 'drx':
                commanded_value = delta_rx
            elif commanded_axis == 'drz':
                commanded_value = delta_rz
            
            # Store results for this axis
            axis_results[commanded_axis]['commanded'].append(commanded_value)
            axis_results[commanded_axis]['G1_translations'].append([
                movement_results['G1_approach_list'][i],
                movement_results['G1_radial_list'][i], 
                movement_results['G1_tangential_list'][i]
            ])
            axis_results[commanded_axis]['G1_rotations'].append([
                movement_results['gripper_rot_about_approach_list'][i],
                movement_results['gripper_rot_about_radial_list'][i],
                movement_results['gripper_rot_about_tangential_list'][i]
            ])
            # Extensions are already unnormalized by 400mm in evaluate_movement_at_point
            axis_results[commanded_axis]['extensions'].append(movement_results['delta_extensions_list'][i] * 400)

    # Convert lists to numpy arrays
    axes = ['dx', 'dy', 'drx', 'drz']
    for axis in axes:
        for key in ['commanded', 'G1_translations', 'G1_rotations', 'extensions']:
            if axis_results[axis][key]:
                axis_results[axis][key] = np.array(axis_results[axis][key])
            else:
                axis_results[axis][key] = np.array([])

    # --- Outlier filtering helper ---
    def filter_outliers_percentile(x, y, lower=1, upper=99):
        """
        Removes outliers from x and y based on percentiles.
        Keeps only points where both x and y are within [lower, upper] percentiles.
        Returns filtered x, y arrays.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_low, x_high = np.percentile(x, [lower, upper])
        y_low, y_high = np.percentile(y, [lower, upper])
        mask = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
        return x[mask], y[mask]

    from sklearn.linear_model import LinearRegression
    
    # Store regression models for reuse in heatmap
    regression_models = {}
    
    # Create a single large grid plot: 3 rows (G1 translations, G1 rotations, extensions) x 4 columns (dx, dy, drx, drz)
    fig, axes_plot = plt.subplots(3, 4, figsize=(20, 15))
    
    # Helper function to add regression line and store model, returns handles
    def add_regression_line_and_store(ax, x, y, color, axis_name, response_name):
        handles = []
        labels = []
        # Robust outlier filtering before regression
        x_filt, y_filt = filter_outliers_percentile(x, y, lower=1, upper=99)
        if len(x_filt) == 0:
            return handles, labels
        
        # Always add scatter plot first
        scatter = ax.scatter(x_filt, y_filt, alpha=0.05, color=color, s=15, zorder=2)
        handles.append(scatter)
        labels.append(response_name)
        
        if len(x_filt) > 1:
            X = x_filt.reshape(-1, 1)
            reg = LinearRegression().fit(X, y_filt)
            correlation = np.corrcoef(x_filt, y_filt)[0, 1] if len(x_filt) > 1 else 0.0
            
            # Store model for later use
            key = f"{response_name}_{axis_name}"
            regression_models[key] = {
                'model': reg,
                'correlation': correlation,
                'gradient': reg.coef_[0]
            }
            
            # Only plot regression line if correlation is above threshold (0.3)
            if abs(correlation) > 0.3:
                # Create line points
                x_line = np.linspace(x_filt.min(), x_filt.max(), 100)
                y_line = reg.predict(x_line.reshape(-1, 1))
                line, = ax.plot(x_line, y_line, color=color, linestyle='-', alpha=0.5, linewidth=3, zorder=1)
                handles.append(line)
                labels.append(response_name + " (reg)")
        return handles, labels

    for axis_idx, axis in enumerate(axes):
        if len(axis_results[axis]['commanded']) == 0:
            # If no data for this axis, skip but keep subplot structure
            for row in range(3):
                axes_plot[row, axis_idx].text(0.5, 0.5, f'No data for {axis}', 
                                            ha='center', va='center', transform=axes_plot[row, axis_idx].transAxes)
                axes_plot[row, axis_idx].set_title(f'{axis}')
            continue

        commanded = axis_results[axis]['commanded']

        # Row 0: G1 translations vs commanded change
        if axis_results[axis]['G1_translations'].size > 0:
            G1_trans = axis_results[axis]['G1_translations']
            colors = ['C0', 'C1', 'C2']  # Default matplotlib colors
            response_names = ['G1_Approach', 'G1_Radial', 'G1_Tangential']

            handles = []
            labels = []
            for i, (color, name) in enumerate(zip(colors, response_names)):
                h, l = add_regression_line_and_store(axes_plot[0, axis_idx], commanded, G1_trans[:, i], color, axis, name)
                handles.extend(h)
                labels.extend(l)

            axes_plot[0, axis_idx].set_ylabel('G1 Translation (mm)')
            axes_plot[0, axis_idx].set_title(f'G1 Translations vs {axis}')
            if handles:
                axes_plot[0, axis_idx].legend(handles, labels, fontsize=8)
            axes_plot[0, axis_idx].grid(True, alpha=0.3)

        # Row 1: G1 rotations vs commanded change
        if axis_results[axis]['G1_rotations'].size > 0:
            G1_rot = np.degrees(axis_results[axis]['G1_rotations'])
            colors = ['C3', 'C4', 'C5']
            response_names = ['G1_Rot_Approach', 'G1_Rot_Radial', 'G1_Rot_Tangential']

            handles = []
            labels = []
            for i, (color, name) in enumerate(zip(colors, response_names)):
                h, l = add_regression_line_and_store(axes_plot[1, axis_idx], commanded, G1_rot[:, i], color, axis, name)
                handles.extend(h)
                labels.extend(l)

            axes_plot[1, axis_idx].set_ylabel('G1 Rotation (deg)')
            axes_plot[1, axis_idx].set_title(f'G1 Rotations vs {axis}')
            if handles:
                axes_plot[1, axis_idx].legend(handles, labels, fontsize=8)
            axes_plot[1, axis_idx].grid(True, alpha=0.3)

        # Row 2: Extension changes vs commanded change
        if axis_results[axis]['extensions'].size > 0:
            extensions = axis_results[axis]['extensions']
            colors = ['C6', 'C7', 'C8', 'C9']
            response_names = ['Ext_0', 'Ext_1', 'Ext_2', 'Ext_3']

            handles = []
            labels = []
            for i, (color, name) in enumerate(zip(colors, response_names)):
                h, l = add_regression_line_and_store(axes_plot[2, axis_idx], commanded, extensions[:, i], color, axis, name)
                handles.extend(h)
                labels.extend(l)

            axes_plot[2, axis_idx].set_ylabel('Extension Change (mm)')
            axes_plot[2, axis_idx].set_title(f'Extension Changes vs {axis}')
            if handles:
                axes_plot[2, axis_idx].legend(handles, labels, fontsize=8)
            axes_plot[2, axis_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create correlation and regression analysis plots using stored models
    # Create correlation and regression analysis plots using stored models
    if regression_models:
        # Extract unique axes and responses from stored models
        unique_axes = sorted(list(set(key.split('_')[-1] for key in regression_models.keys())))
        unique_responses = sorted(list(set('_'.join(key.split('_')[:-1]) for key in regression_models.keys())))

        correlations = np.zeros((len(unique_responses), len(unique_axes)))
        gradients = np.zeros((len(unique_responses), len(unique_axes)))

        # Fill matrices using stored regression models
        for i, response_type in enumerate(unique_responses):
            for j, axis in enumerate(unique_axes):
                key = f"{response_type}_{axis}"
                if key in regression_models:
                    correlations[i, j] = regression_models[key]['correlation']
                    gradients[i, j] = regression_models[key]['gradient']
                else:
                    correlations[i, j] = np.nan
                    gradients[i, j] = np.nan

        # Save gradients to JSON
        gradients_dict = {}
        for i, response_type in enumerate(unique_responses):
            gradients_dict[response_type] = {}
            for j, axis in enumerate(unique_axes):
                val = gradients[i, j]
                gradients_dict[response_type][axis] = None if np.isnan(val) else float(val)
        import json
        with open("movement_relationship_gradients.json", "w") as f:
            json.dump(gradients_dict, f, indent=2)

        # Create the analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Correlation heatmap
        im1 = ax1.imshow(correlations, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(unique_axes)))
        ax1.set_xticklabels(unique_axes)
        ax1.set_yticks(range(len(unique_responses)))
        ax1.set_yticklabels(unique_responses)
        ax1.set_title('Correlation: Response vs Commanded Movement')
        ax1.set_xlabel('Commanded Axis')
        ax1.set_ylabel('Response Variable')

        # Add correlation values as text
        for i in range(len(unique_responses)):
            for j in range(len(unique_axes)):
                if not np.isnan(correlations[i, j]):
                    ax1.text(j, i, f'{correlations[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)

        plt.colorbar(im1, ax=ax1, label='Correlation Coefficient')

        # Gradient heatmap
        gradient_max = np.nanmax(np.abs(gradients))
        if gradient_max > 0:
            im2 = ax2.imshow(gradients, cmap='RdBu_r', aspect='auto', 
                            vmin=-gradient_max, vmax=gradient_max)
        else:
            im2 = ax2.imshow(gradients, cmap='RdBu_r', aspect='auto')
        ax2.set_xticks(range(len(unique_axes)))
        ax2.set_xticklabels(unique_axes)
        ax2.set_yticks(range(len(unique_responses)))
        ax2.set_yticklabels(unique_responses)
        ax2.set_title('Linear Regression Gradients: Response vs Commanded Movement')
        ax2.set_xlabel('Commanded Axis')
        ax2.set_ylabel('Response Variable')

        # Add gradient values as text
        for i in range(len(unique_responses)):
            for j in range(len(unique_axes)):
                if not np.isnan(gradients[i, j]):
                    ax2.text(j, i, f'{gradients[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8)

        plt.colorbar(im2, ax=ax2, label='Gradient (Response/Command)')

        plt.tight_layout()
        plt.show()

    print(f"Evaluated movement relationships at {successful_positions}/{n_positions} random positions.")
    return axis_results

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
    
    # Test the evaluate_movement function
    # print("\n--- Testing Movement Range Across Random Positions ---")
    # robot.go_home()  # Start from a known good position

    # movement_summary = evaluate_movement_scales(robot, ring, collision_render_manager, config, n_positions=300, n_samples_per_pos=50)
    # print("Movement range evaluation complete. See bar charts for details.")

    # Test the evaluate_movement_relationships function
    print("\n--- Testing Movement Relationships (Independent Axes) ---")
    robot.go_home()  # Start from a known good position
    
    relationship_results = evaluate_movement_relationships(robot, ring, collision_render_manager, config, n_positions=1000, n_samples_per_pos=20)
    print("Movement relationship evaluation complete. See scatter plots for details.")

    # # Test the evaluate_gripper_tolerances function
    # print("\n--- Testing Gripper Tolerances Evaluation ---")
    # if robot.last_solve_successful:
    #     print("Testing gripper tolerances at random pose...")
    #     tolerance_results = evaluate_gripper_tolerances(robot, ring, collision_render_manager, 
    #                                                    max_range=50, angle_range_deg=10, n_search=50)

    #     print(f"Gripper tolerance results:")
    #     for key, (neg, pos) in tolerance_results.items():
    #         if 'trans' in key:
    #             print(f"{key}: {neg:.2f} mm to {pos:.2f} mm (range: {pos-neg:.2f} mm)")
    #         else:
    #             print(f"{key}: {neg:.2f}° to {pos:.2f}° (range: {pos-neg:.2f}°)")
    #     print("Bar chart displayed.")
    #     print("Tolerance limits and ranges saved as gripper_tolerances.json")
    # else:
    #     print("Could not find valid pose for tolerance testing")
    