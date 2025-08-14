"""
Action Scaling Analysis

This script analyzes the relationship between movement commands and their effects
on different degrees of freedom (DOF) relative to their tolerance ranges.

The analysis calculates what proportion of each tolerance range is consumed
when a unit movement (1) is commanded in each action direction (dx, dy, drx, drz).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def load_json_data():
    """
    Load the gripper tolerances, movement relationship gradients, and robot config.
    
    Returns:
        tuple: (tolerances_dict, gradients_dict, robot_config_dict)
    """
    # Load gripper tolerances
    with open('gripper_tolerances.json', 'r') as f:
        tolerances = json.load(f)
    
    # Load movement relationship gradients
    with open('movement_relationship_gradients.json', 'r') as f:
        gradients = json.load(f)
    
    # Load robot configuration (YAML format)
    import yaml
    with open('config/robot_config.yaml', 'r') as f:
        robot_config = yaml.safe_load(f)
    
    return tolerances, gradients, robot_config


def calculate_proportion_matrix(tolerances, gradients, robot_config):
    """
    Calculate the proportion of tolerance range consumed by unit movements.
    
    For each movement type and each action (dx, dy, drx, drz), calculates:
    proportion = |gradient_value| / tolerance_range
    
    Args:
        tolerances (dict): Tolerance ranges for each DOF
        gradients (dict): Movement gradients for each movement type
        robot_config (dict): Robot configuration including tilt limits
        
    Returns:
        pandas.DataFrame: Matrix of proportions with movements as rows and actions as columns
    """
    # Define the actions we're analyzing
    actions = ['dx', 'dy', 'drx', 'drz']
    
    # Add rx_E1 and ry_E1 movements based on robot config tilt limits
    enhanced_gradients = gradients.copy()
    
    # Extract tilt limits from robot config (convert degrees to appropriate units)
    tilt_rx_limit = robot_config['tilt_rx_limit_deg']  # degrees
    tilt_rz_limit = robot_config['tilt_rz_limit_deg']  # degrees
    
    # Add rx_E1 movement (gradient of 1 to rx command, 0 to others)
    enhanced_gradients['rx_E1'] = {
        'dx': 0.0,
        'dy': 0.0, 
        'drx': 1.0,  # Direct 1:1 mapping to rx command
        'drz': 0.0
    }
    
    # Add ry_E1 movement (gradient of 1 to ry command, 0 to others)
    # Note: ry maps to drz in the coordinate system based on robot config structure
    enhanced_gradients['ry_E1'] = {
        'dx': 0.0,
        'dy': 0.0,
        'drx': 0.0,
        'drz': 1.0   # Direct 1:1 mapping to rz command (ry in robot frame)
    }
    
    # Define tolerance mapping for each action
    # dx, dy -> translational tolerances, drx, drz -> rotational tolerances
    tolerance_mapping = {
        'dx': ['trans_approach', 'trans_radial', 'trans_tangential'],
        'dy': ['trans_approach', 'trans_radial', 'trans_tangential'], 
        'drx': ['rot_approach', 'rot_radial', 'rot_tangential'],
        'drz': ['rot_approach', 'rot_radial', 'rot_tangential']
    }
    
    # Extension movements have a fixed range of 400mm
    EXTENSION_RANGE = 400.0
    
    # E1 tilt ranges are 2 * abs(tilt_limit) as specified
    E1_RX_RANGE = 2 * abs(tilt_rx_limit)  # Total range for rx_E1
    E1_RZ_RANGE = 2 * abs(tilt_rz_limit)  # Total range for ry_E1 (mapped to rz)
    
    # Initialize the proportion matrix
    movements = list(enhanced_gradients.keys())
    proportion_matrix = np.zeros((len(movements), len(actions)))
    
    # Calculate proportions for each movement-action combination
    for i, movement in enumerate(movements):
        for j, action in enumerate(actions):
            # Get the gradient value for this movement-action combination
            gradient_value = enhanced_gradients[movement].get(action, 0.0)
            
            # Determine the appropriate tolerance range
            if movement.startswith('Ext_'):
                # Extension movements use fixed 400mm range
                tolerance_range = EXTENSION_RANGE
            elif movement == 'rx_E1':
                # rx_E1 uses the rx tilt limit range
                if action == 'drx':
                    tolerance_range = E1_RX_RANGE
                else:
                    tolerance_range = 1.0  # Non-applicable actions get unity range
            elif movement == 'ry_E1':
                # ry_E1 uses the rz tilt limit range (ry maps to rz)
                if action == 'drz':
                    tolerance_range = E1_RZ_RANGE
                else:
                    tolerance_range = 1.0  # Non-applicable actions get unity range
            else:
                # For gripper movements, determine which tolerance applies
                tolerance_range = get_tolerance_range_for_movement(
                    movement, action, tolerances, tolerance_mapping
                )
            
            # Calculate proportion (use absolute value of gradient)
            if tolerance_range > 0:
                proportion = abs(gradient_value) / tolerance_range
            else:
                proportion = 0.0
            
            proportion_matrix[i, j] = proportion
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(
        proportion_matrix,
        index=movements,
        columns=actions
    )
    
    # Convert to percentages
    df = df * 100
    
    # Add average row for gripper-related (G1) movements
    g1_movements = [mov for mov in movements if mov.startswith('G1_')]
    if g1_movements:
        g1_averages = df.loc[g1_movements].mean()
        df.loc['G1_Average'] = g1_averages

    # Add average row for extension movements
    ext_movements = [mov for mov in movements if mov.startswith('Ext_')]
    if ext_movements:
        ext_averages = df.loc[ext_movements].mean()
        df.loc['Extension_Average'] = ext_averages

    # Add overall average row (excluding other averages)
    base_movements = [mov for mov in movements]
    overall_averages = df.loc[base_movements].mean()
    df.loc['Overall_Average'] = overall_averages

    return df


def get_tolerance_range_for_movement(movement, action, tolerances, tolerance_mapping):
    """
    Determine the appropriate tolerance range for a given movement and action.
    
    Args:
        movement (str): Movement name (e.g., 'G1_Approach', 'G1_Radial')
        action (str): Action type ('dx', 'dy', 'drx', 'drz')
        tolerances (dict): Tolerance data
        tolerance_mapping (dict): Mapping of actions to tolerance types
        
    Returns:
        float: Tolerance range value
    """
    # Map movement names to their primary DOF
    movement_to_dof = {
        'G1_Approach': 'trans_approach',
        'G1_Radial': 'trans_radial', 
        'G1_Tangential': 'trans_tangential',
        'G1_Rot_Approach': 'rot_approach',
        'G1_Rot_Radial': 'rot_radial',
        'G1_Rot_Tangential': 'rot_tangential'
    }
    
    # Get the primary DOF for this movement
    primary_dof = movement_to_dof.get(movement)
    
    if primary_dof:
        # Use the primary DOF tolerance
        return tolerances[primary_dof]['range']
    else:
        # For unknown movements, use a default approach based on action type
        if action in ['dx', 'dy']:
            # Use approach tolerance as default for translational actions
            return tolerances['trans_approach']['range']
        else:
            # Use approach tolerance as default for rotational actions
            return tolerances['rot_approach']['range']


def create_heatmap(proportion_df):
    """
    Create a heatmap visualization of the proportion matrix.
    
    Args:
        proportion_df (pandas.DataFrame): Matrix of proportions to visualize
    """
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(
        proportion_df,
        annot=True,  # Show values in cells
        fmt='.2f',   # Format numbers to 2 decimal places for percentages
        cmap='YlOrRd',  # Color map (yellow to orange to red)
        cbar_kws={'label': 'Percentage of Tolerance Range (%)'},
        ax=ax,
        linewidths=0.5,  # Add grid lines
        square=False  # Don't force square cells
    )
    
    # Customize the plot
    ax.set_title(
        'Movement Command Sensitivity Analysis\n'
        'Percentage of Tolerance Range Consumed per Unit Command\n'
        '(Including E1 Tilt Movements: rx_E1, ry_E1)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Action Commands', fontsize=12, fontweight='bold')
    ax.set_ylabel('Movement Types', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('action_scaling_heatmap.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()


def print_analysis_summary(proportion_df):
    """
    Print a summary of the analysis results.
    
    Args:
        proportion_df (pandas.DataFrame): Matrix of percentages
    """
    print("\n" + "="*80)
    print("ACTION SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nHighest sensitivity combinations (top 5):")
    print("-" * 50)
    
    # Find the highest percentages (exclude the average row for ranking)
    df_no_avg = proportion_df.drop('G1_Average', errors='ignore')
    max_values = []
    for movement in df_no_avg.index:
        for action in df_no_avg.columns:
            value = df_no_avg.loc[movement, action]
            max_values.append((value, movement, action))
    
    # Sort and display top 5
    max_values.sort(reverse=True)
    for i, (value, movement, action) in enumerate(max_values[:5]):
        print(f"{i+1}. {movement} -> {action}: {value:.2f}%")
    
    print("\nLowest sensitivity combinations (bottom 5):")
    print("-" * 50)
    
    # Display bottom 5 (excluding zeros)
    non_zero_values = [(v, m, a) for v, m, a in max_values if v > 0]
    for i, (value, movement, action) in enumerate(non_zero_values[-5:]):
        print(f"{i+1}. {movement} -> {action}: {value:.2f}%")
    
    print("\nMovement type averages:")
    print("-" * 30)
    for movement in proportion_df.index:
        avg_sensitivity = proportion_df.loc[movement].mean()
        if movement == 'G1_Average':
            print(f"{movement}: {avg_sensitivity:.2f}% (Average of all G1 movements)")
        elif movement == 'Extension_Average':
            print(f"{movement}: {avg_sensitivity:.2f}% (Average of all Extension movements)")
        elif movement == 'Overall_Average':
            print(f"{movement}: {avg_sensitivity:.2f}% (Average of all movements)")
        else:
            print(f"{movement}: {avg_sensitivity:.2f}%")

    print("\nAction type averages:")
    print("-" * 25)
    for action in proportion_df.columns:
        avg_sensitivity = proportion_df[action].mean()
        print(f"{action}: {avg_sensitivity:.2f}%")

    # Highlight the G1 average row specifically
    if 'G1_Average' in proportion_df.index:
        print(f"\nG1 Gripper Movements Summary:")
        print("-" * 35)
        g1_avg_row = proportion_df.loc['G1_Average']
        for action in g1_avg_row.index:
            print(f"Average G1 sensitivity to {action}: {g1_avg_row[action]:.2f}%")
    # Highlight the Extension average row specifically
    if 'Extension_Average' in proportion_df.index:
        print(f"\nExtension Movements Summary:")
        print("-" * 35)
        ext_avg_row = proportion_df.loc['Extension_Average']
        for action in ext_avg_row.index:
            print(f"Average Extension sensitivity to {action}: {ext_avg_row[action]:.2f}%")
    # Highlight the Overall average row specifically
    if 'Overall_Average' in proportion_df.index:
        print(f"\nOverall Movements Summary:")
        print("-" * 35)
        overall_avg_row = proportion_df.loc['Overall_Average']
        for action in overall_avg_row.index:
            print(f"Overall average sensitivity to {action}: {overall_avg_row[action]:.2f}%")


def main():
    """
    Main execution function that orchestrates the analysis.
    """
    print("Loading tolerance, gradient, and robot configuration data...")
    
    # Load the JSON and YAML data files
    tolerances, gradients, robot_config = load_json_data()
    
    print(f"Loaded {len(tolerances)} tolerance ranges")
    print(f"Loaded {len(gradients)} movement types")
    print(f"Loaded robot config with tilt limits: rx={robot_config['tilt_rx_limit_deg']}°, rz={robot_config['tilt_rz_limit_deg']}°")
    
    # Calculate the proportion matrix
    print("\nCalculating percentage matrix (including E1 tilt movements)...")
    proportion_df = calculate_proportion_matrix(tolerances, gradients, robot_config)
    
    # Display the raw matrix
    print("\nPercentage Matrix:")
    print("=" * 60)
    print(proportion_df.to_string(float_format='%.2f'))
    
    # Create visualization
    print("\nGenerating heatmap visualization...")
    create_heatmap(proportion_df)
    
    # Print analysis summary
    print_analysis_summary(proportion_df)
    
    print("\nAnalysis complete! Heatmap saved as 'action_scaling_heatmap.png'")
    print("\nNote: rx_E1 and ry_E1 movements added with tolerance ranges:")
    print(f"  - rx_E1: {2 * abs(robot_config['tilt_rx_limit_deg'])}° (2 × |{robot_config['tilt_rx_limit_deg']}°|)")
    print(f"  - ry_E1: {2 * abs(robot_config['tilt_rz_limit_deg'])}° (2 × |{robot_config['tilt_rz_limit_deg']}°|)")
    print("  - Both have gradient of 1.0 to their respective rotation commands (drx, drz)")


if __name__ == "__main__":
    main()