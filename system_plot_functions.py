import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_system(robot, ax=None, ring=None):
    """
    Draws the current state of a robot's kinematics on a 3D matplotlib axis.

    This function visualizes the key points and linkages of a robot model. 
    It plots the actuator points (A), linkage points (C), the plane origin (E1), 
    the gripper (G1), and any grab points. It also shows the movement range of the 
    actuators and orientation vectors at the gripper.

    Args:
        robot: An object representing the robot, which must contain attributes for 
               the positions of points A1-A4, C1-C4, E1, G1, orientation vectors 
               (approach, tangential, radial), and parameters like actuator home/end 
               positions (`params`). It should also have a `last_solve_successful`
               boolean flag and a `verbosity` attribute.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): 
               The 3D matplotlib axis to draw on. If None, a new figure and 
               3D axis are created. Defaults to None.
        ring (Ring, optional): A Ring object to visualize alongside the robot.

    Returns:
        matplotlib.axes._subplots.Axes3DSubplot: The axis on which the visualization was drawn.
    """
    # --- Pre-computation and Setup ---
    
    # Do not attempt to visualize if the last kinematic solve failed.
    if not robot.last_solve_successful:
        if robot.verbosity >= 1: 
            print("Cannot visualize: Last solver execution failed. Check console for details.")
        return None

    # If no axis is provided, create a new figure and a 3D subplot.
    show_plot = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True

    # Helper function to simplify plotting lines between two 3D points.
    def plot_line(p1, p2, style='k-', lw=2):
        """Plots a line between two 3D points p1 and p2 with a specified linewidth."""
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], style, linewidth=lw)

    # --- Plotting Robot Components ---

    # Plot the actuator travel paths (from home to end positions) as dashed lines.
    for i in range(1, 5):
        p_home = robot.params['A_home'][f'A{i}']
        p_end = robot.params['A_end'][f'A{i}']
        ax.plot([p_home[0], p_end[0]], [p_home[1], p_end[1]], [p_home[2], p_end[2]], 'k--', alpha=0.3, label=f'Actuator {i} Path' if i==1 else "")

    # Plot the key points of the robot using scatter plots for visibility.
    ax.scatter(robot.A1[0], robot.A1[1], robot.A1[2], c='b', marker='s', label="A-Points (Actuators)")
    ax.scatter([robot.A2[0], robot.A3[0], robot.A4[0]], [robot.A2[1], robot.A3[1], robot.A4[1]], [robot.A2[2], robot.A3[2], robot.A4[2]], c='b', marker='s')
    ax.scatter([robot.C1[0], robot.C2[0], robot.C3[0], robot.C4[0]], [robot.C1[1], robot.C2[1], robot.C3[1], robot.C4[1]], [robot.C1[2], robot.C2[2], robot.C3[2], robot.C4[2]], c='r', marker='o', label="C-Points (Linkage)")
    ax.scatter(robot.E1[0], robot.E1[1], robot.E1[2], c='purple', marker='x', s=100, label="E1 (Plane Origin)")
    ax.scatter(robot.G1[0], robot.G1[1], robot.G1[2], c='g', marker='*', s=150, label="G1 (Gripper)")

    # --- Plotting Grab Points and Connectors ---
    grab_points_to_plot = []
    if robot.gripper_back is not None and robot.approach_vec is not None:
        gripper_back_pt = robot.gripper_back
        u = robot.approach_vec
        other_grab_points = [robot.outer_grip_point, robot.inner_grip_point_1, robot.inner_grip_point_2]
        
        all_grabs = [p for p in other_grab_points if p is not None]
        if all_grabs:
            all_grabs.append(gripper_back_pt)
            grab_points_to_plot.extend(all_grabs)
            for p in other_grab_points:
                if p is None: continue
                v = gripper_back_pt - p
                corner_point = p + np.dot(v, u) * u
                plot_line(p, corner_point, 'k-', lw=2.5)
                plot_line(corner_point, gripper_back_pt, 'k-', lw=2.5)

    # Plot the linkages (solid lines) connecting the points.
    plot_line(robot.A1, robot.C1, 'b-', lw=2.5); plot_line(robot.A2, robot.C2, 'b-', lw=2.5)
    plot_line(robot.A3, robot.C3, 'b-', lw=2.5); plot_line(robot.A4, robot.C4, 'b-', lw=2.5)
    plot_line(robot.C1, robot.C2, 'r-', lw=2.5); plot_line(robot.C2, robot.C4, 'r-', lw=2.5)
    plot_line(robot.C4, robot.C3, 'r-', lw=2.5); plot_line(robot.C3, robot.C1, 'r-', lw=2.5)
    plot_line(robot.E1, robot.gripper_back, 'g-', lw=3)

    # --- Plotting Orientation Vectors ---
    arrow_length = 100
    ax.quiver(robot.G1[0], robot.G1[1], robot.G1[2], robot.approach_vec[0], robot.approach_vec[1], robot.approach_vec[2], length=arrow_length, color='g', label="Approach")
    ax.quiver(robot.G1[0], robot.G1[1], robot.G1[2], robot.tangential_vec[0], robot.tangential_vec[1], robot.tangential_vec[2], length=arrow_length, color='orange', label="Tangential")
    ax.quiver(robot.G1[0], robot.G1[1], robot.G1[2], robot.radial_vec[0], robot.radial_vec[1], robot.radial_vec[2], length=arrow_length, color='purple', label="Radial")

    # --- Plotting Camera Vectors ---
    if robot.camera_pos is not None and robot.camera_view_vector is not None:
        cp = robot.camera_pos
        cv = robot.camera_view_vector
        ax.quiver(cp[0], cp[1], cp[2], cv[0], cv[1], cv[2], length=arrow_length, color='magenta', label="Camera View", linewidth=3)
    if hasattr(robot, 'camera_up_vector') and robot.camera_pos is not None and robot.camera_up_vector is not None:
        cu = robot.camera_up_vector
        ax.quiver(cp[0], cp[1], cp[2], cu[0], cu[1], cu[2], length=arrow_length, color='deepskyblue', label="Camera Up", linewidth=3)


    # --- Visualize the Ring ---
    if ring:
        visualize_ring(ring, ax)

    # --- Final Plot Adjustments ---
    ax.legend()
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Robot Kinematics Visualization')

    all_points_list = [robot.A1, robot.A2, robot.A3, robot.A4, robot.C1, robot.C2, robot.C3, robot.C4, robot.E1, robot.G1]
    if grab_points_to_plot:
        all_points_list.extend(grab_points_to_plot)
    if robot.camera_pos is not None:
        all_points_list.append(robot.camera_pos)
        
    all_points = np.array(all_points_list)
    all_points = np.vstack([all_points, *robot.params['A_home'].values(), *robot.params['A_end'].values()])
    
    x_all, y_all, z_all = all_points[:,0], all_points[:,1], all_points[:,2]
    mid_x, mid_y, mid_z = (x_all.max() + x_all.min()) * 0.5, (y_all.max() + y_all.min()) * 0.5, (z_all.max() + z_all.min()) * 0.5
    # This logic ensures the axes are scaled equally by finding the largest dimension
    # of the robot's bounding box and applying it to all three axes.
    max_range = np.array([x_all.max()-x_all.min(), y_all.max()-y_all.min(), z_all.max()-z_all.min()]).max()
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    if show_plot:
        plt.show()
    
    return ax

def visualize_ring(ring, ax):
    """
    Draws the ring as two annular sections (outer and inner) on a given 3D matplotlib axis.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to draw on.
    """
    # --- Draw Outer Annulus (outer_radius to middle_radius, thickness: outer_thickness) ---
    u = np.linspace(0, 2 * np.pi, 20)  # Lower angular resolution
    r_outer = np.linspace(ring.middle_radius, ring.outer_radius, 10)  # Lower radial resolution
    h_outer = np.linspace(0, -ring.outer_thickness, 2)
    r_outer_grid, u_outer_grid = np.meshgrid(r_outer, u, indexing='ij')
    def transform(x, y, z):
        return (
            ring.position[0] + x * ring.tangential_vec[0] + y * ring.radial_vec[0] + z * ring.approach_vec[0],
            ring.position[1] + x * ring.tangential_vec[1] + y * ring.radial_vec[1] + z * ring.approach_vec[1],
            ring.position[2] + x * ring.tangential_vec[2] + y * ring.radial_vec[2] + z * ring.approach_vec[2]
        )
    # Top and bottom faces
    for h in h_outer:
        x_outer = r_outer_grid * np.cos(u_outer_grid)
        y_outer = r_outer_grid * np.sin(u_outer_grid)
        z_outer = np.full_like(x_outer, h)
        Xo, Yo, Zo = transform(x_outer, y_outer, z_outer)
        ax.plot_surface(Xo, Yo, Zo, color='c', alpha=0.4)
    # Vertical faces (outer and inner radius)
    # Outer radius face
    u_side = np.linspace(0, 2 * np.pi, 20)
    h_side = np.linspace(0, -ring.outer_thickness, 2)
    u_side_grid, h_side_grid = np.meshgrid(u_side, h_side, indexing='ij')
    x_outer_side = ring.outer_radius * np.cos(u_side_grid)
    y_outer_side = ring.outer_radius * np.sin(u_side_grid)
    z_outer_side = h_side_grid
    Xo_side, Yo_side, Zo_side = transform(x_outer_side, y_outer_side, z_outer_side)
    ax.plot_surface(Xo_side, Yo_side, Zo_side, color='c', alpha=0.4)
    # Inner radius face (for outer annulus)
    x_inner_side = ring.middle_radius * np.cos(u_side_grid)
    y_inner_side = ring.middle_radius * np.sin(u_side_grid)
    z_inner_side = h_side_grid
    Xi_side, Yi_side, Zi_side = transform(x_inner_side, y_inner_side, z_inner_side)
    ax.plot_surface(Xi_side, Yi_side, Zi_side, color='c', alpha=0.4)

    # --- Draw Inner Annulus (middle_radius to inner_radius, thickness: inner_thickness) ---
    r_inner = np.linspace(ring.inner_radius, ring.middle_radius, 10)
    h_inner = np.linspace(0, -ring.inner_thickness, 2)
    r_inner_grid, u_inner_grid = np.meshgrid(r_inner, u, indexing='ij')
    for h in h_inner:
        x_inner = r_inner_grid * np.cos(u_inner_grid)
        y_inner = r_inner_grid * np.sin(u_inner_grid)
        z_inner = np.full_like(x_inner, h)
        Xn, Yn, Zn = transform(x_inner, y_inner, z_inner)
        ax.plot_surface(Xn, Yn, Zn, color='c', alpha=0.4)
    # Vertical faces for inner annulus
    # Outer face (middle_radius) -- use correct height for inner annulus
    h_mid_side = np.linspace(0, -ring.inner_thickness, 2)
    u_mid_side_grid, h_mid_side_grid = np.meshgrid(u_side, h_mid_side, indexing='ij')
    x_mid_side = ring.middle_radius * np.cos(u_mid_side_grid)
    y_mid_side = ring.middle_radius * np.sin(u_mid_side_grid)
    z_mid_side = h_mid_side_grid
    Xm_side, Ym_side, Zm_side = transform(x_mid_side, y_mid_side, z_mid_side)
    ax.plot_surface(Xm_side, Ym_side, Zm_side, color='c', alpha=0.4)
    # Inner face (inner_radius) -- use correct height for inner annulus
    h_inner_side = np.linspace(0, -ring.inner_thickness, 2)
    u_inner_side_grid, h_inner_side_grid = np.meshgrid(u_side, h_inner_side, indexing='ij')
    x_innermost_side = ring.inner_radius * np.cos(u_inner_side_grid)
    y_innermost_side = ring.inner_radius * np.sin(u_inner_side_grid)
    z_innermost_side = h_inner_side_grid
    Xin_side, Yin_side, Zin_side = transform(x_innermost_side, y_innermost_side, z_innermost_side)
    ax.plot_surface(Xin_side, Yin_side, Zin_side, color='c', alpha=0.4)

class DummyRing:
    def __init__(self):
        self.outer_radius = 384
        self.middle_radius = 283
        self.inner_radius = 265
        self.outer_thickness = 6
        self.inner_thickness = 26
        self.position = np.array([0, 0, 0])
        self.tangential_vec = np.array([1, 0, 0])
        self.radial_vec = np.array([0, 1, 0])
        self.approach_vec = np.array([0, 0, 1])

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ring = DummyRing()
    visualize_ring(ring, ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()