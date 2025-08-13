import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

# This helper function is assumed to be in a separate file.
from geometry_helper_functions import _get_line_segment_sphere_intersections
from system_plot_functions import visualize_system

def load_yaml_config(filename):
    """
    Load a YAML configuration file and return its contents as a dictionary.
    
    Args:
        filename (str): Path to the YAML file.
        
    Returns:
        dict: Parsed YAML data.
        
    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


class Ring:
    """
    A 3D hollow annulus (ring) with position and orientation in space.

    This class represents a ring-shaped object with geometric properties and spatial
    orientation. Default parameters are loaded from config/ring_config.yaml when
    created via RobotKinematics.create_ring().

    Attributes:
        outer_radius (float): Outer radius of the ring.
        middle_radius (float): Middle radius of the ring (for stepped or multi-layer rings).
        inner_radius (float): Inner radius of the ring.
        outer_thickness (float): Thickness at the outer radius.
        inner_thickness (float): Thickness at the inner radius.
        position (np.ndarray): 3D center position in world coordinates.
        approach_vec (np.ndarray): Unit vector normal to ring surface (central axis).
        tangential_vec (np.ndarray): Unit vector in the ring plane.
        radial_vec (np.ndarray): Unit vector orthogonal to tangential_vec in ring plane.
    """
    
    __slots__ = ['outer_radius', 'middle_radius', 'inner_radius', 'outer_thickness', 
                 'inner_thickness', 'position', 'approach_vec', 'tangential_vec', 'radial_vec']
    
    def __init__(
        self,
        outer_radius=192.0,
        middle_radius=141.5,
        inner_radius=132.5,
        outer_thickness=6.0,
        inner_thickness=26.0,
        position=None,
        approach_vec=None,
        tangential_vec=None,
        radial_vec=None
    ):
        """
        Initialize the Ring object.

        Args:
            outer_radius (float): Outer radius of the ring.
            middle_radius (float): Middle radius of the ring.
            inner_radius (float): Inner radius of the ring.
            outer_thickness (float): Outer thickness of the ring.
            inner_thickness (float): Inner thickness of the ring.
            position (np.ndarray, optional): 3D center position.
            approach_vec (np.ndarray, optional): Unit vector normal to ring surface.
            tangential_vec (np.ndarray, optional): Unit vector in ring plane.
            radial_vec (np.ndarray, optional): Unit vector orthogonal to tangential_vec.
        """
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.middle_radius = middle_radius
        self.outer_thickness = outer_thickness
        self.inner_thickness = inner_thickness
        self.position = position
        self.approach_vec = approach_vec
        self.tangential_vec = tangential_vec
        self.radial_vec = radial_vec

    def __repr__(self):
        """Returns a comprehensive, multi-line string representation of the Ring's properties."""
        def r(vec):
            return "None" if vec is None else np.round(vec, 2)
        header = "--- Ring State ---"
        geom_header = "\n--- Geometry ---"
        geom_str = (f"Outer Radius:      {self.outer_radius}\n"
                    f"Middle Radius:     {self.middle_radius}\n"
                    f"Inner Radius:      {self.inner_radius}\n"
                    f"Outer Thickness:   {self.outer_thickness}\n"
                    f"Inner Thickness:   {self.inner_thickness}")
        pos_header = "\n--- Position ---"
        pos_str = f"Center Position:   {r(self.position)}"
        orient_header = "\n--- Orientation Vectors ---"
        orient_str = (f"Approach Vector:   {r(self.approach_vec)}\n"
                      f"Tangential Vector: {r(self.tangential_vec)}\n"
                      f"Radial Vector:     {r(self.radial_vec)}")
        footer = "\n" + "-" * len(header)
        return "\n".join([header, geom_header, geom_str, pos_header, pos_str, orient_header, orient_str, footer])

class RobotKinematics:
    """
    Stateful kinematics solver for a 4-point linkage robot system.

    This class manages inverse kinematics calculations for a robot with 4 actuated linkages.
    It calculates actuator extensions and end-effector poses, tracks robot state, and provides
    methods for gripper interaction analysis. Configuration is loaded from config/robot_config.yaml.

    Key Capabilities:
        - Inverse kinematics solving from end-effector poses
        - Actuator extension calculation and validation
        - Gripper pose and orientation management
        - Ring interaction evaluation
        - Random pose generation within workspace bounds

    Attributes:
        params (dict): Robot physical parameters from configuration.
        verbosity (int): Console output detail level (0=silent, 1=basic, 2=verbose).
        extensions (np.ndarray): Current actuator extensions [e1, e2, e3, e4].
        last_solve_successful (bool): Flag indicating if last IK solve succeeded.
        last_error_msg (str): Description of last solve outcome.
        
        # World coordinate arrays for key points
        A1-A4, C1-C4, E1, G1 (np.ndarray): Actuator, linkage, end-effector, gripper points.
        
        # Orientation vectors
        approach_vec, tangential_vec, radial_vec (np.ndarray): Gripper orientation basis.
        
        # Delta tracking (for incremental moves)
        delta_extensions, delta_E1, delta_E1_quaternion (np.ndarray): Change tracking arrays.
    """

    def __init__(self, params=None, extensions=None, verbosity=2):
        """
        Initialize the kinematics solver with robot parameters and initial state.
        
        Args:
            params (dict, optional): Robot parameters. If None, loads from config/robot_config.yaml.
            extensions (array-like, optional): Initial actuator extensions [e1, e2, e3, e4].
            verbosity (int): Output verbosity level (0=silent, 1=basic, 2=verbose).
        """
        self.verbosity = verbosity
        self.params = self._load_and_process_params(params)
        self._initialize_bounds()
        self._initialize_state_arrays(extensions)
        self._precompute_geometry()
        self._initialize_deltas()
        self._initialize_home_pose()
        
        # Initialize to home position
        self.go_home()

    def _load_and_process_params(self, params):
        """Load and process robot parameters from config or use provided params."""
        if params is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'robot_config.yaml')
            params = load_yaml_config(config_path)
            
            # Convert parameter lists to numpy arrays for efficiency
            array_sections = ['A_home', 'A_end', 'C_local', 'grab_points_local']
            for section in array_sections:
                for key, val in params[section].items():
                    params[section][key] = np.array(val, dtype=np.float32)
            
            params['G_local'] = np.array(params['G_local'], dtype=np.float32)
            cam_params = params['camera_params']
            cam_params['local_pos_offset'] = np.array(cam_params['local_pos_offset'], dtype=np.float32)
            cam_params['local_rotation_deg'] = np.array(cam_params['local_rotation_deg'], dtype=np.float32)
        
        return params

    def _initialize_bounds(self):
        """Initialize reachable radius for random pose generation."""
        # Load reachable radius for random pose generation
        self.reachable_radius = self.params.get('reachable_radius', 100.0)

    def _initialize_state_arrays(self, extensions):
        """Initialize all state arrays with proper dtypes."""
        # Actuator extensions
        if extensions is not None:
            self.extensions = np.array(extensions, dtype=np.float32)
        else:
            self.extensions = np.zeros(4, dtype=np.float32)
        
        # Initialize coordinate arrays
        self.A1 = self.A2 = self.A3 = self.A4 = np.zeros(3, dtype=np.float32)
        self.C1 = self.C2 = self.C3 = self.C4 = np.zeros(3, dtype=np.float32)
        self.E1 = self.G1 = np.zeros(3, dtype=np.float32)
        
        # Orientation vectors
        self.approach_vec = self.tangential_vec = self.radial_vec = np.zeros(3, dtype=np.float32)
        self.plane_normal = np.zeros(3, dtype=np.float32)
        
        # Euler angles
        self.rx = self.ry = self.rz = 0.0
        
        # Gripper points
        self.outer_grip_point = self.inner_grip_point_1 = np.zeros(3, dtype=np.float32)
        self.inner_grip_point_2 = self.gripper_back = np.zeros(3, dtype=np.float32)

        self.gripped = False # track if the gripper is open

        # Camera
        self.camera_pos = self.camera_view_vector = self.camera_up_vector = np.zeros(3, dtype=np.float32)
        
        # Status tracking
        self.last_solve_successful = False
        self.last_error_msg = "State has not been solved yet."

    def _precompute_geometry(self):
        """Precompute geometric quantities for efficiency."""
        # Precompute local distance measurements
        c_local = self.params['C_local']
        self.dist_c1_c2_local = np.linalg.norm(c_local['C1'] - c_local['C2'])
        self.dist_c1_c3_local = np.linalg.norm(c_local['C1'] - c_local['C3'])
        self.dist_c2_c3_local = np.linalg.norm(c_local['C2'] - c_local['C3'])
        
        # Precompute actuator track data
        self.params['actuator_tracks'] = {}
        for key in ['A1', 'A2', 'A3', 'A4']:
            home, end = self.params['A_home'][key], self.params['A_end'][key]
            vec = end - home
            dist = np.linalg.norm(vec)
            self.params['actuator_tracks'][key] = {
                'vec': vec.astype(np.float32), 
                'dist': np.float32(dist)
            }
        
        # Precompute local points matrix for vectorized operations
        self.local_points_keys = ['C1', 'C2', 'C3', 'C4']
        self.local_points_matrix = np.vstack([
            self.params['C_local'][k] for k in self.local_points_keys
        ]).astype(np.float32)
        
        # Precompute grab points for efficiency
        self.grab_points_local = {
            k: np.array(v, dtype=np.float32) 
            for k, v in self.params['grab_points_local'].items()
        }
        
        # Precompute camera rotation
        self.cam_rot = R.from_euler(
            'xyz', self.params['camera_params']['local_rotation_deg'], degrees=True
        )

    def _initialize_deltas(self):
        """Initialize delta tracking arrays."""
        self.delta_extensions = np.zeros(4, dtype=np.float32)
        self.E1_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity quaternion
        self.delta_E1 = np.zeros(3, dtype=np.float32)
        self.delta_E1_quaternion = np.zeros(4, dtype=np.float32)

    def _initialize_home_pose(self):
        """Initialize home pose parameters."""
        home_pose = self.params.get('E1_home', {'x': 0, 'y': 350, 'rx': 0, 'rz': 0})
        self.E1_home_x = home_pose.get('x', 0)
        self.E1_home_y = home_pose.get('y', 350)
        self.E1_home_rx = home_pose.get('rx', 0)
        self.E1_home_rz = home_pose.get('rz', 0)

    def _zero_deltas(self):
        """Zero all delta tracking arrays."""
        self.delta_extensions.fill(0.0)
        self.delta_E1.fill(0.0)
        self.delta_E1_quaternion.fill(0.0)

    def create_ring(self,
        outer_radius=None,
        middle_radius=None,
        inner_radius=None,
        outer_thickness=None,
        inner_thickness=None,
        ring=None
    ):
        """
        Create or update a Ring instance positioned relative to the gripper.
        
        If an existing ring is provided, only its position and orientation are updated.
        Otherwise, creates a new Ring using specified or default parameters from
        config/ring_config.yaml.
        
        Args:
            outer_radius (float, optional): Outer radius for new ring.
            middle_radius (float, optional): Middle radius for new ring.
            inner_radius (float, optional): Inner radius for new ring.
            outer_thickness (float, optional): Outer thickness for new ring.
            inner_thickness (float, optional): Inner thickness for new ring.
            ring (Ring, optional): Existing Ring to update. If None, creates new Ring.
            
        Returns:
            Ring or None: Updated/created Ring object if robot state is solved, None otherwise.
        """
        if not self.last_solve_successful:
            if self.verbosity >= 1:
                print("Cannot create/update ring: Robot state has not been successfully solved.")
            return None

        # Update existing ring position and orientation
        if ring is not None:
            self._update_ring_pose(ring)
            return ring

        # Create new ring with specified or default parameters
        ring_params = self._get_ring_parameters(
            outer_radius, middle_radius, inner_radius, outer_thickness, inner_thickness
        )
        
        return self._create_new_ring(ring_params)

    def _update_ring_pose(self, ring):
        """Update position and orientation of existing ring."""
        avg_radius = (ring.middle_radius + ring.inner_radius) * 0.5
        approach_offset = ring.outer_thickness + (ring.inner_thickness - ring.outer_thickness) * 0.5
        offset_vector = avg_radius * self.radial_vec + approach_offset * self.approach_vec
        
        ring.position = self.G1 + offset_vector
        ring.approach_vec = self.approach_vec
        ring.tangential_vec = self.tangential_vec
        ring.radial_vec = self.radial_vec

    def _get_ring_parameters(self, outer_radius, middle_radius, inner_radius, outer_thickness, inner_thickness):
        """Get ring parameters from arguments or load defaults from config."""
        params_needed = [outer_radius, middle_radius, inner_radius, outer_thickness, inner_thickness]
        
        if any(param is None for param in params_needed):
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'ring_config.yaml')
            ring_config = load_yaml_config(config_path)
            
            return {
                'outer_radius': outer_radius or ring_config['outer_radius'],
                'middle_radius': middle_radius or ring_config['middle_radius'],
                'inner_radius': inner_radius or ring_config['inner_radius'],
                'outer_thickness': outer_thickness or ring_config['outer_thickness'],
                'inner_thickness': inner_thickness or ring_config['inner_thickness']
            }
        
        return {
            'outer_radius': outer_radius,
            'middle_radius': middle_radius,
            'inner_radius': inner_radius,
            'outer_thickness': outer_thickness,
            'inner_thickness': inner_thickness
        }

    def _create_new_ring(self, params):
        """Create new Ring instance with calculated position."""
        # Cast parameters to float32 for consistency
        for key in params:
            params[key] = np.float32(params[key])
        
        # Calculate ring position
        avg_radius = (params['middle_radius'] + params['inner_radius']) * 0.5
        approach_offset = params['outer_thickness'] + (params['inner_thickness'] - params['outer_thickness']) * 0.5
        offset_vector = avg_radius * self.radial_vec + approach_offset * self.approach_vec
        ring_center = self.G1 + offset_vector

        return Ring(
            outer_radius=params['outer_radius'],
            middle_radius=params['middle_radius'],
            inner_radius=params['inner_radius'],
            outer_thickness=params['outer_thickness'],
            inner_thickness=params['inner_thickness'],
            position=ring_center,
            approach_vec=self.approach_vec,
            tangential_vec=self.tangential_vec,
            radial_vec=self.radial_vec
        )

    def evaluate_grip(self, ring):
        """
        Evaluate if gripper is correctly positioned to grasp the given ring.
        
        Checks geometric constraints for all gripper contact points:
        - Inner grip points must be within ring inner radius and thickness
        - Outer grip point must be outside middle radius and within ring thickness
        - Gripper back must clear the ring (no collision)
        
        Args:
            ring (Ring): Ring object to evaluate grip against.
            
        Returns:
            bool: True if grip evaluation passes all checks, False otherwise.
        """
        if not self.last_solve_successful:
            if self.verbosity >= 1:
                print("Grip Evaluation Failed: Robot state is not solved.")
            return False

        if self.verbosity >= 1:
            print("\n--- Evaluating Grip ---")
        
        # Define grip point validation parameters
        grip_checks = [
            (self.inner_grip_point_1, "Inner Grip Point 1", True, False, False),
            (self.inner_grip_point_2, "Inner Grip Point 2", True, False, False),
            (self.outer_grip_point, "Outer Grip Point", False, True, False),
            (self.gripper_back, "Gripper Back", False, False, True)
        ]
        
        is_successful = True
        for point, name, is_inner, is_outer, is_clear in grip_checks:
            if not self._check_grip_point(point, name, ring, is_inner, is_outer, is_clear):
                is_successful = False

        if self.verbosity >= 1:
            result = "SUCCESS" if is_successful else "FAILURE"
            print(f"{result}: Grip evaluation {'passed' if is_successful else 'failed'} all checks.")
        
        return is_successful

    def _check_grip_point(self, point, point_name, ring, is_inner, is_outer, is_clear):
        """
        Check geometric constraints for a single grip point.
        
        Args:
            point (np.ndarray): 3D position of the grip point.
            point_name (str): Name for logging/debugging.
            ring (Ring): Ring object for constraint checking.
            is_inner (bool): Whether this is an inner grip point.
            is_outer (bool): Whether this is an outer grip point.
            is_clear (bool): Whether this point needs clearance from ring.
            
        Returns:
            bool: True if all constraints are satisfied, False otherwise.
        """
        # Calculate point position relative to ring
        relative_pos = point - ring.position
        axial_dist = -np.dot(relative_pos, ring.approach_vec)
        radial_dist = np.linalg.norm(relative_pos - axial_dist * ring.approach_vec)
        
        point_valid = True
        
        # Inner grip point constraints
        if is_inner:
            axial_check = 0 < axial_dist < ring.inner_thickness
            radial_check = radial_dist <= ring.inner_radius
            
            if not axial_check:
                point_valid = False
                if self.verbosity >= 1:
                    print(f"FAIL: {point_name} outside inner thickness "
                          f"(axial: {axial_dist:.2f} not in (0, {ring.inner_thickness:.2f}))")
            elif self.verbosity >= 2:
                print(f"PASS: {point_name} within inner thickness")
                
            if not radial_check:
                point_valid = False
                if self.verbosity >= 1:
                    print(f"FAIL: {point_name} outside inner radius "
                          f"(radial: {radial_dist:.2f} > {ring.inner_radius:.2f})")
            elif self.verbosity >= 2:
                print(f"PASS: {point_name} inside inner radius")
        
        # Outer grip point constraints
        elif is_outer:
            axial_check = ring.outer_thickness < axial_dist <= ring.inner_thickness
            radial_check = radial_dist >= ring.middle_radius
            
            if not axial_check:
                point_valid = False
                if self.verbosity >= 1:
                    print(f"FAIL: {point_name} outside ring thickness "
                          f"(axial: {axial_dist:.2f} not in ({ring.outer_thickness:.2f}, {ring.inner_thickness:.2f}])")
            elif self.verbosity >= 2:
                print(f"PASS: {point_name} within ring thickness")
                
            if not radial_check:
                point_valid = False
                if self.verbosity >= 1:
                    print(f"FAIL: {point_name} inside outer radius "
                          f"(radial: {radial_dist:.2f} < {ring.middle_radius:.2f})")
            elif self.verbosity >= 2:
                print(f"PASS: {point_name} outside outer radius")
        
        # Clearance constraints
        elif is_clear:
            clearance_check = (radial_dist > ring.middle_radius) or (axial_dist > ring.inner_thickness)
            if not clearance_check:
                point_valid = False
                if self.verbosity >= 1:
                    print(f"FAIL: {point_name} is colliding with the ring")
            elif self.verbosity >= 2:
                print(f"PASS: {point_name} is clear of the ring")
        
        return point_valid

    def update_from_e1_pose(self, e1_pos, rx_deg, rz_deg):
        """
        Calculate robot state from desired end-effector pose (inverse kinematics).
        
        This is the core IK solver that computes actuator positions and extensions
        required to achieve the specified end-effector pose. The method validates
        tilt limits, checks workspace bounds, and ensures all points remain above
        the ground plane.
        
        Note: E1 z-coordinate is fixed at 0, and ry rotation is constrained to 0.
        
        Args:
            e1_pos (array-like): Target [x, y, z=0] world position for E1.
            rx_deg (float): Target rotation about X-axis (pitch, degrees).
            rz_deg (float): Target rotation about Z-axis (roll, degrees).
            
        Returns:
            bool: True if pose is valid and solvable, False otherwise.
            
        Side Effects:
            Updates all robot state variables if successful, including:
            - Actuator positions (A1-A4) and extensions
            - End-effector points (C1-C4, E1, G1)
            - Orientation vectors and gripper points
            - Camera position and orientation
            - Delta arrays (calculated from previous state)
        """
        # Store previous state for delta calculation (if robot was previously solved)
        if hasattr(self, 'E1') and hasattr(self, 'extensions') and hasattr(self, 'E1_quaternion'):
            prev_e1 = self.E1.copy()
            prev_extensions = self.extensions.copy()
            prev_E1_quaternion = self.E1_quaternion.copy()
            calculate_deltas = True
        else:
            calculate_deltas = False
        
        # --- Step 0: Validate input angles against physical limits ---
        if abs(rx_deg) > self.params['tilt_rx_limit_deg']:
            self.last_solve_successful = False
            self.last_error_msg = f"Target pose exceeds rx tilt limit: |{rx_deg:.1f}| > {self.params['tilt_rx_limit_deg']:.1f} deg."
            if self.verbosity >= 1:
                print(f"IK Failed: {self.last_error_msg}")
            return False

        if abs(rz_deg) > self.params['tilt_rz_limit_deg']:
            self.last_solve_successful = False
            self.last_error_msg = f"Target pose exceeds rz tilt limit: |{rz_deg:.1f}| > {self.params['tilt_rz_limit_deg']:.1f} deg."
            if self.verbosity >= 1:
                print(f"IK Failed: {self.last_error_msg}")
            return False

        if self.verbosity >= 2:
            print(f"\n--- IK: Starting update from E1 pose: pos={e1_pos}, rx={rx_deg}, rz={rz_deg} ---")
        
        rotation = R.from_euler('xyz', [rx_deg, 0, rz_deg], degrees=True)
        E1_w = np.array(e1_pos, dtype=np.float32)
        # Store quaternion for E1 orientation
        E1_quaternion = rotation.as_quat().astype(np.float32)  # [x, y, z, w]

        # Use precomputed local_points_matrix
        c_points_w = rotation.apply(self.local_points_matrix).astype(np.float32) + E1_w
        C1_w, C2_w, C3_w, C4_w = c_points_w
        G1_w = E1_w + rotation.apply(self.params['G_local']).astype(np.float32)
        
        if self.verbosity >= 2:
            print(f"IK Step 1: Calculated C points:\n C1:{np.round(C1_w,2)}, C2:{np.round(C2_w,2)}, C3:{np.round(C3_w,2)}, C4:{np.round(C4_w,2)}")

        all_points = np.vstack([c_points_w, E1_w, G1_w])
        if np.any(all_points[:, 1] < 0):
            failing_idx = np.where(all_points[:, 1] < 0)[0][0]
            point_names = self.local_points_keys + ['E1', 'G1']
            name, point = point_names[failing_idx], all_points[failing_idx]
            self.last_solve_successful = False
            self.last_error_msg = f"Pose is invalid: {name} is below the ground plane (y={point[1]:.2f})."
            if self.verbosity >= 1: print(f"IK Failed: {self.last_error_msg}")
            return False
        if self.verbosity >= 2:
            print("IK Step 2: All points are above ground plane.")

        vec_g1_e1 = G1_w - E1_w
        vec_c1_e1 = C1_w - E1_w

        approach_vec = vec_g1_e1 / np.linalg.norm(vec_g1_e1)
        tangential_vec = np.cross(approach_vec, vec_c1_e1)
        tangential_vec /= np.linalg.norm(tangential_vec)
        radial_vec = np.cross(tangential_vec, approach_vec)
        if self.verbosity >= 2:
            print("IK Step 3: Orientation vectors calculated.")
        
        intersections_a1 = _get_line_segment_sphere_intersections(self.params['A_home']['A1'], self.params['A_end']['A1'], C1_w, self.params['R1'])
        intersections_a2 = _get_line_segment_sphere_intersections(self.params['A_home']['A2'], self.params['A_end']['A2'], C2_w, self.params['R1'])
        intersections_a3 = _get_line_segment_sphere_intersections(self.params['A_home']['A3'], self.params['A_end']['A3'], C3_w, self.params['R2'])
        intersections_a4 = _get_line_segment_sphere_intersections(self.params['A_home']['A4'], self.params['A_end']['A4'], C4_w, self.params['R2'])

        if not all([intersections_a1, intersections_a2, intersections_a3, intersections_a4]):
            self.last_solve_successful = False
            self.last_error_msg = "Pose is unreachable. At least one linkage point cannot connect to its actuator track."
            if self.verbosity >= 1: print(f"IK Failed: {self.last_error_msg}")
            return False
        
        A1_w, A2_w, A3_w, A4_w = intersections_a1[0], intersections_a2[0], intersections_a3[0], intersections_a4[0]
        
        if self.verbosity >= 2:
            print(f"IK Step 4: Actuator points found:\n A1:{np.round(A1_w,2)}, A2:{np.round(A2_w,2)}, A3:{np.round(A3_w,2)}, A4:{np.round(A4_w,2)}")

        try:
            # Store extensions as numpy array: [e1, e2, e3, e4]
            extensions = np.array([
                self._get_extension_from_attachment(A1_w, 'A1'),
                self._get_extension_from_attachment(A2_w, 'A2'),
                self._get_extension_from_attachment(A3_w, 'A3'),
                self._get_extension_from_attachment(A4_w, 'A4')
            ], dtype=np.float32)
        except ValueError as e:
            self.last_solve_successful = False
            self.last_error_msg = str(e)
            if self.verbosity >= 1: print(f"IK Failed: {self.last_error_msg}")
            return False
        if self.verbosity >= 2:
            print(f"IK Step 5: Extensions calculated and are valid: {extensions}")

        gpl = self.grab_points_local
        outer_grip_point_w = G1_w + gpl['outer_grip_point'][0] * approach_vec + gpl['outer_grip_point'][1] * tangential_vec + gpl['outer_grip_point'][2] * radial_vec
        inner_grip_point_1_w = G1_w + gpl['inner_grip_point_1'][0] * approach_vec + gpl['inner_grip_point_1'][1] * tangential_vec + gpl['inner_grip_point_1'][2] * radial_vec
        inner_grip_point_2_w = G1_w + gpl['inner_grip_point_2'][0] * approach_vec + gpl['inner_grip_point_2'][1] * tangential_vec + gpl['inner_grip_point_2'][2] * radial_vec
        gripper_back_w = G1_w + gpl['gripper_back'][0] * approach_vec + gpl['gripper_back'][1] * tangential_vec + gpl['gripper_back'][2] * radial_vec

        cam_params = self.params['camera_params']
        cam_offset = cam_params['local_pos_offset'].astype(np.float32)
        camera_pos_w = G1_w + cam_offset[0] * approach_vec + cam_offset[1] * tangential_vec + cam_offset[2] * radial_vec

        camera_view_w = self.cam_rot.apply(approach_vec).astype(np.float32)
        camera_up_w = self.cam_rot.apply(radial_vec).astype(np.float32)

        self.E1, self.G1 = E1_w, G1_w
        self.C1, self.C2, self.C3, self.C4 = C1_w, C2_w, C3_w, C4_w
        self.A1, self.A2, self.A3, self.A4 = A1_w, A2_w, A3_w, A4_w
        self.rx, self.ry, self.rz = rx_deg, 0.0, rz_deg
        self.approach_vec, self.tangential_vec, self.radial_vec = approach_vec, tangential_vec, radial_vec
        self.outer_grip_point, self.inner_grip_point_1, self.inner_grip_point_2, self.gripper_back = outer_grip_point_w, inner_grip_point_1_w, inner_grip_point_2_w, gripper_back_w
        self.camera_pos, self.camera_view_vector, self.camera_up_vector = camera_pos_w, camera_view_w, camera_up_w
        self.plane_normal = np.cross(C2_w - C1_w, C3_w - C1_w)
        self.plane_normal /= np.linalg.norm(self.plane_normal)
        self.extensions = extensions
        self.E1_quaternion = E1_quaternion
        
        # Calculate deltas if previous state was available
        if calculate_deltas:
            actuator_delta = self.extensions - prev_extensions
            self.delta_extensions[:] = actuator_delta
            self.delta_E1[:] = self.E1 - prev_e1
            self.delta_E1_quaternion[:] = self.E1_quaternion - prev_E1_quaternion
        
        self.last_solve_successful = True
        self.last_error_msg = "IK solve successful."
        if self.verbosity >= 1:
            print("--- IK successful: Robot state updated. ---")
        return True

    def go_home(self):
        """
        Set the robot to its configured home pose.
        
        Moves the robot to the home position defined in the configuration
        (E1_home_x, E1_home_y, E1_home_rx, E1_home_rz). If the move fails,
        delta arrays are zeroed.
        
        Returns:
            bool: True if successfully moved to home pose, False otherwise.
        """
        success = self.update_from_e1_pose(np.array([self.E1_home_x, self.E1_home_y, 0], dtype=np.float32), self.E1_home_rx, self.E1_home_rz)
        
        if not success:
            if self.verbosity >= 1:
                print(f"go_home failed: {self.last_error_msg}")

        else:
            self._zero_deltas()
        
        return success

    def move_E1(self, dx=0, dy=0, drx=0, drz=0):
        """
        Move the E1 end-effector pose incrementally from current position.
        
        This method attempts to move the end-effector by the specified deltas.
        If the move fails, all delta arrays are zeroed and the robot state
        remains unchanged.
        
        Args:
            dx (float): Translation in x-direction (mm).
            dy (float): Translation in y-direction (mm).
            drx (float): Rotation about x-axis (degrees).
            drz (float): Rotation about z-axis (degrees).
            
        Returns:
            tuple: (success_flag, reason, actuator_delta_array)
                - success_flag (bool): True if move succeeded.
                - reason (str): Description of outcome.
                - actuator_delta_array (np.ndarray or None): Change in actuator extensions.
        """
        # Calculate new target pose
        new_e1 = self.E1 + np.array([dx, dy, 0], dtype=np.float32)
        new_rx = self.rx + drx
        new_rz = self.rz + drz
        
        # Attempt to solve for new pose (deltas calculated automatically in update_from_e1_pose)
        success = self.update_from_e1_pose(new_e1, new_rx, new_rz)
        
        if not success:
            # Failed move: zero all deltas and maintain previous state
            self._zero_deltas()
            return False, self.last_error_msg, None
        
        return True, "Move successful", self.delta_extensions.copy()

    def set_random_e1_pose(self, min_difficulty=0, max_difficulty=1, max_attempts=30):
        """
        Set robot to a random E1 pose within workspace bounds and tilt limits.

        Generates random poses using circular radius around home position for XY,
        and independent random tilts. Difficulty is randomly chosen between min and max bounds.

        Args:
            min_difficulty (float, optional): Minimum difficulty level in [0, 1].
            max_difficulty (float, optional): Maximum difficulty level in [0, 1].
                - If both None: Each parameter gets completely random difficulty [0, 1]
                - 0: Pose at home position with no tilt
                - 1: Up to reachable_radius with full tilt range
            max_attempts (int): Maximum random samples before fallback to home.

        Returns:
            tuple: (success_flag, pose_dict, actual_difficulty)
                - success_flag (bool): True if valid pose was found
                - pose_dict (dict): Final pose {'x', 'y', 'rx', 'rz'}
                - actual_difficulty (float): Calculated difficulty of resulting pose
        """

        # Cache limits and home position
        rx_lim = self.params.get('tilt_rx_limit_deg', 0.0)
        rz_lim = self.params.get('tilt_rz_limit_deg', 0.0)
        x_home = self.E1_home_x
        y_home = self.E1_home_y
        
        # Use reachable radius from config
        max_radius = self.reachable_radius

        for _ in range(max_attempts):

            # Random difficulty for each parameter independently
            d_radius = np.random.rand() * (max_difficulty - min_difficulty) + min_difficulty
            d_rx = np.random.rand() * (max_difficulty - min_difficulty) + min_difficulty
            d_rz = np.random.rand() * (max_difficulty - min_difficulty) + min_difficulty

            # Generate random position in circular pattern around home
            if d_radius > 0:
                # Random angle and radius within circular bounds
                angle = np.random.uniform(0, 2 * np.pi)
                radius = d_radius * max_radius
                x = x_home + radius * np.cos(angle)
                y = y_home + radius * np.sin(angle)
            else:
                # Stay at home position
                x, y = x_home, y_home

            # Generate random tilts with random signs
            rx = np.random.choice([-1, 1]) * d_rx * rx_lim
            rz = np.random.choice([-1, 1]) * d_rz * rz_lim

            # Test pose validity
            e1_position = np.array([x, y, 0], dtype=np.float32)
            if self.update_from_e1_pose(e1_position, rx, rz):
                # Calculate actual difficulty of successful pose
                actual_difficulty = self._calculate_pose_difficulty_circular(
                    x, y, rx, rz, x_home, y_home, max_radius, rx_lim, rz_lim
                )
                pose = {'x': x, 'y': y, 'rx': rx, 'rz': rz}
                return True, pose, actual_difficulty

        # Fallback to home pose
        return self._fallback_to_home_pose(x_home, y_home)

    def _calculate_pose_difficulty_circular(self, x, y, rx, rz, x_home, y_home, max_radius, rx_lim, rz_lim):
        """Calculate normalized difficulty of a pose using circular distance and independent tilts."""
        # Calculate circular distance from home position
        distance = np.sqrt((x - x_home)**2 + (y - y_home)**2)
        d_radius_norm = distance / max_radius if max_radius > 0 else 0.0
        
        # Calculate normalized tilt difficulties
        d_rx_norm = abs(rx) / rx_lim if rx_lim > 0 else 0.0
        d_rz_norm = abs(rz) / rz_lim if rz_lim > 0 else 0.0
        
        # Average of the three independent difficulty components
        return (d_radius_norm + d_rx_norm + d_rz_norm) / 3.0

    def _fallback_to_home_pose(self, x_home, y_home):
        """Attempt to fall back to home pose if random generation fails."""
        e1_home = np.array([x_home, y_home, 0], dtype=np.float32)
        home_pose = {'x': x_home, 'y': y_home, 'rx': 0, 'rz': 0}
        
        if self.update_from_e1_pose(e1_home, 0, 0):
            return True, home_pose, 0.0
        else:
            # Even home pose failed
            return False, home_pose, 0.0

    def _get_extension_from_attachment(self, A_pos, actuator_key):
        """
        Calculate actuator extension value [0,1] from world position.
        
        Uses precomputed track data for efficiency. Extension represents the
        normalized distance along the actuator track from home to end position.
        
        Args:
            A_pos (np.ndarray): World position of actuator attachment point.
            actuator_key (str): Actuator identifier ('A1', 'A2', 'A3', 'A4').
            
        Returns:
            float: Extension value clamped to [0, 1] range.
            
        Side Effects:
            Sets last_solve_successful to False if extension is out of valid range.
        """
        home = self.params['A_home'][actuator_key]
        track_data = self.params['actuator_tracks'][actuator_key]
        total_dist = track_data['dist']
        
        if total_dist < 1e-9:
            return 0.0
        
        current_vec = A_pos - home
        total_vec = track_data['vec']
        extension_dist = np.dot(current_vec, total_vec / total_dist)
        extension = extension_dist / total_dist
        
        # Validate extension range with small tolerance
        if not (-1e-9 <= extension <= 1.0 + 1e-9):
            self.last_solve_successful = False
            self.last_error_msg = (
                f"Actuator {actuator_key} extension {extension:.4f} "
                f"is out of physical range [0, 1]. Value clipped."
            )
            if self.verbosity >= 1:
                print(f"WARNING: {self.last_error_msg}")
        
        return np.clip(extension, 0.0, 1.0)

    def __repr__(self):
        """Return formatted string representation of robot state."""
        if not self.last_solve_successful:
            return f"<RobotKinematics: Unsolved state. Reason: {self.last_error_msg}>"
        
        def format_vec(vec):
            """Format vector for display."""
            return "None" if vec is None else np.round(vec, 2)
        
        def format_extensions():
            """Format extensions array for display."""
            try:
                return (f"[e1: {self.extensions[0]:.3f}, e2: {self.extensions[1]:.3f}, "
                       f"e3: {self.extensions[2]:.3f}, e4: {self.extensions[3]:.3f}]")
            except (ValueError, TypeError, IndexError):
                return "Could not be formatted."
        
        sections = [
            "--- Robot Kinematics State ---",
            "\n--- Gripper Pose ---",
            f"Gripper Position (G1): {format_vec(self.G1)}",
            f"Approach Vector:       {format_vec(self.approach_vec)}",
            f"Tangential Vector:     {format_vec(self.tangential_vec)}",
            f"Radial Vector:         {format_vec(self.radial_vec)}",
            "\n--- Plate Orientation ---",
            f"Plane Normal: {format_vec(self.plane_normal)}",
            f"Angles (deg): rx: {self.rx:.2f}, ry: {self.ry:.2f}, rz: {self.rz:.2f}",
            "\n--- Actuator State ---",
            f"Current Extensions: {format_extensions()}",
            f"A1: {format_vec(self.A1)}, A2: {format_vec(self.A2)}",
            f"A3: {format_vec(self.A3)}, A4: {format_vec(self.A4)}",
            "\n--- End-Effector Points (World Coords) ---",
            f"E1: {format_vec(self.E1)}",
            f"C1: {format_vec(self.C1)}, C2: {format_vec(self.C2)}",
            f"C3: {format_vec(self.C3)}, C4: {format_vec(self.C4)}",
            "\n--- Grip Points (World Coords) ---",
            f"Outer Grip Point:   {format_vec(self.outer_grip_point)}",
            f"Inner Grip Point 1: {format_vec(self.inner_grip_point_1)}",
            f"Inner Grip Point 2: {format_vec(self.inner_grip_point_2)}",
            f"Gripper Back:       {format_vec(self.gripper_back)}",
            "\n--- Camera State ---",
            f"Camera Position: {format_vec(self.camera_pos)}",
            f"Camera View Vec: {format_vec(self.camera_view_vector)}",
            f"Camera Up Vec:   {format_vec(self.camera_up_vector)}",
            "\n" + "-" * 30
        ]
        
        return "\n".join(sections)

if __name__ == '__main__':
    """Test the robot kinematics system with basic operations."""
    
    robot = RobotKinematics(verbosity=0)

    print("--- Test Case 1: Go to home pose ---")
    home_success = robot.go_home()
    print(f"Go home success: {home_success}")

    print("\n--- Test Case 2: Move E1 incrementally ---")
    move_success, reason, actuator_delta = robot.move_E1(dx=10, dy=20, drx=5, drz=-5)
    print(f"Move success: {move_success}, reason: {reason}")
    if move_success:
        print(f"Actuator delta: {actuator_delta}")

    print("\n--- Test Case 3: Random pose generation ---")
    ring = None
    success, pose, difficulty = robot.set_random_e1_pose(min_difficulty=0.5, max_difficulty=1.0)
    if success:
        print(f"Random pose: {pose}, difficulty: {difficulty:.3f}")
        ring = robot.create_ring()
        if ring:
            grip_success = robot.evaluate_grip(ring)
            print(f"Grip evaluation: {grip_success}")
    else:
        print(f"Failed to find valid random pose. Attempted: {pose}")

    # Visualization if available
    try:
        if ring:
            visualize_system(robot, ring=ring)
            print(robot)
            print(ring)
        else:
            visualize_system(robot)
    except Exception as e:
        print(f"Visualization not available: {e}")
