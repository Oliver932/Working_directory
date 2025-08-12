import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrender
import trimesh
import time


class RingProjectorSimplified:
    """
    Simplified version of RingProjector that only uses the custom vectorized method
    and includes proper visibility checking with last visible state tracking.
    """

    def __init__(self, vertical_fov_deg=45.0, image_width=960, image_height=540):
        """
        Initializes the simplified RingProjector.

        Args:
            vertical_fov_deg (float): Camera vertical field of view in degrees
            image_width (int): Image width in pixels
            image_height (int): Image height in pixels
        """
        self.vertical_fov_deg = vertical_fov_deg
        self.image_width = image_width
        self.image_height = image_height
        self.aspect_ratio = image_width / image_height
        self.horizontal_fov_deg = 2 * np.degrees(
            np.arctan(np.tan(np.radians(vertical_fov_deg) / 2) * self.aspect_ratio)
        )
        
        self.projected_properties = {}
        self.last_visible_properties = None  # Track last visible state
        self.last_g1_position_2d = None  # Track last G1 projection

        # Pre-calculate constant values for point generation and projection
        self.num_ring_points = 20
        theta = np.linspace(0, 2 * np.pi, self.num_ring_points)
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)
        self.tan_v_fov_half = np.tan(np.deg2rad(self.vertical_fov_deg) / 2)
        self.tan_h_fov_half = np.tan(np.deg2rad(self.horizontal_fov_deg) / 2)

        # Pre-calculate half screen dimensions for visibility checking
        self.half_width = image_width / 2
        self.half_height = image_height / 2

        # Initialize internal state variables
        self.ring_position = None
        self.ring_radius = None
        self.ring_normal = None
        self.ring_radial = None
        self.ring_tangential = None
        self.camera_position = None
        self.camera_view_dir = None
        self.camera_up_dir = None
        self.cam_z_axis = None
        self.cam_right = None
        self.cam_up_true = None

    def _project_point_to_camera_plane(self, point_3d):
        """
        Projects a single 3D point to normalized camera coordinates.
        
        Args:
            point_3d: 3D point to project
            
        Returns:
            tuple: (position_2d, in_front) where position_2d is normalized coords and in_front is bool
        """
        vec_to_point = point_3d - self.camera_position
        dist_forward = np.dot(vec_to_point, self.camera_view_dir)
        
        # Check if point is in front of camera
        if dist_forward <= 0:
            return np.array([0, 0], dtype=np.float32), False
            
        # Project to view space
        x_view = np.dot(vec_to_point, self.cam_right)
        y_view = np.dot(vec_to_point, self.cam_up_true)
        
        # Project to normalized coordinates
        inv_dist = 1.0 / dist_forward
        x_norm = (x_view * inv_dist) / (2 * self.tan_h_fov_half)
        y_norm = (y_view * inv_dist) / (2 * self.tan_v_fov_half)
        
        position_2d = np.array([x_norm, y_norm], dtype=np.float32)
        return position_2d, True

    def _normalize(self, vector):
        """Normalizes a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def update(self, robot=None, ring=None):
        """
        Updates the state of the projector from robot and ring objects and recalculates the ellipse properties.
        
        Args:
            robot: Object with camera_pos, camera_view_vector, camera_up_vector attributes
            ring: Object with position, outer_radius, approach_vec, radial_vec, tangential_vec attributes
        """
        if robot is not None and ring is not None:
            self.ring_position = np.array(ring.position, dtype=float)
            self.ring_radius = float(ring.outer_radius)
            self.ring_normal = np.array(ring.approach_vec, dtype=float)
            self.ring_radial = np.array(ring.radial_vec, dtype=float)
            self.ring_tangential = np.array(ring.tangential_vec, dtype=float)
            self.camera_position = np.array(robot.camera_pos, dtype=float)
            self.camera_view_dir = np.array(robot.camera_view_vector, dtype=float)
            self.camera_up_dir = np.array(robot.camera_up_vector, dtype=float)
            # Store G1 position from robot kinematics
            self.g1_position = np.array(robot.G1, dtype=float)

            # Pre-calculate camera basis vectors for performance
            self.cam_z_axis = -self.camera_view_dir
            self.cam_right = self._normalize(np.cross(self.camera_up_dir, self.cam_z_axis))
            self.cam_up_true = np.cross(self.cam_z_axis, self.cam_right)  # Re-orthogonalize

        # Only calculate if we have valid state
        if self.ring_position is not None:
            self.projected_properties = self._calculate_projection_custom_vectorized()

    def set_state(self, ring_position, ring_radius, ring_normal, ring_radial, ring_tangential,
                  camera_position, camera_view_dir, camera_up_dir, g1_position):
        """
        Directly set the internal state for testing purposes.
        """
        self.ring_position = np.array(ring_position, dtype=float)
        self.ring_radius = float(ring_radius)
        self.ring_normal = np.array(ring_normal, dtype=float)
        self.ring_radial = np.array(ring_radial, dtype=float)
        self.ring_tangential = np.array(ring_tangential, dtype=float)
        self.camera_position = np.array(camera_position, dtype=float)
        self.camera_view_dir = np.array(camera_view_dir, dtype=float)
        self.camera_up_dir = np.array(camera_up_dir, dtype=float)
        self.g1_position = np.array(g1_position, dtype=float)

        # Pre-calculate camera basis vectors for performance
        self.cam_z_axis = -self.camera_view_dir
        self.cam_right = self._normalize(np.cross(self.camera_up_dir, self.cam_z_axis))
        self.cam_up_true = np.cross(self.cam_z_axis, self.cam_right)  # Re-orthogonalize
        
        # Calculate projection
        self.projected_properties = self._calculate_projection_custom_vectorized()

    def _calculate_projection_custom_vectorized(self):
        """
        Projects points from the ring's centerline and fits an ellipse
        using vectorized NumPy operations for high performance.
        Only calculates ellipse properties if the ring is visible.
        """
        # --- Use pre-cached attributes (local variables for faster access) ---
        u, v = self.ring_radial, self.ring_tangential
        ring_pos, ring_rad = self.ring_position, self.ring_radius
        cos_th, sin_th = self.cos_theta, self.sin_theta
        cam_pos, view_dir = self.camera_position, self.camera_view_dir
        cam_right, cam_up = self.cam_right, self.cam_up_true
        tan_h_fov, tan_v_fov = self.tan_h_fov_half, self.tan_v_fov_half
        img_w, img_h = self.image_width, self.image_height
        
        # Pre-define the "not visible" return value to avoid duplication
        not_visible_result = {
            "visible": False,
            "position_2d": np.array([0, 0], dtype=np.float32),
            "delta_position_2d": np.array([0, 0], dtype=np.float32),
            "major_axis_norm": np.float32(0),
            "aspect_ratio": np.float32(1),
            "delta_major_axis_norm": np.float32(0),
            "delta_aspect_ratio": np.float32(0),
            "orientation_2d": np.array([1, 0], dtype=np.float32),
            "delta_orientation_2d": np.array([0, 0], dtype=np.float32),
            "g1_position_2d": np.array([0, 0], dtype=np.float32),
            "delta_g1_position_2d": np.array([0, 0], dtype=np.float32)
        }
        
        # --- Always project G1 position regardless of ring visibility ---
        g1_position_2d, g1_in_front = self._project_point_to_camera_plane(self.g1_position)
        
        # Calculate G1 delta from last projection
        if self.last_g1_position_2d is not None:
            delta_g1_position_2d = g1_position_2d - self.last_g1_position_2d
        else:
            delta_g1_position_2d = np.array([0, 0], dtype=np.float32)
        
        # Update not_visible_result with G1 data
        not_visible_result["g1_position_2d"] = g1_position_2d
        not_visible_result["delta_g1_position_2d"] = delta_g1_position_2d

        # --- 1. Generate 3D points on the ring ---
        points_3d = (ring_pos +
                     ring_rad * (np.outer(cos_th, u) +
                                 np.outer(sin_th, v)))

        # --- 2. Project all 3D points to 2D in a single vectorized operation ---
        vecs_to_points = points_3d - cam_pos
        dist_forward = vecs_to_points.dot(view_dir)

        # Check if points are in front of camera
        in_front_mask = dist_forward > 0
        if np.count_nonzero(in_front_mask) < 5:
            self.last_g1_position_2d = g1_position_2d  # Update G1 tracking
            return not_visible_result

        vecs_to_points_in_front = vecs_to_points[in_front_mask]
        dist_forward_in_front = dist_forward[in_front_mask]

        # Project to view space and pixel coordinates in one step
        inv_dist = 1.0 / dist_forward_in_front
        x_view = vecs_to_points_in_front.dot(cam_right)
        y_view = vecs_to_points_in_front.dot(cam_up)
        
        x_pixel = (x_view * inv_dist) / (2 * tan_h_fov) * img_w + self.half_width
        y_pixel = -(y_view * inv_dist) / (2 * tan_v_fov) * img_h + self.half_height
        
        # Check visibility in single operation
        visible_mask = ((x_pixel >= 0) & (x_pixel <= img_w) & 
                       (y_pixel >= 0) & (y_pixel <= img_h))
        if not np.any(visible_mask):
            self.last_g1_position_2d = g1_position_2d  # Update G1 tracking
            return not_visible_result
        
        # --- 3. Fit Ellipse (only if visible) ---
        projected_points_px = np.stack((x_pixel, y_pixel), axis=1).astype(np.int64)
        (xc_px, yc_px), (d1_px, d2_px), angle_deg = cv2.fitEllipse(projected_points_px)

        # --- 4. Convert results to normalized coordinates and new format ---
        # Convert center and calculate parameters in one step
        position_2d = np.array([
            (xc_px / img_w) - 0.5,
            -((yc_px / img_h) - 0.5)  # Flip Y to make it point up
        ], dtype=np.float32)

        # Determine major/minor axes and angle (optimized)
        if d1_px >= d2_px:
            major_axis_norm = np.float32(d1_px / (2 * img_h))
            aspect_ratio = np.float32(d2_px / d1_px) if d1_px > 0 else np.float32(1)
            angle_rad = np.radians(angle_deg)
        else:
            major_axis_norm = np.float32(d2_px / (2 * img_h))
            aspect_ratio = np.float32(d1_px / d2_px) if d2_px > 0 else np.float32(1)
            angle_rad = np.radians(angle_deg + 90)

        # Calculate orientation vector directly
        orientation_2d = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)

        # --- 5. Compute deltas from last visible state ---
        if self.last_visible_properties is not None:
            delta_position_2d = position_2d - self.last_visible_properties["position_2d"]
            delta_major_axis_norm = major_axis_norm - self.last_visible_properties["major_axis_norm"]
            delta_aspect_ratio = aspect_ratio - self.last_visible_properties["aspect_ratio"]
            delta_orientation_2d = orientation_2d - self.last_visible_properties["orientation_2d"]
        else:
            delta_position_2d = np.array([0, 0], dtype=np.float32)
            delta_major_axis_norm = np.float32(0)
            delta_aspect_ratio = np.float32(0)
            delta_orientation_2d = np.array([0, 0], dtype=np.float32)

        # Store current state as last visible
        current_properties = {
            "visible": True,
            "position_2d": position_2d,
            "delta_position_2d": delta_position_2d,
            "major_axis_norm": major_axis_norm,
            "aspect_ratio": aspect_ratio,
            "delta_major_axis_norm": delta_major_axis_norm,
            "delta_aspect_ratio": delta_aspect_ratio,
            "orientation_2d": orientation_2d,
            "delta_orientation_2d": delta_orientation_2d,
            "g1_position_2d": g1_position_2d,
            "delta_g1_position_2d": delta_g1_position_2d
        }
        
        self.last_visible_properties = current_properties.copy()
        self.last_g1_position_2d = g1_position_2d  # Update last G1 position
        
        return current_properties


def run_and_plot_case(ax, title, case_params, common_params, width, height, projector, plot_renderer, scene, mesh_node, camera_node):
    """
    Renders a single validation case and plots it on the given subplot axis.
    Reuses renderer components for performance.
    """
    # --- 1. Calculate ring basis vectors for the test case ---
    ring_normal = np.array(case_params['ring_normal'], dtype=float)
    if np.allclose(ring_normal, [0, 0, 1]) or np.allclose(ring_normal, [0, 0, -1]):
        ring_radial = np.array([1, 0, 0])
    else:
        ring_radial = projector._normalize(np.cross(ring_normal, [0, 0, 1]))
    ring_tangential = projector._normalize(np.cross(ring_normal, ring_radial))

    # --- 2. Set projector state directly ---
    # For testing, assume G1 is offset from ring position along approach vector
    g1_test_position = np.array(case_params['ring_position']) - 15.0 * ring_normal
    
    projector.set_state(
        ring_position=case_params['ring_position'],
        ring_radius=common_params.get('ring_radius', 10.0),
        ring_normal=ring_normal,
        ring_radial=ring_radial,
        ring_tangential=ring_tangential,
        camera_position=common_params['camera_position'],
        camera_view_dir=common_params['camera_view_dir'],
        camera_up_dir=common_params['camera_up_dir'],
        g1_position=g1_test_position
    )

    print(f"\n--- Running Test Case: {title} ---")
    
    # --- 3. Update poses in the existing scene for ground-truth rendering ---
    ring_transform = trimesh.geometry.align_vectors([0, 0, 1], ring_normal)
    ring_transform[:3, 3] = case_params['ring_position']
    scene.set_pose(mesh_node, pose=ring_transform)

    z_axis = -np.array(common_params['camera_view_dir'])
    x_axis = projector._normalize(np.cross(common_params['camera_up_dir'], z_axis))
    y_axis = np.cross(z_axis, x_axis)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    camera_pose[:3, 3] = common_params['camera_position']
    scene.set_pose(camera_node, pose=camera_pose)

    color_img, _ = plot_renderer.render(scene)

    # --- 4. Get stats from the simplified projector ---
    stats = projector.projected_properties

    # --- 5. Print properties ---
    print(f"{'Parameter':<20} | {'Value':<20}")
    print("-" * 45)
    print(f"{'Visible':<20} | {stats.get('visible', False)}")
    
    if stats.get("visible"):
        # Calculate derived properties for printing
        major_axis = stats['major_axis_norm'] * 2
        minor_axis = major_axis * stats['aspect_ratio']
        angle = np.degrees(np.arctan2(stats['orientation_2d'][1], stats['orientation_2d'][0]))

        print(f"{'Position X (norm)':<20} | {stats['position_2d'][0]:<20.4f}")
        print(f"{'Position Y (norm)':<20} | {stats['position_2d'][1]:<20.4f}")
        print(f"{'Major Axis (norm)':<20} | {major_axis:<20.4f}")
        print(f"{'Minor Axis (norm)':<20} | {minor_axis:<20.4f}")
        print(f"{'Aspect Ratio':<20} | {stats['aspect_ratio']:<20.4f}")
        print(f"{'Orientation (deg)':<20} | {angle:<20.2f}")
    
    # Always print G1 information regardless of visibility
    print(f"{'G1 Position X (norm)':<20} | {stats['g1_position_2d'][0]:<20.4f}")
    print(f"{'G1 Position Y (norm)':<20} | {stats['g1_position_2d'][1]:<20.4f}")
    print(f"{'G1 Delta X (norm)':<20} | {stats['delta_g1_position_2d'][0]:<20.4f}")
    print(f"{'G1 Delta Y (norm)':<20} | {stats['delta_g1_position_2d'][1]:<20.4f}")
    print("-" * 45)

    # --- 6. Draw overlay ---
    overlay_img = color_img.copy()
    
    def draw_ellipse(img, stats, color):
        if not stats.get("visible"): 
            return
        # Convert normalized center back to pixel coordinates only for drawing
        center_x_px = int((stats['position_2d'][0] + 0.5) * width)
        center_y_px = int((-stats['position_2d'][1] + 0.5) * height)
        center_px = (center_x_px, center_y_px)
        
        # Calculate ellipse dimensions from new parameters
        major_axis_norm = stats['major_axis_norm']
        aspect_ratio = stats['aspect_ratio']
        
        major_axis_px = int(major_axis_norm * height * 2)
        minor_axis_px = int(major_axis_norm * aspect_ratio * height * 2)
        
        angle_deg = np.degrees(np.arctan2(stats['orientation_2d'][1], stats['orientation_2d'][0]))

        cv2.ellipse(img, center_px, (major_axis_px // 2, minor_axis_px // 2),
                    angle_deg, 0, 360, color, 3)
        x0, y0 = width // 2, height // 2
        cv2.arrowedLine(img, (x0, y0), center_px, color, 2, tipLength=0.05)

    draw_ellipse(overlay_img, stats, (0, 255, 0))  # Green for simplified method
    
    # --- 7. Plot on the subplot ---
    ax.imshow(overlay_img)
    title_with_visibility = f"{title}\n(Visible: {stats.get('visible', False)})"
    ax.set_title(title_with_visibility)
    ax.axis('off')


# --- Run Visualization ---
if __name__ == '__main__':
    # --- General camera and image settings ---
    IMG_WIDTH = 960
    IMG_HEIGHT = 540
    
    common_params = {
        "ring_radius": 10.0,
        "camera_position": [0.0, 0.0, 0.0],
        "camera_view_dir": [0.0, 0.0, 1.0],
        "camera_up_dir": [0.0, 1.0, 0.0],
        "camera_fov_deg": 30.0,
    }

    test_cases = {
        "Tilted & Off-Center (Fully Visible)": {
            "ring_position": [10.0, 5.0, 60.0],
            "ring_normal": np.array([0.4, 0.6, np.sqrt(1 - 0.4**2 - 0.6**2)])
        },
        "Very Off-Screen": {
            "ring_position": [33.0, 22.0, 60.0],
            "ring_normal": np.array([0.4, 0.6, np.sqrt(1 - 0.4**2 - 0.6**2)])
        },
        "Slightly Off-Screen": {
            "ring_position": [20.0, 10.0, 60.0],
            "ring_normal": np.array([0.4, 0.6, np.sqrt(1 - 0.4**2 - 0.6**2)])
        },
        "Almost Edge-On": {
            "ring_position": [0.0, 0.0, 50.0],
            "ring_normal": np.array([np.sin(np.deg2rad(88)), 0.0, np.cos(np.deg2rad(88))])
        }
    }

    # --- Create plot and run cases ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()

    projector = RingProjectorSimplified(vertical_fov_deg=common_params['camera_fov_deg'], image_width=IMG_WIDTH, image_height=IMG_HEIGHT)

    # --- Pre-build rendering components for performance ---
    plot_renderer = pyrender.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
    # Create and add the mesh
    torus_mesh = trimesh.creation.torus(major_radius=common_params['ring_radius'], minor_radius=0.2, major_segments=128, minor_segments=32)
    mesh = pyrender.Mesh.from_trimesh(torus_mesh, smooth=False)
    mesh_node = scene.add(mesh)
    # Create and add the camera
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(common_params['camera_fov_deg']), aspectRatio=(IMG_WIDTH/IMG_HEIGHT))
    camera_node = scene.add(camera)

    for i, (title, params) in enumerate(test_cases.items()):
        if i < len(axes):
            run_and_plot_case(axes[i], title, params, common_params, IMG_WIDTH, IMG_HEIGHT, projector, plot_renderer, scene, mesh_node, camera_node)

    # Clean up the persistent renderer
    plot_renderer.delete()

    fig.legend(handles=[plt.Line2D([0], [0], color='lime', lw=4, label='Simplified Method')],
               loc='lower center', ncol=1, fontsize=14)
    plt.suptitle("Simplified Ring Projector - Visibility & Ellipse Fitting", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()

    # --- Performance Benchmark ---
    projector = RingProjectorSimplified(vertical_fov_deg=45.0, image_width=960, image_height=540)
    num_iterations = 100000

    print("--- Running Performance Benchmark (Simplified) ---")
    
    # Set up a baseline state
    projector.set_state(
        ring_position=[0.0, 0.0, 50.0],
        ring_radius=10.0,
        ring_normal=[0.0, 0.0, 1.0],
        ring_radial=[1.0, 0.0, 0.0],
        ring_tangential=[0.0, 1.0, 0.0],
        camera_position=[0.0, 0.0, 0.0],
        camera_view_dir=[0.0, 0.0, 1.0],
        camera_up_dir=[0.0, 1.0, 0.0],
        g1_position=[0.0, 0.0, 35.0]  # G1 positioned behind ring
    )
    
    start_time = time.perf_counter()
    for i in range(num_iterations):
        # Simulate random camera poses
        camera_pos = np.random.rand(3) * 10
        camera_view_dir = np.random.rand(3) - 0.5
        camera_view_dir = camera_view_dir / np.linalg.norm(camera_view_dir)
        
        # Create orthogonal up vector
        temp_vec = np.random.rand(3) - 0.5
        if np.abs(np.dot(camera_view_dir, temp_vec / np.linalg.norm(temp_vec))) > 0.99:
            temp_vec = np.array([1.0, 0.0, 0.0])
        right_vec = np.cross(camera_view_dir, temp_vec)
        camera_up_dir = np.cross(right_vec, camera_view_dir)
        camera_up_dir = camera_up_dir / np.linalg.norm(camera_up_dir)
        
        # Update projector state
        projector.set_state(
            ring_position=[0.0, 0.0, 50.0],
            ring_radius=10.0,
            ring_normal=[0.0, 0.0, 1.0],
            ring_radial=[1.0, 0.0, 0.0],
            ring_tangential=[0.0, 1.0, 0.0],
            camera_position=camera_pos,
            camera_view_dir=camera_view_dir,
            camera_up_dir=camera_up_dir,
            g1_position=[0.0, 0.0, 35.0]  # G1 positioned behind ring
        )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    fps = num_iterations / total_time

    print(f"Completed {num_iterations} iterations in {total_time:.2f} seconds.")
    print(f"Average time per update: {avg_time_ms:.4f} ms")
    print(f"Equivalent FPS (Simplified): {fps:.2f}")
