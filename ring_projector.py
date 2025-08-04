import numpy as np
import os
import time

# The line below is commented out to avoid EGL errors on systems without it.
# This will make pyrender use the default, on-screen rendering backend.
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from pyrender.constants import GLTF

# Mock classes for standalone execution
class RobotKinematics:
    def __init__(self, verbosity=0):
        self.camera_pos = np.array([0.0, 0.0, 0.0])
        self.camera_view_vector = np.array([0.0, 0.0, 1.0])
        self.camera_up_vector = np.array([0.0, 1.0, 0.0])
        self.verbosity = verbosity

    def create_ring(self):
        return Ring()

    def go_home(self):
        if self.verbosity > 0:
            print("Robot going to home position.")

    def set_random_e1_pose(self):
        if self.verbosity > 0:
            print("Setting random E1 pose.")
        # Simulate a successful pose change
        self.camera_pos = np.random.rand(3) * 10
        self.camera_view_vector = self._normalize(np.random.rand(3) - 0.5)
        # Create an orthogonal up vector
        temp_vec = np.random.rand(3) - 0.5
        if np.abs(np.dot(self.camera_view_vector, self._normalize(temp_vec))) > 0.99:
            temp_vec = np.array([0.0, 1.0, 0.0]) # A safe fallback
        
        right_vec = np.cross(self.camera_view_vector, temp_vec)
        self.camera_up_vector = self._normalize(np.cross(right_vec, self.camera_view_vector))
        return True, None, None
    
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

class Ring:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 50.0])
        self.approach_vec = np.array([0.0, 0.0, 1.0]) # Normal vector
        self.radial_vec = np.array([1.0, 0.0, 0.0])   # u-vector in the ring's plane
        self.tangential_vec = np.array([0.0, 1.0, 0.0]) # v-vector in the ring's plane
        self.outer_radius = 10.0
        self.middle_radius = 10.0
        self.inner_radius = 9.8


class RingProjector:
    """
    Calculates the 2D projected properties of a 3D ring using a perspective camera model.
    This version uses a numerical method to project multiple points for higher accuracy.
    The output coordinates are normalized to the range [-0.5, 0.5] with Y pointing up.
    Now initialized with a robot and ring instance.
    """

    def __init__(self, robot, ring, vertical_fov_deg=45.0, image_width=960, image_height=540, method='custom'):
        """
        Initializes the RingProjector with robot and ring objects.

        Args:
            robot: RobotKinematics instance
            ring: Ring instance
            vertical_fov_deg (float): Camera vertical field of view in degrees
            image_width (int): Image width in pixels
            image_height (int): Image height in pixels
            method (str): The calculation method to use ('custom' or 'render').
        """
        self.robot = robot
        self.ring = ring
        self.vertical_fov_deg = vertical_fov_deg

        self.image_width = image_width
        self.image_height = image_height
        self.aspect_ratio = image_width / image_height
        self.horizontal_fov_deg = 2 * np.degrees(
            np.arctan(np.tan(np.radians(vertical_fov_deg) / 2) * self.aspect_ratio)
        )
        self.method = method
        self.projected_properties = {}

        # Pre-calculate constant values for point generation and projection
        self.num_ring_points = 25
        theta = np.linspace(0, 2 * np.pi, self.num_ring_points)
        self.cos_theta = np.cos(theta)
        self.sin_theta = np.sin(theta)
        self.tan_v_fov_half = np.tan(np.deg2rad(self.vertical_fov_deg) / 2)
        self.tan_h_fov_half = np.tan(np.deg2rad(self.horizontal_fov_deg) / 2)

        # Create persistent renderer. This is a key efficiency gain.
        self.scale_factor = 0.05
        self._render_width = int(image_width * self.scale_factor)
        self._render_height = int(image_height * self.scale_factor)

        self._renderer = pyrender.OffscreenRenderer(self._render_width, self._render_height)
        self._cached_scene = None
        self._cached_mesh = None
        self._mesh_node = None
        self._camera_node = None
        self.update()


    def load_meshes_and_scene(self):
        """
        Loads and caches the mesh and creates the pyrender scene.
        Instead of a thick torus, it now creates a thin ring of vertices to match the 'custom' method.
        """
        if self.method != 'render':
            return

        # Create the mesh only once and cache it
        if getattr(self, '_cached_mesh', None) is None:
            num_points = 64 # Number of points to define the ring
            theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            
            points = np.zeros((num_points, 3))
            points[:, 0] = self.ring_radius * np.cos(theta)
            points[:, 1] = self.ring_radius * np.sin(theta)

            scaled_points = points * self.scale_factor
            
            primitive = pyrender.Primitive(positions=scaled_points, mode=GLTF.POINTS)
            self._cached_mesh = pyrender.Mesh([primitive])

        # Create the scene, camera, and lights only once
        if getattr(self, '_cached_scene', None) is None:
            scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
            
            ring_transform = trimesh.geometry.align_vectors([0, 0, 1], self.ring_normal)
            ring_transform[:3, 3] = self.ring_position * self.scale_factor
            self._mesh_node = scene.add(self._cached_mesh, pose=ring_transform)
            
            camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(self.vertical_fov_deg), aspectRatio=self.aspect_ratio)
            z_axis = -self.camera_view_dir
            x_axis = self._normalize(np.cross(self.camera_up_dir, z_axis))
            y_axis = np.cross(z_axis, x_axis)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
            camera_pose[:3, 3] = self.camera_position * self.scale_factor
            self._camera_node = scene.add(camera, pose=camera_pose)
            self._cached_scene = scene
        else:
            ring_transform = trimesh.geometry.align_vectors([0, 0, 1], self.ring_normal)
            ring_transform[:3, 3] = self.ring_position * self.scale_factor
            self._cached_scene.set_pose(self._mesh_node, pose=ring_transform)
            
            z_axis = -self.camera_view_dir
            x_axis = self._normalize(np.cross(self.camera_up_dir, z_axis))
            y_axis = np.cross(z_axis, x_axis)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
            camera_pose[:3, 3] = self.camera_position * self.scale_factor
            self._cached_scene.set_pose(self._camera_node, pose=camera_pose)

    @staticmethod
    def normalize_point(x, y, width, height):
        """
        Normalize pixel coordinates (x, y) to [-0.5, 0.5] range with Y-up.
        """
        return (x / width) - 0.5, -((y / height) - 0.5)

    @staticmethod
    def denormalize_point(x_norm, y_norm, width, height):
        """
        Convert normalized coordinates (x_norm, y_norm) in [-0.5, 0.5] to pixel coordinates (Y-down).
        """
        x = (x_norm + 0.5) * width
        y = (-y_norm + 0.5) * height
        return int(x), int(y)

    def _normalize(self, vector):
        """Normalizes a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def update(self):
        """
        Updates the state of the projector from the robot and ring objects and recalculates the ellipse properties.
        """
        self.ring_position = np.array(self.ring.position, dtype=float)
        self.ring_radius = float(self.ring.outer_radius)
        self.ring_normal = np.array(self.ring.approach_vec, dtype=float)
        self.ring_radial = np.array(self.ring.radial_vec, dtype=float)
        self.ring_tangential = np.array(self.ring.tangential_vec, dtype=float)
        self.camera_position = np.array(self.robot.camera_pos, dtype=float)
        self.camera_view_dir = np.array(self.robot.camera_view_vector, dtype=float)
        self.camera_up_dir = np.array(self.robot.camera_up_vector, dtype=float)

        # Pre-calculate camera basis vectors for performance
        self.cam_z_axis = -self.camera_view_dir
        self.cam_right = self._normalize(np.cross(self.camera_up_dir, self.cam_z_axis))
        self.cam_up_true = np.cross(self.cam_z_axis, self.cam_right) # Re-orthogonalize

        self._recalculate()

    def _recalculate(self):
        """
        Helper method to trigger the appropriate calculation method for projection.
        """
        if self.method == 'custom':
            self.projected_properties = self._calculate_projection_custom_vectorized()
        elif self.method == 'render':
            self.load_meshes_and_scene()
            self.projected_properties = self._calculate_projection_from_render()

    def _calculate_projection_custom_vectorized(self):
        """
        Projects points from the ring's centerline and fits an ellipse
        using vectorized NumPy operations for high performance.
        """
        # --- Use pre-cached attributes from self ---
        u, v = self.ring_radial, self.ring_tangential
        ring_pos, ring_rad = self.ring_position, self.ring_radius
        cos_th, sin_th = self.cos_theta, self.sin_theta
        cam_pos, view_dir = self.camera_position, self.camera_view_dir
        cam_right, cam_up = self.cam_right, self.cam_up_true
        tan_h_fov, tan_v_fov = self.tan_h_fov_half, self.tan_v_fov_half
        img_w, img_h = self.image_width, self.image_height

        # --- 1. Generate 3D points on the ring ---
        points_3d = (ring_pos +
                     ring_rad * (np.outer(cos_th, u) +
                                 np.outer(sin_th, v)))

        # --- 2. Project all 3D points to 2D in a single vectorized operation ---
        vecs_to_points = points_3d - cam_pos
        dist_forward = vecs_to_points.dot(view_dir)

        in_front_mask = dist_forward > 0
        if np.count_nonzero(in_front_mask) < 5:
            # RL-friendly: return last valid observation with zero deltas if available
            last_obs = self.projected_properties.copy() if self.projected_properties and self.projected_properties.get("calculable", False) else None
            if last_obs:
                last_obs["delta_center_2d"] = np.zeros(2, dtype=np.float32)
                last_obs["delta_semi_major_vector"] = np.zeros(2, dtype=np.float32)
                last_obs["delta_semi_minor_vector"] = np.zeros(2, dtype=np.float32)
                return last_obs
            # Fallback to zeros if no valid observation yet
            return {
                "visible": False,
                "calculable": False,
                "center_2d": np.array([0, 0], dtype=np.float32),
                "delta_center_2d": np.array([0, 0], dtype=np.float32),
                "semi_major_vector": np.array([0, 0], dtype=np.float32),
                "semi_minor_vector": np.array([0, 0], dtype=np.float32),
                "delta_semi_major_vector": np.array([0, 0], dtype=np.float32),
                "delta_semi_minor_vector": np.array([0, 0], dtype=np.float32)
            }

        vecs_to_points_in_front = vecs_to_points[in_front_mask]
        dist_forward_in_front = dist_forward[in_front_mask]

        # Project to view space
        x_view = vecs_to_points_in_front.dot(cam_right)
        y_view = vecs_to_points_in_front.dot(cam_up)

        # Project to normalized screen space
        # This is faster than two separate divisions
        inv_dist = 1.0 / dist_forward_in_front
        x_proj = (x_view * inv_dist) / (2 * tan_h_fov)
        y_proj = (y_view * inv_dist) / (2 * tan_v_fov)
        
        projected_points_2d = np.stack((x_proj, y_proj), axis=1)

        # --- 3. Fit Ellipse ---
        points_for_fit_x = (projected_points_2d[:, 0] + 0.5) * img_w
        points_for_fit_y = (-projected_points_2d[:, 1] + 0.5) * img_h
        points_for_fit = np.stack((points_for_fit_x, points_for_fit_y), axis=1).astype(np.int64)
        
        (xc_px, yc_px), (d1_px, d2_px), angle_deg = cv2.fitEllipse(points_for_fit)

        # --- 4. Normalize results consistently ---
        center_x_proj, center_y_proj = self.normalize_point(xc_px, yc_px, img_w, img_h)

        # Check if any of the actual projected ring points are visible on screen
        # projected_points_2d is already in normalized coordinates [-0.5, 0.5]
        x_coords = projected_points_2d[:, 0]
        y_coords = projected_points_2d[:, 1]
        visible_mask = ((x_coords >= -0.5) & (x_coords <= 0.5) & 
                       (y_coords >= -0.5) & (y_coords <= 0.5))
        center_visible = np.any(visible_mask)

        # OpenCV's angle corresponds to d1_px, but we need to ensure it corresponds to the major axis
        if d1_px >= d2_px:
            # d1 is major axis, angle is correct
            semi_major_length = np.float32(d1_px / (2 * img_h))
            semi_minor_length = np.float32(d2_px / (2 * img_h))
            major_angle_rad = np.radians(angle_deg)
        else:
            # d2 is major axis, need to rotate angle by 90 degrees
            semi_major_length = np.float32(d2_px / (2 * img_h))
            semi_minor_length = np.float32(d1_px / (2 * img_h))
            major_angle_rad = np.radians(angle_deg + 90)

        semi_major_vector = np.array([
            semi_major_length * np.cos(major_angle_rad),
            semi_major_length * np.sin(major_angle_rad)
        ], dtype=np.float32)

        semi_minor_vector = np.array([
            -semi_minor_length * np.sin(major_angle_rad),
             semi_minor_length * np.cos(major_angle_rad)
        ], dtype=np.float32)

        # --- 5. Compute deltas ---
        delta_center_2d = center_x_proj - self.projected_properties.get("center_2d", np.array([0, 0], dtype=np.float32))
        delta_semi_major_vector = semi_major_vector - self.projected_properties.get("semi_major_vector", np.array([0, 0], dtype=np.float32))
        delta_semi_minor_vector = semi_minor_vector - self.projected_properties.get("semi_minor_vector", np.array([0, 0], dtype=np.float32))

        center_2d = np.array([center_x_proj, center_y_proj], dtype=np.float32)

        return {
            "visible": center_visible,
            "calculable": True,
            "center_2d": center_2d,
            "delta_center_2d": delta_center_2d,
            "semi_major_vector": semi_major_vector,
            "semi_minor_vector": semi_minor_vector,
            "delta_semi_major_vector": delta_semi_major_vector,
            "delta_semi_minor_vector": delta_semi_minor_vector
        }


    def _calculate_projection_from_render(self):
        """
        Calculates projection by rendering the scene and fitting an ellipse to the rendered points.
        """
        width = self._render_width
        height = self._render_height

        scene = getattr(self, '_cached_scene', None)
        if scene is None:
            raise RuntimeError("Scene not loaded. Call load_meshes_and_scene() before rendering.")

        color_img, _ = self._renderer.render(scene)

        image_gray = (rgb2gray(color_img) * 255).astype(np.uint8)
        points_to_fit = np.argwhere(image_gray > 10) # Get (row, col) of all white pixels
        
        if len(points_to_fit) < 5:
            # RL-friendly: return last valid observation with zero deltas if available
            last_obs = self.projected_properties.copy() if self.projected_properties and self.projected_properties.get("calculable", False) else None
            if last_obs:
                last_obs["delta_center_2d"] = np.zeros(2, dtype=np.float32)
                last_obs["delta_semi_major_vector"] = np.zeros(2, dtype=np.float32)
                last_obs["delta_semi_minor_vector"] = np.zeros(2, dtype=np.float32)
                return last_obs
            # Fallback to zeros if no valid observation yet
            return {
                "visible": False,
                "calculable": False,
                "center_2d": np.array([0, 0], dtype=np.float32),
                "delta_center_2d": np.array([0, 0], dtype=np.float32),
                "semi_major_vector": np.array([0, 0], dtype=np.float32),
                "semi_minor_vector": np.array([0, 0], dtype=np.float32),
                "delta_semi_major_vector": np.array([0, 0], dtype=np.float32),
                "delta_semi_minor_vector": np.array([0, 0], dtype=np.float32)
            }

        points_to_fit_xy = points_to_fit[:, [1, 0]]

        (xc_px, yc_px), (d1_px, d2_px), angle_deg = cv2.fitEllipse(points_to_fit_xy)

        # Normalize results
        center_x_proj, center_y_proj = self.normalize_point(xc_px, yc_px, width, height)

        # For render method, we need to check if rendered points are visible
        # Since we already have the points that were used for ellipse fitting
        center_visible = len(points_to_fit) > 0  # If we found points to fit, some are visible

        # OpenCV's angle corresponds to d1_px, but we need to ensure it corresponds to the major axis
        if d1_px >= d2_px:
            # d1 is major axis, angle is correct
            semi_major_length = np.float32(d1_px / (2 * height))
            semi_minor_length = np.float32(d2_px / (2 * height))
            major_angle_rad = np.radians(angle_deg)
        else:
            # d2 is major axis, need to rotate angle by 90 degrees
            semi_major_length = np.float32(d2_px / (2 * height))
            semi_minor_length = np.float32(d1_px / (2 * height))
            major_angle_rad = np.radians(angle_deg + 90)

        semi_major_vector = np.array([
            semi_major_length * np.cos(major_angle_rad),
            semi_major_length * np.sin(major_angle_rad)
        ], dtype=np.float32)

        semi_minor_vector = np.array([
            -semi_minor_length * np.sin(major_angle_rad),
             semi_minor_length * np.cos(major_angle_rad)
        ], dtype=np.float32)

        # --- Compute deltas ---
        delta_center_2d = np.array([center_x_proj, center_y_proj], dtype=np.float32) - self.projected_properties.get("center_2d", np.array([0, 0], dtype=np.float32))
        delta_semi_major_vector = semi_major_vector - self.projected_properties.get("semi_major_vector", np.array([0, 0], dtype=np.float32))
        delta_semi_minor_vector = semi_minor_vector - self.projected_properties.get("semi_minor_vector", np.array([0, 0], dtype=np.float32))

        center_2d = np.array([center_x_proj, center_y_proj], dtype=np.float32)

        return {
            "visible": center_visible,
            "calculable": True,
            "center_2d": center_2d,
            "delta_center_2d": delta_center_2d,
            "semi_major_vector": semi_major_vector,
            "semi_minor_vector": semi_minor_vector,
            "delta_semi_major_vector": delta_semi_major_vector,
            "delta_semi_minor_vector": delta_semi_minor_vector
        }


def run_and_plot_case(ax, title, case_params, common_params, width, height, robot, ring, projector, plot_renderer, scene, mesh_node, camera_node):
    """
    Renders a single validation case and plots it on the given subplot axis.
    Reuses renderer components for performance.
    """
    # --- 1. Update system state for the current test case ---
    ring.outer_radius = common_params.get('ring_radius', 10.0)
    ring.position = np.array(case_params['ring_position'], dtype=float)
    ring.approach_vec = np.array(case_params['ring_normal'], dtype=float)
    
    # Dynamically calculate the ring's basis vectors for the test case
    ring_normal = ring.approach_vec
    if np.allclose(ring_normal, [0, 0, 1]) or np.allclose(ring_normal, [0, 0, -1]):
        ring.radial_vec = np.array([1, 0, 0])
    else:
        ring.radial_vec = projector._normalize(np.cross(ring_normal, [0, 0, 1]))
    ring.tangential_vec = projector._normalize(np.cross(ring_normal, ring.radial_vec))

    robot.camera_pos = np.array(common_params['camera_position'], dtype=float)
    robot.camera_view_vector = np.array(common_params['camera_view_dir'], dtype=float)
    robot.camera_up_vector = np.array(common_params['camera_up_dir'], dtype=float)

    print(f"\n--- Running Test Case: {title} ---")
    
    # --- 2. Update poses in the existing scene for ground-truth rendering ---
    ring_transform = trimesh.geometry.align_vectors([0, 0, 1], ring.approach_vec)
    ring_transform[:3, 3] = ring.position
    scene.set_pose(mesh_node, pose=ring_transform)

    z_axis = -robot.camera_view_vector
    x_axis = projector._normalize(np.cross(robot.camera_up_vector, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    camera_pose[:3, 3] = robot.camera_pos
    scene.set_pose(camera_node, pose=camera_pose)

    color_img, _ = plot_renderer.render(scene)

    # --- 3. Get stats from both methods ---
    projector.method = 'custom'
    projector.update()
    stats_custom = projector.projected_properties

    projector.method = 'render'
    projector.update()
    stats_render = projector.projected_properties

    # --- 4. Print comparison table ---
    print(f"{'Parameter':<20} | {'Vectorized':<20} | {'Render':<20}")
    print("-" * 65)
    if stats_custom.get("calculable") and stats_render.get("calculable"):
        # Calculate derived properties for printing
        major_axis_custom = np.linalg.norm(stats_custom['semi_major_vector']) * 2
        minor_axis_custom = np.linalg.norm(stats_custom['semi_minor_vector']) * 2
        angle_custom = np.degrees(np.arctan2(stats_custom['semi_major_vector'][1], stats_custom['semi_major_vector'][0]))

        major_axis_render = np.linalg.norm(stats_render['semi_major_vector']) * 2
        minor_axis_render = np.linalg.norm(stats_render['semi_minor_vector']) * 2
        angle_render = np.degrees(np.arctan2(stats_render['semi_major_vector'][1], stats_render['semi_major_vector'][0]))

        print(f"{'Center X (norm)':<20} | {stats_custom['center_2d'][0]:<20.4f} | {stats_render['center_2d'][0]:<20.4f}")
        print(f"{'Center Y (norm)':<20} | {stats_custom['center_2d'][1]:<20.4f} | {stats_render['center_2d'][1]:<20.4f}")
        print(f"{'Major Axis (norm)':<20} | {major_axis_custom:<20.4f} | {major_axis_render:<20.4f}")
        print(f"{'Minor Axis (norm)':<20} | {minor_axis_custom:<20.4f} | {minor_axis_render:<20.4f}")
        print(f"{'Orientation (deg)':<20} | {angle_custom:<20.2f} | {angle_render:<20.2f}")
    else:
        print("One or more methods could not calculate properties.")
    print("-" * 65)

    # --- 5. Draw overlays ---
    overlay_img = color_img.copy()
    
    def draw_ellipse(img, stats, color):
        if not stats.get("calculable"): return
        center_px = RingProjector.denormalize_point(*stats['center_2d'], width, height)
        
        semi_major_vec = stats['semi_major_vector']
        semi_minor_vec = stats['semi_minor_vector']

        major_axis_length = np.linalg.norm(semi_major_vec)
        minor_axis_length = np.linalg.norm(semi_minor_vec)

        major_axis_px = int(major_axis_length * height * 2)
        minor_axis_px = int(minor_axis_length * height * 2)
        
        angle_deg = np.degrees(np.arctan2(semi_major_vec[1], semi_major_vec[0]))

        cv2.ellipse(img, center_px, (major_axis_px // 2, minor_axis_px // 2),
                    angle_deg, 0, 360, color, 3)
        x0, y0 = width // 2, height // 2
        cv2.arrowedLine(img, (x0, y0), center_px, color, 2, tipLength=0.05)

    draw_ellipse(overlay_img, stats_custom, (0, 255, 0)) # Green for vectorized
    draw_ellipse(overlay_img, stats_render, (255, 0, 0)) # Red for render
    
    # --- 6. Plot on the subplot ---
    ax.imshow(overlay_img)
    ax.set_title(title)
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

    robot = RobotKinematics()
    ring = robot.create_ring()
    projector = RingProjector(robot, ring, vertical_fov_deg=common_params['camera_fov_deg'], image_width=IMG_WIDTH, image_height=IMG_HEIGHT)

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
            run_and_plot_case(axes[i], title, params, common_params, IMG_WIDTH, IMG_HEIGHT, robot, ring, projector, plot_renderer, scene, mesh_node, camera_node)

    # Clean up the persistent renderer
    plot_renderer.delete()

    fig.legend(handles=[plt.Line2D([0], [0], color='lime', lw=4, label='Vectorized'),
                        plt.Line2D([0], [0], color='red', lw=4, label='Render Method')],
               loc='lower center', ncol=2, fontsize=14)
    plt.suptitle("Validation: Vectorized (Green) vs. Render (Red)", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.show()

    # --- Performance Benchmark ---
    robot = RobotKinematics(verbosity=0)
    ring = robot.create_ring()
    projector = RingProjector(robot, ring, vertical_fov_deg=45.0, image_width=960, image_height=540)
    num_iterations = 100000

    print("--- Running Performance Benchmark (Vectorized) ---")
    projector.method = 'custom'
    robot.go_home()
    projector.update()
    
    start_time_vec = time.perf_counter()
    for i in range(num_iterations):
        robot.set_random_e1_pose()
        projector.update()
    end_time_vec = time.perf_counter()

    total_time_vec = end_time_vec - start_time_vec
    avg_time_ms_vec = (total_time_vec / num_iterations) * 1000
    fps_vec = num_iterations / total_time_vec

    print(f"Completed {num_iterations} iterations in {total_time_vec:.2f} seconds.")
    print(f"Average time per update: {avg_time_ms_vec:.4f} ms")
    print(f"Equivalent FPS (Vectorized): {fps_vec:.2f}")