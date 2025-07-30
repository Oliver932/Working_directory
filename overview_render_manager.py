import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image
import os
import matplotlib.pyplot as plt

# Monkey-patch for numpy 2.0 compatibility: np.infty was removed.
# This makes it available again for libraries that haven't updated yet.
np.infty = np.inf

# Assuming your arm_ik_model is in a location Python can find
from arm_ik_model import RobotKinematics, Ring

# --- Color Definitions ---
COLORS = {
    'A_points': [0.0, 0.0, 1.0, 1.0], 'C_points': [1.0, 0.0, 0.0, 1.0],
    'E1_point': [0.5, 0.0, 0.5, 1.0], 'G1_point': [0.0, 1.0, 0.0, 1.0],
    'actuator_path': [0.3, 0.3, 0.3, 0.5], 'link_AC': [0.0, 0.0, 1.0, 1.0],
    'link_CC': [1.0, 0.0, 0.0, 1.0], 'link_EG': [0.0, 0.5, 0.0, 1.0],
    'grab_connectors': [0.1, 0.1, 0.1, 1.0], 'approach_vec': [0.0, 1.0, 0.0, 1.0],
    'tangential_vec': [1.0, 0.65, 0.0, 1.0], 'radial_vec': [0.5, 0.0, 0.5, 1.0],
    'camera_view': [1.0, 0.0, 1.0, 1.0], 'camera_up': [0.18, 0.55, 0.9, 1.0],
    'ring': [0.0, 0.75, 0.75, 0.6],
}

class OverviewRenderManager:
    """
    Manages the rendering of a robot and ring scene to an offscreen buffer,
    allowing for the generation of images as NumPy arrays.
    This version is optimized for performance by creating a persistent scene
    and updating node poses instead of rebuilding the scene for each frame.
    """
    def __init__(self, robot, ring, ring_stl_path, width=300, height=200):
        self.robot = robot
        self.ring = ring
        self.width = width
        self.height = height
        self.nodes = {}
        self.mesh_cache = {}  # Mesh cache for reusing identical meshes

        # Load the ring mesh once
        try:
            self.ring_trimesh = trimesh.load_mesh(ring_stl_path)
            print(f"Successfully loaded ring mesh from '{ring_stl_path}'")
        except Exception as e:
            raise IOError(f"Error loading STL file '{ring_stl_path}': {e}")

        # Initialize the persistent scene and renderer
        self.scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4, 1.0], bg_color=[1.0, 1.0, 1.0, 1.0])
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        
        # Build the scene graph with all objects once
        self._build_persistent_scene()

    def _build_persistent_scene(self):
        """Creates all meshes and adds them as nodes to the scene once."""
        # --- Create Static Meshes (Actuator Paths) ---
        for i in range(1, 5):
            p_home = self.robot.params['A_home'][f'A{i}']
            p_end = self.robot.params['A_end'][f'A{i}']
            mesh = self._create_line_mesh(p_home, p_end, COLORS['actuator_path'], thickness=1.0)
            if mesh: self.scene.add(mesh)

        # --- Create Dynamic Meshes and Nodes ---
        # Points (Spheres)
        point_keys = [f'A{i}' for i in range(1,5)] + [f'C{i}' for i in range(1,5)] + ['E1', 'G1']
        point_colors = ['A_points']*4 + ['C_points']*4 + ['E1_point', 'G1_point']
        for key, color_key in zip(point_keys, point_colors):
            mesh = self._get_cached_sphere_mesh(radius=5.0, color=tuple(COLORS[color_key]))
            self.nodes[key] = self.scene.add(mesh, pose=np.eye(4))
        
        # Linkages (Cylinders)
        link_keys = [f'link_A{i}_C{i}' for i in range(1,5)] + ['link_C1_C2', 'link_C2_C4', 'link_C4_C3', 'link_C3_C1', 'link_E1_G']
        link_colors = ['link_AC']*4 + ['link_CC']*4 + ['link_EG']
        for key, color_key in zip(link_keys, link_colors):
            mesh = self._get_cached_cylinder_mesh(thickness=5.0, color=tuple(COLORS[color_key]))
            self.nodes[key] = self.scene.add(mesh, pose=np.eye(4))

        # Grab Points and Connectors
        for i in range(3): # outer, inner1, inner2
            self.nodes[f'grab_point_{i}'] = self.scene.add(self._get_cached_sphere_mesh(radius=4.0, color=tuple(COLORS['grab_connectors'])))
            self.nodes[f'grab_connector_{i}_a'] = self.scene.add(self._get_cached_cylinder_mesh(thickness=5.0, color=tuple(COLORS['grab_connectors'])))
            self.nodes[f'grab_connector_{i}_b'] = self.scene.add(self._get_cached_cylinder_mesh(thickness=5.0, color=tuple(COLORS['grab_connectors'])))

        # Orientation Vectors (Arrows) -- REMOVED

        # Ring
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=COLORS['ring'], alphaMode='BLEND', doubleSided=True)
        mesh = pyrender.Mesh.from_trimesh(self.ring_trimesh, material=material)
        self.nodes['ring'] = self.scene.add(mesh, pose=np.eye(4))

        # Camera and Light
        self.nodes['camera'] = self.scene.add(pyrender.OrthographicCamera(xmag=1, ymag=1, znear=0.01, zfar=10000), pose=np.eye(4))
        self.nodes['light'] = self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))

    def _get_cached_sphere_mesh(self, radius, color):
        key = ('sphere', radius, color)
        if key not in self.mesh_cache:
            self.mesh_cache[key] = self._create_sphere_mesh_at_origin(radius, color)
        return self.mesh_cache[key]

    def _get_cached_cylinder_mesh(self, thickness, color):
        key = ('cylinder', thickness, color)
        if key not in self.mesh_cache:
            self.mesh_cache[key] = self._create_unit_cylinder_mesh(thickness, color)
        return self.mesh_cache[key]

    # _get_cached_arrow_mesh removed (arrows not used)

    def _update_scene_poses(self):
        """Updates the poses of all dynamic nodes in the scene."""
        # Update points
        for key in self.nodes:
            if key.endswith('_point'):
                point_attr = key.replace('_point', '')
                point_pos = getattr(self.robot, point_attr, None)
                if point_pos is not None:
                    self.scene.set_pose(self.nodes[key], trimesh.transformations.translation_matrix(point_pos))

        # Update linkages
        self._update_line_pose('link_A1_C1', self.robot.A1, self.robot.C1)
        self._update_line_pose('link_A2_C2', self.robot.A2, self.robot.C2)
        self._update_line_pose('link_A3_C3', self.robot.A3, self.robot.C3)
        self._update_line_pose('link_A4_C4', self.robot.A4, self.robot.C4)
        self._update_line_pose('link_C1_C2', self.robot.C1, self.robot.C2)
        self._update_line_pose('link_C2_C4', self.robot.C2, self.robot.C4)
        self._update_line_pose('link_C4_C3', self.robot.C4, self.robot.C3)
        self._update_line_pose('link_C3_C1', self.robot.C3, self.robot.C1)
        self._update_line_pose('link_E1_G', self.robot.E1, self.robot.gripper_back)
        
        # Update Grab Points
        grab_points = [self.robot.outer_grip_point, self.robot.inner_grip_point_1, self.robot.inner_grip_point_2]
        for i, p in enumerate(grab_points):
            if p is not None:
                self.scene.set_pose(self.nodes[f'grab_point_{i}'], trimesh.transformations.translation_matrix(p))
                gripper_back_pt = self.robot.gripper_back
                u = self.robot.approach_vec
                v = gripper_back_pt - p
                corner_point = p + np.dot(v, u) * u
                self._update_line_pose(f'grab_connector_{i}_a', p, corner_point)
                self._update_line_pose(f'grab_connector_{i}_b', corner_point, gripper_back_pt)

        # Update vectors -- REMOVED (arrows not used)

        # Update Ring
        source_vec, target_vec = [1, 0, 0], self.ring.approach_vec
        align_rot = trimesh.geometry.align_vectors(source_vec, target_vec)
        trans = trimesh.transformations.translation_matrix(self.ring.position)
        self.scene.set_pose(self.nodes['ring'], trans @ align_rot)

        # Update Camera and Light
        bounds = self.scene.bounds
        centroid, scale = self.scene.centroid, self.scene.scale
        eye = centroid + np.array([-0.1 * scale, 0.15 * scale, -1.5 * scale])
        target = centroid
        up = np.array([0.0, 0.7, 0.4])
        cam_pose = self._create_look_at_pose(eye, target, up)
        
        distance = np.linalg.norm(eye - target)
        znear = max(scale / 100.0, distance - scale * 2)
        zfar = distance + scale * 2
        mag = scale * 0.4
        
        self.nodes['camera'].camera.xmag = mag
        self.nodes['camera'].camera.ymag = mag
        self.nodes['camera'].camera.znear = znear
        self.nodes['camera'].camera.zfar = zfar

        self.scene.set_pose(self.nodes['camera'], cam_pose)
        self.scene.set_pose(self.nodes['light'], cam_pose)

    def _update_line_pose(self, key, p1, p2):
        """Helper to calculate and set the pose for a line node."""
        node = self.nodes[key]
        vector = p2 - p1
        length = np.linalg.norm(vector)
        if length < 1e-6:
            self.scene.set_pose(node, np.diag([0,0,0,1])) # Hide if zero length
            return
        midpoint = (p1 + p2) / 2
        transform = trimesh.transformations.translation_matrix(midpoint)
        align = trimesh.geometry.align_vectors([0, 0, 1], vector)
        scale = np.diag([1, 1, length, 1])
        self.scene.set_pose(node, transform @ align @ scale)

    # _update_arrow_pose removed (arrows not used)

    def render_to_image(self):
        if self.robot.last_solve_successful:
            self._update_scene_poses()

        color_image, _ = self.renderer.render(self.scene)
        return color_image

    # --- Static methods for creating mesh primitives at the origin ---
    @staticmethod
    def _create_sphere_mesh_at_origin(radius, color):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        mesh.visual.face_colors = color
        return pyrender.Mesh.from_trimesh(mesh, smooth=False)

    @staticmethod
    def _create_line_mesh(p1, p2, color, thickness):
        """Creates a line mesh at a specific location (for static objects)."""
        vector = p2 - p1
        length = np.linalg.norm(vector)
        if length < 1e-6: return None
        midpoint = (p1 + p2) / 2
        transform = trimesh.transformations.translation_matrix(midpoint)
        align = trimesh.geometry.align_vectors([0, 0, 1], vector)
        cylinder = trimesh.creation.cylinder(radius=thickness/2, height=length, transform=transform @ align)
        cylinder.visual.face_colors = color
        return pyrender.Mesh.from_trimesh(cylinder, smooth=False)

    @staticmethod
    def _create_unit_cylinder_mesh(thickness, color):
        """Creates a cylinder of length 1 at the origin, aligned with Z-axis."""
        cylinder = trimesh.creation.cylinder(radius=thickness/2, height=1.0)
        cylinder.visual.face_colors = color
        return pyrender.Mesh.from_trimesh(cylinder, smooth=False)

    # _create_unit_arrow_mesh removed (arrows not used)

    @staticmethod
    def _create_look_at_pose(eye, target, up):
        f = (target - eye)
        f /= np.linalg.norm(f)
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        pose = np.eye(4)
        pose[0, :3], pose[1, :3], pose[2, :3], pose[:3, 3] = s, u, -f, eye
        return pose

if __name__ == "__main__":
    stl_directory = './meshes/'
    stl_filename = 'ring_render_mesh.stl'
    stl_path = os.path.join(stl_directory, stl_filename)

    if not os.path.exists(stl_path):
        print(f"Error: '{stl_path}' not found. Please ensure the STL file exists.")
        exit(1)

    robot_instance = RobotKinematics(verbosity=0)
    ring_instance = robot_instance.create_ring()

    try:
        print("Initializing Render Manager...")
        render_manager = OverviewRenderManager(robot_instance, ring_instance, stl_path)
        
        print("Rendering scene to image...")
        image_array = render_manager.render_to_image()

        if image_array is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(image_array)
            plt.title('Rendered Robot Scene')
            plt.axis('off')
            plt.show()
            print("Successfully displayed rendered image")

        # --- Example of updating and re-rendering (for animation) ---
        print("\n--- Simulating an update for animation ---")
        robot_instance.set_random_e1_pose( max_attempts=10)
        print("Robot pose updated. Re-rendering...")
        
        updated_image_array = render_manager.render_to_image()
        
        if updated_image_array is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(updated_image_array)
            plt.title('Updated Robot Scene')
            plt.axis('off')
            plt.show()
            print("Successfully displayed updated image")

    except Exception as e:
        print(f"An error occurred during setup or rendering: {e}")