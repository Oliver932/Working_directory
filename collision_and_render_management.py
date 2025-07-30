import sys
import numpy as np
# Monkey-patch for numpy 2.0 compatibility: np.infty was removed.
# This makes it available again for libraries that haven't updated yet.
np.infty = np.inf
import trimesh
import pyrender
import os
from PIL import Image

# The line forcing the EGL platform has been removed to allow pyrender
# to auto-detect an available rendering platform.
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Import your actual RobotKinematics and Ring classes here
from arm_ik_model import RobotKinematics, Ring


class CollisionAndRenderManager:
    """
    Manages collision detection using trimesh and efficient rendering using pyrender.
    Allows for separate meshes for visuals and collision.
    """
    def __init__(self, gripper_visual_path, gripper_collision_path, ring_visual_path, ring_collision_path, vertical_FOV=60.0, render_width=640, render_height=480):
        """
        Initializes the manager, loads meshes, and sets up the scene.
        
        Args:
            gripper_visual_path (str): Path to the gripper's visual STL file.
            gripper_collision_path (str): Path to the gripper's collision STL file.
            ring_visual_path (str): Path to the ring's visual STL file.
            ring_collision_path (str): Path to the ring's collision STL file.
            vertical_FOV (float): Vertical field of view for the pyrender camera in degrees.
            render_width (int): Width of the rendered image (for offscreen renderer only).
            render_height (int): Height of the rendered image (for offscreen renderer only).
        
        Saves vertical_FOV, render_width, render_height, and precalculates aspect ratio and horizontal FOV.
        """
        # Load the collision meshes for trimesh
        print(f"Attempting to load gripper collision mesh from: {gripper_collision_path}")
        gripper_collision_mesh = trimesh.load_mesh(gripper_collision_path)
        assert not gripper_collision_mesh.is_empty, f"Gripper collision mesh is empty: {gripper_collision_path}"
        print(f"-> Gripper collision mesh loaded successfully.")

        print(f"Attempting to load ring collision mesh from: {ring_collision_path}")
        ring_collision_mesh = trimesh.load_mesh(ring_collision_path)
        assert not ring_collision_mesh.is_empty, f"Ring collision mesh is empty: {ring_collision_path}"
        print(f"-> Ring collision mesh loaded successfully.")

        # Load the visual meshes for pyrender
        print(f"Attempting to load gripper visual mesh from: {gripper_visual_path}")
        gripper_visual_mesh = trimesh.load_mesh(gripper_visual_path)
        assert not gripper_visual_mesh.is_empty, f"Gripper visual mesh is empty: {gripper_visual_path}"
        print(f"-> Gripper visual mesh loaded successfully.")

        print(f"Attempting to load ring visual mesh from: {ring_visual_path}")
        ring_visual_mesh = trimesh.load_mesh(ring_visual_path)
        assert not ring_visual_mesh.is_empty, f"Ring visual mesh is empty: {ring_visual_path}"
        print(f"-> Ring visual mesh loaded successfully.")

        # Create persistent CollisionManager for trimesh using collision meshes
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('gripper', gripper_collision_mesh)
        self.collision_manager.add_object('ring', ring_collision_mesh)
        
        # --- Pyrender Setup ---
        # Create pyrender-compatible mesh objects from visual meshes
        self.gripper_prim = pyrender.Mesh.from_trimesh(gripper_visual_mesh, smooth=False)
        self.ring_prim = pyrender.Mesh.from_trimesh(ring_visual_mesh, smooth=False)

        # Save camera/render parameters
        self.vertical_FOV = vertical_FOV
        self.render_width = render_width
        self.render_height = render_height
        self.aspect_ratio = render_width / render_height
        self.horizontal_FOV = 2 * np.rad2deg(np.arctan(np.tan(np.deg2rad(self.vertical_FOV) / 2) * self.aspect_ratio))

        # Create the pyrender scene and add nodes for the objects
        self.pyrender_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        self.gripper_node = self.pyrender_scene.add(self.gripper_prim, name='gripper')
        self.ring_node = self.pyrender_scene.add(self.ring_prim, name='ring')

        yfov = np.deg2rad(self.vertical_FOV)
        self.camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=self.aspect_ratio)
        self.camera_node = self.pyrender_scene.add(self.camera, name='camera')
        self.pyrender_scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2e3), name='light')

        # Keep a simple trimesh scene ONLY for debugging the overview, using collision meshes
        self.debug_scene = trimesh.Scene()
        self.debug_scene.add_geometry(gripper_collision_mesh, node_name='gripper')
        self.debug_scene.add_geometry(ring_collision_mesh, node_name='ring')

        # Store the latest transforms for debugging
        self.gripper_tf = np.eye(4)
        self.ring_tf = np.eye(4)
        
        # Create the offscreen renderer (still uses width/height for output image size)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=render_width, viewport_height=render_height)



    @staticmethod
    def _make_transform(position, approach_vec, tangential_vec, radial_vec):
        """
        Builds a 4x4 transform matrix from position and orientation vectors.
        """
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([approach_vec, radial_vec, tangential_vec])
        transform[:3, 3] = position
        return transform

    @staticmethod
    def _create_look_at(eye, center, up):
        
        """
            vertical_FOV (float): Vertical field of view for the pyrender camera in degrees.
            render_width (int): Width of the rendered image (for offscreen renderer only).
            render_height (int): Height of the rendered image (for offscreen renderer only).
        
        Saves vertical_FOV, render_width, render_height, and precalculates aspect ratio and horizontal FOV."""

        up = np.asarray(up, dtype=np.float64)

        # Calculate forward, right, and new up vectors
        direction = center - eye
        # Handle cases where eye and center are the same
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0, 0, -1])
        else:
            direction /= np.linalg.norm(direction)

        right = np.cross(direction, up)
        if np.linalg.norm(right) < 1e-6:
            # If direction and up are parallel, choose a different 'up'
            if np.allclose(np.abs(direction), [0, 0, 1]):
                right = np.cross(direction, [0, 1, 0])
            else:
                right = np.cross(direction, [0, 0, 1])

        right /= np.linalg.norm(right)
        new_up = np.cross(right, direction)

        # Create the camera transform matrix
        rotation = np.eye(4)
        rotation[:3, 0] = right
        rotation[:3, 1] = new_up
        rotation[:3, 2] = -direction
        
        translation = np.eye(4)
        translation[:3, 3] = -eye
        
        # The camera transform is the product of the rotation and translation
        return np.dot(rotation.T, translation)

    def update_poses(self, robot, ring):
        self.gripper_tf = self._make_transform(robot.G1, robot.approach_vec, robot.tangential_vec, robot.radial_vec)
        self.ring_tf = self._make_transform(ring.position, ring.approach_vec, ring.tangential_vec, ring.radial_vec)

        # Update collision manager
        self.collision_manager.set_transform('gripper', transform=self.gripper_tf)
        self.collision_manager.set_transform('ring', transform=self.ring_tf)

        # Update pyrender node poses
        self.pyrender_scene.set_pose(self.gripper_node, pose=self.gripper_tf)
        self.pyrender_scene.set_pose(self.ring_node, pose=self.ring_tf)

        cam_pos = np.array(robot.camera_pos)
        target = cam_pos + np.array(robot.camera_view_vector)
        cam_up = np.array(robot.camera_up_vector)

        # Pyrender's look_at is different from trimesh's
        # It needs to be inverted for the camera node pose
        view_matrix = self._create_look_at(eye=cam_pos, center=target, up=cam_up)
        camera_pose = np.linalg.inv(view_matrix)
        self.pyrender_scene.set_pose(self.camera_node, pose=camera_pose)

    def check_collision(self):
        """
        Checks for collision using the trimesh collision manager.
        """
        return self.collision_manager.in_collision_internal()

    def render(self):
        """
        Renders the scene from the robot's perspective using an offscreen renderer.
        
        Returns:
            np.ndarray: The color image as a numpy array (RGB, uint8), suitable for saving with PIL.Image.fromarray.
        """
        color, _ = self.renderer.render(self.pyrender_scene)
        return color

    def render_scene_overview(self, show=True):
        """
        Renders a simple overview using trimesh for debugging purposes.
        """
        if not show:
            return

        # Get current transforms from the stored attributes
        gripper_tf = self.gripper_tf
        ring_tf = self.ring_tf

        # Update the debug scene
        self.debug_scene.graph.update('gripper', matrix=gripper_tf)
        self.debug_scene.graph.update('ring', matrix=ring_tf)

        # Use trimesh's automatic camera to frame everything
        self.debug_scene.set_camera()
        
        print("Note: Trimesh 3D viewer disabled to prevent hanging. Scene updated successfully.")
        print("If you need 3D visualization, consider using the pyrender output instead.")
        
        # Optionally, you can still try to show it but with better error handling
        # Uncomment the lines below if you want to try the 3D viewer:
        # try:
        #     import threading
        #     import time
        #     
        #     def show_scene():
        #         self.debug_scene.show(block=False)
        #     
        #     # Run in a separate thread with timeout
        #     viewer_thread = threading.Thread(target=show_scene)
        #     viewer_thread.daemon = True
        #     viewer_thread.start()
        #     viewer_thread.join(timeout=2.0)  # 2 second timeout
        #     
        # except Exception as e:
        #     print(f"Warning: Could not display trimesh scene: {e}")
        #     print("Continuing without trimesh visualization...")

    def cleanup(self):
        """
        Clean up resources, especially the offscreen renderer.
        """
        if hasattr(self, 'renderer'):
            self.renderer.delete()
            print("-> Cleaned up offscreen renderer")


if __name__ == "__main__":
    import time
    # --- Example Usage with Benchmarking ---

    # 1. Initialize the manager with paths to your STL files
    gripper_mesh_path = './meshes/gripper_collision_mesh.stl'
    ring_collision_mesh_path = './meshes/ring_collision_mesh.stl'
    ring_render_mesh_path = './meshes/ring_render_mesh.stl'

    t0 = time.perf_counter()
    scene_manager = CollisionAndRenderManager(
        gripper_visual_path=gripper_mesh_path,
        gripper_collision_path=gripper_mesh_path,
        ring_visual_path=ring_render_mesh_path,
        ring_collision_path=ring_collision_mesh_path,
        vertical_FOV=60.0,
        render_width=640,
        render_height=480
    )
    t1 = time.perf_counter()
    print(f"[Benchmark] CollisionAndRenderManager init: {t1-t0:.4f} s")

    # 2. Create your robot and ring instances
    t0 = time.perf_counter()
    robot = RobotKinematics(verbosity=0)
    t1 = time.perf_counter()
    print(f"[Benchmark] RobotKinematics init: {t1-t0:.4f} s")

    t0 = time.perf_counter()
    robot.update_from_e1_pose(np.array([0, 100, 0]), 0, 0)
    t1 = time.perf_counter()
    print(f"[Benchmark] update_from_e1_pose: {t1-t0:.4f} s")

    t0 = time.perf_counter()
    ring = robot.create_ring()
    t1 = time.perf_counter()
    print(f"[Benchmark] robot.create_ring: {t1-t0:.4f} s")
    if ring is None:
        raise ValueError("The 'robot.create_ring()' method returned None.")

    # 3. Update the scene with the new poses
    t0 = time.perf_counter()
    scene_manager.update_poses(robot, ring)
    t1 = time.perf_counter()
    print(f"[Benchmark] scene_manager.update_poses: {t1-t0:.4f} s")

    # 4. Perform a collision check
    t0 = time.perf_counter()
    collides = scene_manager.check_collision()
    t1 = time.perf_counter()
    print(f"[Benchmark] scene_manager.check_collision: {t1-t0:.4f} s")
    print(f"Collision detected (initial position): {collides}")

    # 5. Render a static image from the robot's perspective using Pyrender
    print("Rendering static image from robot's perspective (Pyrender)...")
    t0 = time.perf_counter()
    color_image = scene_manager.render()
    t1 = time.perf_counter()
    print(f"[Benchmark] scene_manager.render: {t1-t0:.4f} s")

    # Display the rendered image
    if color_image is not None:
        img = Image.fromarray(color_image, 'RGB')
        img.show()
        print("-> Displayed pyrender output")

    # 6. Render the scene from an overview perspective using Trimesh for debugging
    print("Rendering scene from overview perspective (Trimesh)...")
    t0 = time.perf_counter()
    # Set show=False to disable the 3D viewer and prevent hanging
    scene_manager.render_scene_overview(show=False)  
    t1 = time.perf_counter()
    print(f"[Benchmark] scene_manager.render_scene_overview: {t1-t0:.4f} s")
    
    # Clean up resources
    scene_manager.cleanup()
    print("Script completed successfully!")
    sys.exit()