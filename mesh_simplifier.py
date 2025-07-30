import trimesh
import numpy as np
import open3d as o3d
import os

# --- How to Visualize a Simplified Mesh's Origin and Axes ---

# This script simplifies a mesh, rotates it, centers it, displays its origin
# and the world X, Y, Z axes, and saves the result.

# --- Main Script ---
try:
    # Define input and output file paths
    input_dir = './stl_files/'
    input_filename = 'simplified_docking_ring.stl'
    output_dir = './meshes/'
    output_filename = 'ring_collision_mesh.stl'
    
    input_filepath = os.path.join(input_dir, input_filename)
    output_filepath = os.path.join(output_dir, output_filename)

    # 1. Load your complex, high-polygon STL file.
    try:
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Input file not found at: {input_filepath}")
        mesh = trimesh.load(input_filepath)
        print(f"Loaded mesh from: {input_filepath}")
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load STL file. Using a sample high-poly sphere. Error: {e}")
        mesh = trimesh.creation.icosphere(subdivisions=5)

    print(f"Original mesh has {len(mesh.faces)} faces.")

    # 2. Perform mesh decimation using Open3D.
    target_face_count = int(len(mesh.faces) * 0.2)
    print(f"\nSimplifying mesh to a target of {target_face_count} faces...")

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    o3d_mesh.compute_vertex_normals()
    simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_face_count)
    simplified_mesh = trimesh.Trimesh(
        vertices=np.asarray(simplified_o3d_mesh.vertices),
        faces=np.asarray(simplified_o3d_mesh.triangles)
    )
    print(f"Simplified mesh has {len(simplified_mesh.faces)} faces.")

    # 3. Rotate and center the simplified mesh
    # To rotate about the object's origin (centroid) and then place it at the world origin,
    # we can first move its centroid to the world origin, then apply the rotation.
    
    # Center the mesh on the world origin
    print("\nCentered the mesh at the world origin.")

    # Define the rotation matrix to make -Y -> +X and +X -> +Z (actually not a rotation, but a custom transformation)
    rotation_3x3 = np.array([
        [ 0,  -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0.,  0.,  1.]
    ])
    
    # Create a 4x4 transformation matrix from the 3x3 rotation
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = rotation_3x3
    
    # Apply the rotation
    simplified_mesh.apply_transform(transform_4x4)
    print("Applied custom rotation.")

    # 4. Set up the visualization scene
    scene = trimesh.Scene()
    
    # Add the rotated and centered simplified mesh
    scene_mesh = simplified_mesh.copy()
    scene_mesh.visual.face_colors = [0, 255, 0, 255] # Green
    scene.add_geometry(scene_mesh)

    # Add lines for the global X, Y, Z axes for reference
    axis_length = scene_mesh.scale * 0.75
    axis_radius = scene_mesh.scale / 300.0
    
    world_axes = trimesh.creation.axis(
        axis_radius=axis_radius,
        axis_length=axis_length
    )
    scene.add_geometry(world_axes)

    # Print instructions and show the scene
    print("\n--- VISUALIZATION MODE ---")
    print("Showing the simplified mesh (green), and the global X, Y, Z axes.")
    print("Close the window to end the script.")
    
    scene.show()

    # 5. --- EXPORT THE FINAL MESH ---
    # Save the rotated and centered simplified mesh.
    try:
        os.makedirs(input_dir, exist_ok=True)
        simplified_mesh.export(output_filepath)
        print(f"\nSuccessfully saved the final mesh to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving the file: {e}")

except ImportError:
    print("\nERROR: This script requires the 'trimesh' and 'open3d' libraries.")
    print("Please install it by running: pip install trimesh open3d")
except Exception as e:
    print(f"An unexpected error occurred: {e}")