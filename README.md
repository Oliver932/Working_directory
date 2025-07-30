# Robotic Arm Inverse Kinematics and Visualization

A comprehensive Python-based robotics simulation system for robotic arm inverse kinematics, collision detection, 3D visualization, and reinforcement learning training.

## Features

- **Inverse Kinematics**: Real-time calculation of joint positions for robotic arm control
- **3D Visualization**: Interactive rendering of robot configurations and ring targets
- **Collision Detection**: Mesh-based collision avoidance and safety constraints
- **Reachability Analysis**: Workspace exploration and gripper tolerance calculations
- **Reinforcement Learning**: PPO-based RL implementation with TensorBoard logging
- **Mesh Processing**: STL mesh simplification and geometry utilities
- **Video Generation**: Automated replay generation for training analysis

## Key Components

### Core Systems

- `arm_ik_model.py` - Core inverse kinematics solver and robot model
- `overview_render_manager.py` - 3D scene rendering and visualization
- `collision_and_render_management.py` - Collision detection systems
- `RL_implementation.py` - Reinforcement learning algorithms with PPO

### Analysis Tools

- `find_gripper_tolerances.py` - Gripper tolerance analysis
- `ring_projector.py` - Ring projection and targeting utilities
- `system_plot_functions.py` - Plotting and visualization functions
- `geometry_helper_functions.py` - Geometric computation utilities
- `mesh_simplifier.py` - STL mesh optimization tools

## Requirements

- Python 3.9+
- NumPy, SciPy
- PyRender, Trimesh
- PIL, Matplotlib
- PyYAML (for configuration)
- Stable-Baselines3 (for RL)
- TensorBoard (for training monitoring)

## Project Structure

Working_directory/
├── README.md
├── .gitignore
├── Core Python modules
│   ├── arm_ik_model.py              # Main IK solver and robot model
│   ├── overview_render_manager.py   # 3D visualization system
│   ├── collision_and_render_management.py  # Collision detection
│   ├── RL_implementation.py         # Reinforcement learning algorithms
│   ├── ring_projector.py           # Ring targeting utilities
│   ├── find_gripper_tolerances.py  # Tolerance analysis
│   ├── geometry_helper_functions.py # Geometric computations
│   ├── system_plot_functions.py    # Plotting utilities
│   └── mesh_simplifier.py          # STL mesh optimization
├── config/                         # Configuration files
│   ├── robot_config.yaml          # Robot parameters
│   └── ring_config.yaml           # Ring target parameters
│   ├── ring_render_mesh.stl        # Ring visualization mesh
│   ├── ring_collision_mesh.stl     # Ring collision mesh
│   └── gripper_collision_mesh.stl  # Gripper collision mesh
└── outputs/                        # Training outputs and logs
    ├── tensorboard_logs/           # TensorBoard training logs
    └── [run_folders]/              # Individual training runs
        ├── ppo_custom_robot.zip    # Trained model weights
        └── replays/                # Generated video replays
            ├── final_replay_overview.mp4
            └── final_replay_robot.mp4

## Usage

```python
from arm_ik_model import RobotKinematics
from overview_render_manager import OverviewRenderManager

# Initialize robot and render manager
robot = RobotKinematics()
ring = robot.create_ring()
renderer = OverviewRenderManager(robot, ring, 'meshes/ring_render_mesh.stl')

# Render scene
image = renderer.render_to_image()
```

## Configuration

Robot and ring parameters are configurable via YAML files in the `config/` directory.

## STL Meshes

Place your 3D mesh files in the `meshes/` directory. Required:

- `ring_render_mesh.stl` - Target ring visualization mesh
- `ring_collision_mesh.stl` - Target ring collision mesh
- `gripper_collision_mesh.stl` - Gripper collision mesh

## Training Outputs

The `outputs/` directory contains:

- `tensorboard_logs/` - TensorBoard logs for monitoring training progress
- Individual run folders with trained models (`ppo_custom_robot.zip`)
- Generated replay videos (`final_replay_overview.mp4`, `final_replay_robot.mp4`)
