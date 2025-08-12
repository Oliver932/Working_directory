import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 20
})

from RL_training import CustomRobotEnv
from RL_evaluation import load_trained_model_and_envs

import serial
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque
import traceback
import shap
import mlflow
import os
import shutil

# The maximum travel distance of the linear actuators (in millimeters).
ACTUATOR_LIMIT = 400
# This is calculated from a 2mm pitch leadscrew (it has 8mm lead but 2mm pitch!!) and a stepper motor with 400 steps per revolution.
DIST_PER_STEP = 8 / 400

# --- Mock Serial for Simulation Mode ---
class MockSerial:
    def __init__(self):
        self.in_waiting = 1
        self._last_command = None
    def write(self, data):
        self._last_command = data.decode('utf-8').strip()
        print(f"[SIM] Sent to Arduino: {self._last_command}")
        self.in_waiting = 1
    def readline(self):
        # Simulate immediate DONE response
        if self._last_command is not None:
            self.in_waiting = 0
            return b"DONE\n"
        return b""
    def close(self):
        print("[SIM] Closed mock serial connection.")

def connect_to_arduino(simulation_mode=False):
    """
    Prompts user for Arduino COM port and establishes a serial connection.
    Returns the serial connection object, or None if connection fails.
    If simulation_mode is True, returns a MockSerial instance.
    """
    if simulation_mode:
        print("[SIM] Running in simulation mode. Using mock serial connection.")
        return MockSerial()
    print("--- Arduino Serial Connection Setup ---")
    try:
        port_name = input("Enter the Arduino's serial port (e.g., COM4): ")
        arduino = serial.Serial(port=port_name, baudrate=115200, timeout=1)
        time.sleep(2)  # Wait for connection to initialize
        print(f"Connected to Arduino on {port_name}")
        return arduino
    except serial.SerialException as e:
        print(f"Error: Could not open serial port '{port_name}'. Details: {e}")
        return None

class RealRobot:
    """
    A stateful class to manage real robot communication and action execution.
    Handles scaling, formatting, and sending actions over serial to the Arduino.
    """
    
    def __init__(self, config, arduino):
        """
        Initialize the RealRobot with configuration and serial connection.
        Args:
            config (dict): The configuration dictionary containing multipliers.
            arduino (serial.Serial or MockSerial): The serial connection to the Arduino.
        """
        self.config = config
        self.arduino = arduino

        # set the robot to home position
        self.arduino.write(('HOME\n').encode('utf-8'))

        # Wait for Arduino to confirm home position
        while True:
            try:
                response = self.arduino.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino says: {response}")
                    if response == "DONE":
                        break
            except UnicodeDecodeError:
                print("Arduino says: (Could not decode response)")

        self.extensions = np.zeros(4)  # Initialize extensions for 4 actuators

        # ungrip the gripper
        self.set_gripper_state('UNGRIP')

    def set_robot_position(self, extensions):

        """
        Set the robot's end effector position based on the provided extensions.
        Args:
            extensions (np.ndarray): The extension values for each actuator.
        """
        self.extensions = np.asarray(extensions).flatten()

        # validate that the extensions are in the range 0 to 1
        if not np.all((0 <= self.extensions) & (self.extensions <= 1)):
            raise ValueError("Extensions must be in the range [0, 1].")

        # Format the extensions as a comma-separated string
        extensions_str = ','.join(str(float(e) * ACTUATOR_LIMIT / DIST_PER_STEP) for e in self.extensions)

        print(f"Setting robot position with actuator steps: {extensions_str}")
        self.arduino.write((extensions_str + '\n').encode('utf-8'))

        # Wait for Arduino to confirm position setting
        while True:
            try:
                response = self.arduino.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino says: {response}")
                    if response == "DONE":
                        break
            except UnicodeDecodeError:
                print("Arduino says: (Could not decode response)")
        
        print(f"Set robot position with extensions: {self.extensions}")

    def set_gripper_state(self, state):
        """
        Set the gripper state to either grip or ungrip.
        Args:
            state (str): 'GRIP' to grip, 'UNGRIP' to release.
        """

        if state not in ['GRIP', 'UNGRIP']:
            raise ValueError("Invalid gripper state. Use 'GRIP' or 'UNGRIP'.")

        self.arduino.write((state + '\n').encode('utf-8'))

        # Wait for Arduino to confirm gripper action
        while True:
            try:
                response = self.arduino.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino says: {response}")
                    if response == "DONE":
                        break
            except UnicodeDecodeError:
                print("Arduino says: (Could not decode response)")

        self.gripped = (state == 'GRIP')
    
    def close(self):
        """
        Close the serial connection to the Arduino.
        """
        self.arduino.close()

def main():

    # 0. Ask if running in simulation mode
    simulation_mode = input("Run in simulation mode? (y/N): ").strip().lower() == 'y'

    # Ask user if manual confirmation is required after each step
    wait_for_user = input("Wait for user confirmation after each robot action? (y/N): ").strip().lower() == 'y'

    # Ask user if they want SHAP explanations alongside the existing plots
    plot_shap = input("Plot SHAP heatmaps for each action during the episode? (y/N): ").strip().lower() == 'y'

    # 1. Connect to Arduino (or mock)
    arduino = connect_to_arduino(simulation_mode=simulation_mode)
    if arduino is None:
        print("Exiting due to serial connection failure.")
        return

    # 2. Prompt user for MLflow run ID
    mlflow_run_id = input("Enter the MLflow run ID to load the model and environment: ")

    # 3. Load trained model and environment
    print("Loading model and environment from MLflow...")
    try:
        config, model, env = load_trained_model_and_envs(mlflow_run_id)
    except Exception as e:
        print(f"Failed to load model/env: {e}")
        arduino.close()
        return
    print("Model and environment loaded successfully.")

    # --- Always import SHAP and MLflow, always load background and compute baseline ---
    import mlflow
    import shap
    print("Loading SHAP background for baseline computation...")
    background_sample = None
    model_predict_wrapper = None
    baseline_actions = None
    try:
        mlflow.set_tracking_uri("file:///C:/temp/mlruns")  # Adjust if needed
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(mlflow_run_id, path="shap")
        bg_flat_file = None
        for art in artifacts:
            if art.path.endswith("shap_background.npy"):
                bg_flat_file = art.path
                break
        if bg_flat_file is not None:
            # Use run-specific temp directory for SHAP background download
            temp_dir_shap = os.path.join("C:/temp/artifacts", f"run_{mlflow_run_id}")
            os.makedirs(temp_dir_shap, exist_ok=True)
            
            with mlflow.start_run(run_id=mlflow_run_id):
                bg_local_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run_id, artifact_path=bg_flat_file, dst_path=temp_dir_shap)
            shap_background_flat = np.load(bg_local_path, allow_pickle=True)
            print(f"Loaded SHAP background dataset (flattened), {shap_background_flat.shape[0]} samples, {shap_background_flat.shape[1]} features")
            REDUCED_BG_SAMPLES = 20
            background_sample = shap_background_flat[:REDUCED_BG_SAMPLES]
            print(f"Using only {REDUCED_BG_SAMPLES} background samples for SHAP baseline.")

            # Build obs_keys for reconstruction
            obs_keys = list(env.observation_space.spaces.keys()) if hasattr(env.observation_space, 'spaces') else ['obs']

            # Model prediction wrapper: reconstruct dicts from flat arrays
            def model_predict_wrapper(X):
                obs_dicts = []
                for row in X:
                    obs_dict = {}
                    idx = 0
                    for k in obs_keys:
                        space = env.observation_space.spaces[k]
                        size = int(np.prod(space.shape))
                        obs_dict[k] = row[idx:idx+size].reshape(space.shape)
                        idx += size
                    obs_dicts.append(obs_dict)
                # Batch predict
                if len(obs_dicts) == 1:
                    actions, _ = model.predict(obs_dicts[0], deterministic=True)
                    return actions[None, :]
                else:
                    obs_dict_batch = {k: np.stack([obs[k] for obs in obs_dicts]) for k in obs_keys}
                    actions, _ = model.predict(obs_dict_batch, deterministic=True)
                    return actions

            # Always compute baseline for each action
            baseline_actions = model_predict_wrapper(background_sample).mean(axis=0)
            print("Baseline (expected) value for each action:", baseline_actions)
        else:
            print("Could not find SHAP background flat file in MLflow artifacts. Baseline will be zeros.")
    except Exception as e:
        print(f"[Warning] Could not load SHAP background or compute baseline: {e}")
        background_sample = None
        model_predict_wrapper = None
        baseline_actions = None

    # --- Only setup SHAP explainer if requested ---
    shap_explainer = None
    if plot_shap and background_sample is not None and model_predict_wrapper is not None:
        try:
            shap_explainer = shap.KernelExplainer(model_predict_wrapper, background_sample)
            print("SHAP explainer ready.")
        except Exception as e:
            print(f"Failed to setup SHAP explainer: {e}")
            shap_explainer = None



    # Optionally wait for user confirmation before initial homing
    input("Press Enter to home the robot...")

    # Create RealRobot instance for managing robot communication
    real_robot = RealRobot(config, arduino)

    # 4. Reset environment
    obs = env.reset()
    done = False
    step_count = 0

    # Wait for user confirmation before syncing to the virtual environment
    input("Press Enter to synchronize the real robot with the virtual environment...")

    # synchronise the real robot
    print("Synchronizing real robot with environment...")
    real_robot.set_robot_position(env.envs[0].unwrapped.robot.extensions)

    # --- Setup live display windows with enhanced plotting ---
    plt.ion()
    
    # Create figure with larger size and better layout
    # Adjust layout based on whether SHAP is enabled
    if plot_shap:
        n_shap = 5  # Number of SHAP heatmaps (actions)
        # New layout: SHAPs (5), obs heatmap (1), action history (2 rows), total = 8 rows
        n_rows = n_shap + 1  # 6
        n_cols = 2           # 2 columns: SHAPs/obs | Everything else
        fig = plt.figure(figsize=(16, 32))
        # Height ratios: 1 for each SHAP, 1 for obs heatmap, 2 for action history
        height_ratios = [1]*n_shap + [2]  # 5 SHAP + 1 (double height for obs heatmap/action history)
        gs = gridspec.GridSpec(n_rows, n_cols, height_ratios=height_ratios, width_ratios=[1, 1], hspace=0.4, wspace=0.3)

        # SHAP heatmaps: column 0, rows 0-4
        ax_shap = []
        for i in range(n_shap):
            ax_shap.append(fig.add_subplot(gs[i, 0]))
        # Observation heatmap: column 0, row 5
        ax_obs = fig.add_subplot(gs[n_shap, 0])

        # Column 1:
        # Overview camera: spans rows 0 and 1 in column 1
        ax_overview = fig.add_subplot(gs[0:2, 1])
        # Robot perspective: spans rows 2 and 3 in column 1 (double height)
        ax_robot = fig.add_subplot(gs[2:4, 1])
        # Action history: spans rows 4-5 in column 1 (double height)
        ax_action = fig.add_subplot(gs[4:6, 1])
        # (No obs heatmap in col 1)
    else:
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 1], hspace=0.25, wspace=0.25)
        # Camera frames (top row, larger)
        ax_overview = fig.add_subplot(gs[0, 0])
        ax_robot = fig.add_subplot(gs[0, 1])
        # Action history (top right)
        ax_action = fig.add_subplot(gs[0, 2])
        # Observation heatmap (bottom row, spans all columns)
        ax_obs = fig.add_subplot(gs[1, :])
    
    # Get initial images for both cameras
    try:
        overview_img = env.envs[0].unwrapped.render(camera_name='scene_overview')
        robot_img = env.envs[0].unwrapped.render(camera_name='robot_perspective')
    except Exception:
        overview_img = robot_img = None

    # Setup camera displays
    if overview_img is not None:
        img_overview = ax_overview.imshow(overview_img)
    else:
        img_overview = ax_overview.imshow([[0]])
    ax_overview.set_title('Overview Camera')
    ax_overview.axis('off')
    
    if robot_img is not None:
        img_robot = ax_robot.imshow(robot_img)
    else:
        img_robot = ax_robot.imshow([[0]])
    ax_robot.set_title('Robot Perspective')
    ax_robot.axis('off')
    
    # Setup action history tracking
    max_history = 15  # Keep last 100 steps
    action_history = deque(maxlen=max_history)
    step_history = deque(maxlen=max_history)
    
    # Setup observation history tracking
    obs_keys = list(env.observation_space.spaces.keys()) if hasattr(env.observation_space, 'spaces') else ['obs']
    obs_history = deque(maxlen=max_history)  # Store flattened observations
    
    # Calculate total observation dimension for the heatmap
    if hasattr(env.observation_space, 'spaces'):
        obs_dims = []
        obs_labels = []
        for key in obs_keys:
            space = env.observation_space.spaces[key]
            if hasattr(space, 'shape'):
                dim = np.prod(space.shape)
                obs_dims.append(dim)
                if dim == 1:
                    obs_labels.append(key)
                else:
                    obs_labels.extend([f"{key}_{i}" for i in range(dim)])
            else:
                obs_dims.append(1)
                obs_labels.append(key)
        total_obs_dim = sum(obs_dims)
    else:
        total_obs_dim = np.prod(env.observation_space.shape) if hasattr(env.observation_space, 'shape') else 1
        obs_labels = [f"obs_{i}" for i in range(total_obs_dim)]
    
    # Initialize action plot - dynamically determine action space structure
    action_lines = []
    if hasattr(env.action_space, 'spaces'):
        # Dict action space
        action_keys = list(env.action_space.spaces.keys())
        action_labels = []
        for key in action_keys:
            space = env.action_space.spaces[key]
            if hasattr(space, 'shape'):
                dim = np.prod(space.shape)
                if dim == 1:
                    action_labels.append(key)
                else:
                    action_labels.extend([f"{key}_{i}" for i in range(dim)])
            else:
                action_labels.append(key)
        action_dim = len(action_labels)
    else:
        # Box action space - use specific robot action labels
        if hasattr(env.action_space, 'shape'):
            action_dim = np.prod(env.action_space.shape)
        else:
            action_dim = 1
        
        # Custom labels for robot actions: x, y, rx, rz, grip
        if action_dim == 5:
            action_labels = ['Delta X', 'Delta Y', 'Delta RX', 'Delta RZ', 'Grip Signal']
        else:
            action_labels = [f"Action_{i}" for i in range(action_dim)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))
    action_lines = []
    baseline_lines = []
    baseline_shades = []
    for i in range(action_dim):
        line, = ax_action.plot([], [], color=colors[i], label=action_labels[i])
        action_lines.append(line)
        # Plot baseline as a horizontal dashed line
        if baseline_actions is not None:
            baseline_line = ax_action.axhline(baseline_actions[i], color=colors[i], linestyle='--', linewidth=1, alpha=0.5)
        else:
            baseline_line = ax_action.axhline(0, color=colors[i], linestyle='--', linewidth=1, alpha=0.5)
        baseline_lines.append(baseline_line)
        # Initialize shaded region (empty for now)
        shade = ax_action.fill_between([], [], [], color=colors[i], alpha=0.15)
        baseline_shades.append(shade)
    ax_action.set_title('Action History (Including Planned & Baseline)')
    ax_action.set_xlabel('Step')
    ax_action.set_ylabel('Action Value')
    ax_action.legend(loc='upper left')
    ax_action.grid(True, alpha=0.3)
    ax_action.set_ylim(-1.1, 1.1)  # Allow for negative values
    ax_action.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Initialize action history with the baseline action at step 0
    if baseline_actions is not None and len(baseline_actions) == action_dim:
        action_history.append(np.array(baseline_actions))
    else:
        action_history.append(np.zeros(action_dim))
    step_history.append(0)
    
     # --- Initialize observation heatmap (for current obs only) ---
    # Build a 2D structure: horizontally for each key, vertically for each value in the key
    obs_heatmap_labels = []  # List of (key, idx) tuples for y-axis
    obs_heatmap_xlabels = []  # List of keys for x-axis
    obs_heatmap_col_offsets = []  # Start index for each key in the flat obs
    obs_heatmap_shape = []  # Number of rows for each key
    flat_idx = 0
    if hasattr(env.observation_space, 'spaces'):
        for key in obs_keys:
            space = env.observation_space.spaces[key]
            if hasattr(space, 'shape') and np.prod(space.shape) > 1:
                obs_heatmap_labels.extend([(key, i) for i in range(np.prod(space.shape))])
                obs_heatmap_xlabels.append(key)
                obs_heatmap_col_offsets.append(flat_idx)
                obs_heatmap_shape.append(np.prod(space.shape))
                flat_idx += np.prod(space.shape)
            else:
                obs_heatmap_labels.append((key, 0))
                obs_heatmap_xlabels.append(key)
                obs_heatmap_col_offsets.append(flat_idx)
                obs_heatmap_shape.append(1)
                flat_idx += 1
        total_obs_dim = flat_idx
    else:
        # Single Box obs
        total_obs_dim = np.prod(env.observation_space.shape) if hasattr(env.observation_space, 'shape') else 1
        obs_heatmap_labels = [('obs', i) for i in range(total_obs_dim)]
        obs_heatmap_xlabels = ['obs']
        obs_heatmap_col_offsets = [0]
        obs_heatmap_shape = [total_obs_dim]

    # Build a 2D array for the heatmap: rows = max vertical stack, cols = number of keys
    max_rows = max(obs_heatmap_shape)
    n_cols = len(obs_heatmap_xlabels)
    obs_heatmap_data = np.full((max_rows, n_cols), np.nan)

    # Set up the heatmap and colorbar (scale)
    heatmap_img = ax_obs.imshow(obs_heatmap_data, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    cbar = plt.colorbar(heatmap_img, ax=ax_obs, orientation='vertical', pad=0.01, aspect=30)
    cbar.set_label('Normalized Value')
    ax_obs.set_title('Current Normalized Observation')
    # Set x-ticks to keys
    ax_obs.set_xticks(np.arange(n_cols))
    ax_obs.set_xticklabels(obs_heatmap_xlabels, rotation=30, ha='right')
    # Set y-ticks to indices (0 to max_rows-1)
    ax_obs.set_yticks(np.arange(max_rows))
    ax_obs.set_yticklabels([str(i) for i in range(max_rows)])
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    
    # --- Initialize SHAP heatmaps if enabled ---
    shap_heatmap_imgs = []
    if plot_shap and shap_explainer is not None:
        for i, ax in enumerate(ax_shap):
            # Initialize empty SHAP heatmap data
            shap_data = np.full((max_rows, n_cols), np.nan)
            img = ax.imshow(shap_data, aspect='auto', cmap='bwr', vmin=-1, vmax=1)
            shap_heatmap_imgs.append(img)
            # Set SHAP heatmap title color to match action line color
            title_text = str(action_labels[i]) + ' Determinants' if i < len(action_labels) else f'Action {i}'
            ax.set_title(title_text, color=colors[i])
            # Add ticks (but no labels) for SHAP heatmaps
            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels([''] * n_cols)
            ax.set_yticks(np.arange(max_rows))
            ax.set_yticklabels([''] * max_rows)
            # Add colorbar
            cbar_shap = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.01, aspect=20)
            cbar_shap.set_label('Importance')
    
    # Add overall figure title (will update per step)
    fig.suptitle(f'RL ANALYSIS (Step 0)', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.show(block=False)

    input("Press Enter to start the episode...")

    print("\n--- Running one episode ---")
    while not done:


        # 5. Get action from model (planned action)
        action, _ = model.predict(obs, deterministic=True)

        # --- Show planned action as a star marker on the action history plot ---
        # Handle action flattening based on action space type
        if hasattr(env.action_space, 'spaces'):
            # Dict action space - flatten all components
            flat_action = np.concatenate([action[k].flatten() for k in action_keys if k in action])
        else:
            # Box action space - just flatten
            flat_action = action.flatten() if hasattr(action, 'flatten') else np.array([action])

        # Plot planned action as a dotted line from the last action to the planned action
        planned_step = step_count + 1
        # Remove previous planned lines if any
        if hasattr(ax_action, '_planned_lines'):
            for l in ax_action._planned_lines:
                l.remove()
        ax_action._planned_lines = []
        # Get last action (if available)
        if len(action_history) > 0:
            last_action = action_history[-1]
            last_step = step_history[-1]
            for i, line in enumerate(action_lines):
                if i < flat_action.shape[0]:
                    # Plot a dotted line from last action to planned action
                    planned_line, = ax_action.plot([last_step, planned_step], [last_action[i], flat_action[i]], linestyle='dotted', color=line.get_color(), linewidth=3, zorder=5)
                    ax_action._planned_lines.append(planned_line)


        # Get current (normalized) observation as flat array
        if isinstance(obs, dict):
            flat_obs = np.concatenate([obs[k].flatten() for k in obs_keys if k in obs])
        else:
            flat_obs = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])

        # --- Live update of camera views ---
        try:
            overview_img = env.envs[0].unwrapped.render(camera_name='scene_overview')
            robot_img = env.envs[0].unwrapped.render(camera_name='robot_perspective')
            img_overview.set_data(overview_img)
            img_robot.set_data(robot_img)

        except Exception as e:
            print(f"[Warning] Could not render camera views: {e}")

        # --- Update action history plot ---
        if len(action_history) > 1:
            try:
                steps_array = np.array(list(step_history))
                actions_array = np.array(list(action_history))  # Already flattened actions
                print(f"Debug: steps: {steps_array.shape}, actions: {actions_array.shape}")
                # Ensure exactly matching lengths
                min_len = min(len(steps_array), len(actions_array))
                steps_array = steps_array[:min_len]
                actions_array = actions_array[:min_len]
                for i, line in enumerate(action_lines):
                    if i < actions_array.shape[1]:
                        line.set_data(steps_array, actions_array[:, i])
                        # Remove previous shade
                        if baseline_shades[i]:
                            try:
                                baseline_shades[i].remove()
                            except Exception:
                                pass
                        # Shade between baseline and action
                        if baseline_actions is not None:
                            baseline_y = np.full_like(steps_array, baseline_actions[i])
                        else:
                            baseline_y = np.zeros_like(steps_array)
                        baseline_shades[i] = ax_action.fill_between(
                            steps_array, baseline_y, actions_array[:, i],
                            color=colors[i], alpha=0.15
                        )
                ax_action.set_xlim(max(0, step_count - max_history), step_count + 1)
                ax_action.set_ylim(-1.1, 1.1)  # Allow for negative values
            except Exception as e:
                print(f"[Warning] Could not update action plot: {e}")
                import traceback
                traceback.print_exc()

        # --- Update current observation heatmap ---
        try:
            # Reset heatmap data
            obs_heatmap_data[:, :] = np.nan
            flat_idx = 0
            for col, (key, n_rows) in enumerate(zip(obs_heatmap_xlabels, obs_heatmap_shape)):
                if hasattr(env.observation_space, 'spaces'):
                    # Dict obs
                    if isinstance(obs, dict) and key in obs:
                        val = obs[key].flatten() if hasattr(obs[key], 'flatten') else np.array([obs[key]])
                        obs_heatmap_data[:n_rows, col] = val[:n_rows]
                        flat_idx += n_rows
                else:
                    # Box obs
                    obs_heatmap_data[:n_rows, col] = flat_obs[flat_idx:flat_idx + n_rows]
                    flat_idx += n_rows
            heatmap_img.set_data(obs_heatmap_data)
            # Remove previous text overlays
            for txt in ax_obs.texts:
                txt.remove()
            # Optionally add value text overlay for small obs spaces
            if total_obs_dim <= 20:
                for col in range(n_cols):
                    for row in range(obs_heatmap_shape[col]):
                        val = obs_heatmap_data[row, col]
                        if not np.isnan(val):
                            ax_obs.text(col, row, f"{val:.2f}", ha='center', va='center', color='black')
            # No need to reset axis labels every frame unless obs space changes
        except Exception as e:
            print(f"[Warning] Could not update observation heatmap: {e}")
            traceback.print_exc()

        # --- Update SHAP heatmaps if enabled ---
        if plot_shap and shap_explainer is not None:
            try:
                # Prepare observation for SHAP (flattened format)
                if isinstance(obs, dict):
                    flat_obs_for_shap = np.concatenate([obs[k].flatten() for k in obs_keys if k in obs])
                else:
                    flat_obs_for_shap = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])
                # --- SPEEDUP: Reduce number of SHAP samples per explanation ---
                NSAMPLES_SHAP = 40  # <<<< Reduce this for speed (default is much higher)
                shap_vals = shap_explainer.shap_values(flat_obs_for_shap.reshape(1, -1), nsamples=NSAMPLES_SHAP)

                print("shap_vals type:", type(shap_vals))
                print("shap_vals shape:", np.shape(shap_vals))

                # Handle known (1, obs_dim, action_dim) shape directly
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[0] == 1:
                    for i, (ax, img) in enumerate(zip(ax_shap, shap_heatmap_imgs)):
                        if i < shap_vals.shape[2]:
                            shap_vals_i = shap_vals[0, :, i]  # shape: (obs_dim,)
                            # Build a 2D array for the SHAP heatmap (same structure as obs heatmap)
                            shap_heatmap_data = np.full((max_rows, n_cols), np.nan)
                            flat_idx = 0
                            for col, n_rows in enumerate(obs_heatmap_shape):
                                if flat_idx + n_rows <= len(shap_vals_i):
                                    shap_slice = shap_vals_i[flat_idx:flat_idx + n_rows]
                                    shap_heatmap_data[:n_rows, col] = shap_slice.flatten()[:n_rows]
                                flat_idx += n_rows

                            # Update the heatmap image
                            max_abs_val = np.nanmax(np.abs(shap_heatmap_data))
                            if max_abs_val > 0:
                                img.set_clim(-max_abs_val, max_abs_val)
                            img.set_data(shap_heatmap_data)

                            # Remove previous text overlays
                            for txt in ax.texts:
                                txt.remove()

                            # Optionally add value text overlay for small obs spaces
                            if total_obs_dim <= 20:
                                for col in range(n_cols):
                                    for row in range(obs_heatmap_shape[col]):
                                        val = shap_heatmap_data[row, col]
                                        if not np.isnan(val):
                                            color = 'white' if abs(val) > max_abs_val * 0.5 else 'black'
                                            ax.text(col, row, f"{val:.2f}", ha='center', va='center', color=color)
            except Exception as e:
                print(f"[Warning] Could not update SHAP heatmaps: {e}")
                traceback.print_exc()

        # Redraw the figure
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception as e:
            print(f"[Warning] Could not redraw figure: {e}")

        # Optionally wait for user confirmation before next step
        if wait_for_user:
            input("Press Enter to continue to the next step...")

        # 6. Step environment
        obs, reward, done, info = env.step(action)
        step_count += 1
        fig.suptitle(f'RL ANALYSIS (Step {step_count})', fontweight='bold')

        action_history.append(flat_action)
        step_history.append(step_count)
        

        # Send the extensions calculated by the robot kinematics to the real robot
        real_robot.set_robot_position(env.envs[0].unwrapped.robot.extensions)
        
        if (env.envs[0].unwrapped.robot.gripped and not real_robot.gripped):
            # If the robot is gripped in the environment but not in the real robot, grip it
            real_robot.set_gripper_state('GRIP')

        elif (not env.envs[0].unwrapped.robot.gripped and real_robot.gripped):
            # If the robot is not gripped in the environment but is in the real robot, ungrip it
            real_robot.set_gripper_state('UNGRIP')




    # Optionally rehome at the end of the episode
    if wait_for_user:
        input("Press Enter to rehome the robot at the end of the episode...")
    print("Rehoming robot...")
    real_robot.arduino.write(('HOME\n').encode('utf-8'))
    while True:
        try:
            response = real_robot.arduino.readline().decode('utf-8').strip()
            if response:
                print(f"Arduino says: {response}")
                if response == "DONE":
                    break
        except UnicodeDecodeError:
            print("Arduino says: (Could not decode response)")

    print("Episode finished. Closing serial connection.")
    real_robot.close()
    
    # Clean up run-specific temp directory
    temp_dir_shap = os.path.join("C:/temp/artifacts", f"run_{mlflow_run_id}")
    if os.path.exists(temp_dir_shap):
        import shutil
        shutil.rmtree(temp_dir_shap)
        print(f"Cleaned up temporary directory: {temp_dir_shap}")
    
    print("The plot window will remain open until you close it manually.")
    plt.ioff()
    plt.show(block=True)
    # Do not call plt.close(fig) here; let the user close the window manually



if __name__ == "__main__":
    main()


