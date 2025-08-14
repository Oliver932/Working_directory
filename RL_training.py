import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import glob
import shutil
from collections import deque
import json

# MLFLOW: Import the mlflow library
import mlflow

# Set MLflow tracking URI to avoid OneDrive permission issues
mlflow.set_tracking_uri("file:///C:/temp/mlruns")

# Import custom modules for the robot simulation
from arm_ik_model import RobotKinematics, Ring
from collision_and_render_management import CollisionAndRenderManager
from ring_projector_simplified import RingProjectorSimplified
from overview_render_manager import OverviewRenderManager
from ring_placement import set_random_pose_box_limit

# Import Stable Baselines3 components for reinforcement learning
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==================================================================================
# == Custom Metrics Callback                                                      ==
# ==================================================================================
class CustomMetricsCallback(BaseCallback):
    """
    A custom callback to log domain-specific metrics to TensorBoard and MLflow.
    """
    def __init__(self, window_size=100, verbose=0):
        super().__init__(verbose)
        self.success_rate_window = deque(maxlen=window_size)
        self.collision_episode_window = deque(maxlen=window_size)
        self.invalid_move_prop_window = deque(maxlen=window_size)
        self.failed_grip_rate_window = deque(maxlen=window_size)
        self.avg_difficulty_window = deque(maxlen=window_size)
        self.ellipse_visible_prop_window = deque(maxlen=window_size)
        # Track whether episodes ended due to truncation (e.g., max steps reached)
        self.truncated_episode_window = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # Only log metrics at the end of an episode (when 'episode' info is present)
        if self.locals.get("infos")[0].get("episode"):
            info = self.locals.get("infos")[0]
            ep_len = info["episode"]["l"]

            # --- Success Rate: Proportion of successful episodes ---
            self.success_rate_window.append(info.get("is_success", 0))
            success_rate = np.mean(self.success_rate_window)
            self.logger.record("custom/success_rate", success_rate)

            # --- Collision Rate: Proportion of episodes with at least one collision ---
            had_collision = 1 if info.get("is_collision", 0) > 0 else 0
            self.collision_episode_window.append(had_collision)
            collision_rate = np.mean(self.collision_episode_window)
            self.logger.record("custom/collision_rate", collision_rate)

            # --- Invalid Move Proportion: Proportion of steps in an episode that were invalid moves ---
            num_invalid_moves = info.get("is_invalid_move", 0)
            invalid_move_prop = num_invalid_moves / ep_len if ep_len > 0 else 0
            self.invalid_move_prop_window.append(invalid_move_prop)
            avg_invalid_prop = np.mean(self.invalid_move_prop_window)
            self.logger.record("custom/invalid_move_proportion", avg_invalid_prop)

            # --- Failed Grip Rate: Proportion of grip attempts that fail ---
            total_grip_attempts = info.get("grip_attempts", 0)
            total_failed_grips = info.get("failed_grips", 0)
            failure_rate = total_failed_grips / total_grip_attempts if total_grip_attempts > 0 else 0.0
            self.failed_grip_rate_window.append(failure_rate)
            avg_failed_grip = np.mean(self.failed_grip_rate_window)
            self.logger.record("custom/failed_grip_rate", avg_failed_grip)

            # --- Average Difficulty: The average spawn difficulty of episodes ---
            self.avg_difficulty_window.append(info.get("difficulty", 0))
            avg_difficulty = np.mean(self.avg_difficulty_window)
            self.logger.record("custom/avg_difficulty", avg_difficulty)

            # --- Ellipse Visible Proportion ---
            visible_steps = info.get("visible_steps", 0)
            self.ellipse_visible_prop_window.append(visible_steps / ep_len if ep_len > 0 else 0)
            avg_visible_prop = np.mean(self.ellipse_visible_prop_window)
            self.logger.record("custom/ellipse_visible_proportion", avg_visible_prop)

            # --- Truncated Episode Proportion: Proportion of episodes ending due to truncation (max steps) ---
            self.truncated_episode_window.append(int(info.get("is_truncated", False)))
            truncated_prop = np.mean(self.truncated_episode_window)
            self.logger.record("custom/truncated_episode_proportion", truncated_prop)

        return True

# ==================================================================================
# == Curriculum Trainer Callback                                                  ==
# ==================================================================================
class CurriculumTrainerCallback(BaseCallback):
    """
    A callback to implement curriculum learning. It adjusts the environment's
    difficulty based on performance relative to two success rate thresholds.
    """
    def __init__(self, metrics_callback: CustomMetricsCallback, increase_threshold: float, decrease_threshold: float, check_freq: int, 
                 difficulty_step: float = 0.1, min_difficulty: float = 0.0, max_difficulty: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.metrics_callback = metrics_callback
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.check_freq = check_freq
        self.difficulty_step = difficulty_step
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.last_difficulty_check_step = 0

    def _on_step(self) -> bool:
        # Check if we should evaluate difficulty at the specified interval
        if (self.n_calls - self.last_difficulty_check_step) >= self.check_freq and len(self.metrics_callback.success_rate_window) > 0:
            current_success_rate = np.mean(self.metrics_callback.success_rate_window)
            current_difficulty = self.training_env.env_method('get_difficulty')[0]
            
            if current_success_rate > self.increase_threshold:
                # Performance is too good, increase difficulty
                new_difficulty = min(current_difficulty + self.difficulty_step, self.max_difficulty)
                if new_difficulty != current_difficulty:
                    self.training_env.env_method('set_difficulty', new_difficulty)
                    if self.verbose > 0:
                        print(f"--- Increasing Difficulty: Success rate {current_success_rate:.2f} > {self.increase_threshold:.2f}, "
                              f"difficulty {current_difficulty:.2f} -> {new_difficulty:.2f} ---")
                    self.logger.record("curriculum/difficulty_increased", new_difficulty)
                    
            elif current_success_rate < self.decrease_threshold:
                # Performance is too poor, decrease difficulty
                new_difficulty = max(current_difficulty - self.difficulty_step, self.min_difficulty)
                if new_difficulty != current_difficulty:
                    self.training_env.env_method('set_difficulty', new_difficulty)
                    if self.verbose > 0:
                        print(f"--- Decreasing Difficulty: Success rate {current_success_rate:.2f} < {self.decrease_threshold:.2f}, "
                              f"difficulty {current_difficulty:.2f} -> {new_difficulty:.2f} ---")
                    self.logger.record("curriculum/difficulty_decreased", new_difficulty)
            else:
                # Performance is in the stable zone, no change needed
                if self.verbose > 0:
                    print(f"--- Difficulty Stable: Success rate {current_success_rate:.2f} between {self.decrease_threshold:.2f} and {self.increase_threshold:.2f}, "
                          f"maintaining difficulty {current_difficulty:.2f} ---")
                    
            # Always log current difficulty level
            self.logger.record("curriculum/current_difficulty", current_difficulty)
            self.last_difficulty_check_step = self.n_calls
            
        return True

# ==================================================================================
# == Custom Robot Environment                                                     ==
# ==================================================================================
class CustomRobotEnv(gym.Env):
    """A custom Gymnasium environment for a robot arm tasked with gripping a ring."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode='rgb_array', config: dict = None):
        super().__init__()
        
        if config is None:
            raise ValueError("A configuration dictionary must be provided to the environment.")
        self.config = config

        # --- Unpack Config Values ---
        self.MAX_STEPS_PER_EPISODE = self.config["max_steps"]
        self.rewards = self.config["rewards"]
        self.multipliers = self.config["multipliers"]
    
        
        paths = self.config["paths"]
        camera_settings = self.config["camera"]
        overview_camera_settings = self.config["overview_camera"]

        # --- Define Observation and Action Spaces ---
        self.observation_space = spaces.Dict({
            # Ellipse position relative to G1 position
            "ellipse_position_relative": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "delta_ellipse_position_relative": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),

            # Ellipse shape and orientation (new simplified format)
            "ellipse_major_axis_norm": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "delta_ellipse_major_axis_norm": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "ellipse_aspect_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "delta_ellipse_aspect_ratio": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "ellipse_orientation_2d": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "delta_ellipse_orientation_2d": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),

            # "actuator_extensions": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            # "delta_extensions": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),

            "relative_G1_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "G1_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "approach_vec": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "radial_vec": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "G1_angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        })


        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        # --- Initialize Environment State and Components ---
        self.current_step = 0
        self.render_mode = render_mode
        robot_verbosity = self.config.get("robot", {}).get("verbosity", 0)
        self.robot = RobotKinematics(verbosity=robot_verbosity)
        self.ring = Ring()
        
        # --- Difficulty Setup ---
        # Start with the minimum difficulty from curriculum config if available
        curriculum_config = self.config.get("curriculum", {})
        min_difficulty = curriculum_config.get("min_difficulty", 0.0)
        self.difficulty = min_difficulty  # Start with minimum difficulty

        
        self.ring_projector = RingProjectorSimplified(vertical_fov_deg=camera_settings["fov"], image_width=camera_settings["width"], image_height=camera_settings["height"])
        self.collision_render_manager = CollisionAndRenderManager(paths["gripper_col"], paths["gripper_col"], paths["ring_render"], paths["ring_col"], vertical_FOV=camera_settings["fov"], render_width=camera_settings["width"], render_height=camera_settings["height"])
        self.overview_render_manager = OverviewRenderManager(self.robot, self.ring, paths["ring_render"], width=overview_camera_settings["width"], height=overview_camera_settings["height"])
        
        self._setup_episode()


    def _setup_episode(self):
        # Set the robot to a random pose with difficulty from 0.0 to 1.0
        success, pose, actual_difficulty = set_random_pose_box_limit(self.robot, self.ring, self.collision_render_manager, difficulty=self.difficulty)

        if not success:
            raise ValueError("Failed to set a valid random pose for the robot and ring. Please check the configuration and constraints.")
        
        # Use the actual calculated difficulty for logging purposes
        self.episode_difficulty = actual_difficulty

        # setup tracking for improvements
        self.ideal_relative_G1 = self.robot.G1_relative_position.copy()
        self.ideal_rotation_vector = np.column_stack([self.robot.approach_vec, self.robot.radial_vec, self.robot.tangential_vec])

        self.current_G1_closeness = np.zeros(1, dtype=np.float32)
        self.current_G1_angular_closeness = np.zeros(1, dtype=np.float32)

        self.robot.go_home()
        self.collision_render_manager.update_poses(self.robot, self.ring)
        self.ring_projector.update(self.robot, self.ring)


    def get_difficulty(self):
        """Get the current difficulty level (0.0 to 1.0)"""
        return self.difficulty

    def set_difficulty(self, new_difficulty):
        """Set the difficulty level (0.0 to 1.0)"""
        self.difficulty = np.clip(new_difficulty, 0.0, 1.0)
        if hasattr(self, 'verbose') and getattr(self, 'verbose', 0) > 0:
            print(f"DIFFICULTY SET TO {self.difficulty:.2f}")

    def adjust_difficulty(self, delta):
        """Adjust difficulty by a delta amount"""
        self.set_difficulty(self.difficulty + delta)

    def _get_obs(self):
        
        ellipse_details = self.ring_projector.projected_properties

        # Calculate relative positions (ellipse relative to G1)
        observations = {
            # Ellipse position relative to G1
            "ellipse_position_relative": ellipse_details['position_2d'] - ellipse_details['g1_position_2d'],
            "delta_ellipse_position_relative": ellipse_details['delta_position_2d'] - ellipse_details['delta_g1_position_2d'],
            
            # Ellipse shape and orientation (new simplified format)
            "ellipse_major_axis_norm": np.array([ellipse_details['major_axis_norm']], dtype=np.float32),
            "delta_ellipse_major_axis_norm": np.array([ellipse_details['delta_major_axis_norm']], dtype=np.float32),
            "ellipse_aspect_ratio": np.array([ellipse_details['aspect_ratio']], dtype=np.float32),
            "delta_ellipse_aspect_ratio": np.array([ellipse_details['delta_aspect_ratio']], dtype=np.float32),
            "ellipse_orientation_2d": ellipse_details['orientation_2d'],
            "delta_ellipse_orientation_2d": ellipse_details['delta_orientation_2d'],

            # "actuator_extensions": self.robot.extensions,
            # "delta_extensions": self.robot.delta_extensions,

            "relative_G1_position": self.robot.G1_relative_position,
            "G1_velocity": self.robot.G1_velocity,

            "approach_vec": self.robot.approach_vec,
            "radial_vec": self.robot.radial_vec,
            "G1_angular_velocity": self.robot.G1_angular_velocity

        }

        return observations

    def _get_info(self):
        # This is now only called when the episode ends
        return {
            "is_success": self.is_success,
            "is_collision": self.is_collision,
            "is_truncated": self.is_truncated,
            "is_invalid_move": self.is_invalid_move,
            "grip_attempts": self.grip_attempts,
            "failed_grips": self.failed_grips,
            "difficulty": self.episode_difficulty,
            "visible_steps": self.visible_steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset episode-specific trackers
        self.is_success = False
        self.is_collision = False
        self.is_truncated = False
        self.is_invalid_move = 0
        self.grip_attempts = 0
        self.failed_grips = 0
        self.visible_steps = 0
        
        self._setup_episode()

        return self._get_obs(), {} # Return empty info dict on reset

    def step(self, action):
        self.current_step += 1
        terminated = False
        reward = 0.0
        
        pose_deltas = action[0:4]
        should_grip = action[4] > 0.0

        if should_grip:
            
            #update robot gripped state
            self.robot.gripped = True

            self.grip_attempts += 1
            
            if self.robot.evaluate_grip(self.ring):
                reward += self.rewards["success"]
                terminated = True
                self.is_success = True
            else:
                reward += self.rewards["fail_grip"]
                self.failed_grips += 1
                # Check for collision even after failed grip
                self.collision_render_manager.update_poses(self.robot, self.ring)
                if self.collision_render_manager.check_collision():
                    reward += self.rewards["collision"]
                    self.is_collision = True
                    terminated = True

                terminated = True  # NEW ADDITION END ON FAILED GRIP
        else:

            # update robot gripped state
            self.robot.gripped = False

        if not terminated:
            self.robot.move_E1(
                pose_deltas[0] * self.multipliers["linear"], 
                pose_deltas[1] * self.multipliers["linear"], 
                pose_deltas[2] * self.multipliers["rx"], 
                pose_deltas[3] * self.multipliers["rz"]
            )
            self.ring_projector.update(self.robot, self.ring)
            
            # Update visibility counter
            ellipse_details = self.ring_projector.projected_properties
            if ellipse_details.get('visible', False):
                self.visible_steps += 1

            self.collision_render_manager.update_poses(self.robot, self.ring)
            if self.collision_render_manager.check_collision():
                reward += self.rewards["collision"]
                self.is_collision = True
                terminated = True


        # Reward movement towards ideal position
        TRANSLATION_PRECISION_FACTOR = 1 # can be neatly set to determine the proportion of max translation reward gained at the tolerance boundary
        TRANSLATION_REWARD_SCALING = 1
        prev_G1_closeness = self.current_G1_closeness.copy()
        # convert the current distance into the stationary rings frame
        current_G1_dist = self.robot.G1_relative_position - self.ideal_relative_G1
        current_G1_dist_ring_frame = self.ideal_rotation_vector.T @ current_G1_dist
        # we scale the closeness based on the tolerances for each axis
        translation_tolerances = np.asarray([19.68666481971737/2, 44.19397354125971/2, 57.67107772827144/2])
        # use a gaussian a smoth closeness (1 when in the same place, 0 when infinitely apart)
        self.current_G1_closeness = np.exp(np.sum(-TRANSLATION_PRECISION_FACTOR * np.pow(current_G1_dist_ring_frame / translation_tolerances, 2)))
        # add the progress towards closeness 1 to the reward.
        reward += TRANSLATION_REWARD_SCALING * (self.current_G1_closeness - prev_G1_closeness)

        # reward movement towards ideal orientation
        ROTATION_PRECISION_FACTOR = 1 # can be neatly set to determine the proportion of max rotation reward gained at the tolerance boundary
        ROTATION_REWARD_SCALING = 1
        prev_G1_angular_closeness = self.current_G1_angular_closeness.copy()
        current_rotation_vector = np.column_stack([self.robot.approach_vec, self.robot.radial_vec, self.robot.tangential_vec])
        # Relative rotation matrix
        R_rel = current_rotation_vector @ self.ideal_rotation_vector.T
        # Angle of rotation (in radians)
        angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        angular_tolerance = 10 #DEGREES
        self.current_G1_angular_closeness = np.exp(-ROTATION_PRECISION_FACTOR * np.pow(angle_deg / angular_tolerance, 2))
        reward += ROTATION_REWARD_SCALING * (self.current_G1_angular_closeness - prev_G1_angular_closeness)

        if not self.robot.last_solve_successful:
            reward += self.rewards["fail_move"]
            self.is_invalid_move += 1

        # Determine episode end state with mutual exclusivity (collision > success > truncation)
        truncated = False
        if not terminated:  # Only check for truncation if not already terminated
            if self.current_step >= self.MAX_STEPS_PER_EPISODE:
                self.is_truncated = True
                truncated = True
        
        info = {}
        if terminated or truncated:
            # Only populate the info dict at the end of the episode
            info = self._get_info()
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, camera_name='scene_overview'):
        if self.render_mode == 'rgb_array':
            if camera_name == 'scene_overview':
                return self.overview_render_manager.render_to_image()
            elif camera_name == 'robot_perspective':
                return self.collision_render_manager.render()
            else:
                raise ValueError(f"Unknown camera name: {camera_name}")
        return None

    def close(self):
        pass

def load_config(config_path="config.json"):
    """Load configuration from JSON file with error handling."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the configuration file exists.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing config file: {e}. Please check the JSON syntax in {config_path}.")

# MLFLOW: Helper function to flatten the config dictionary for logging
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def import_tensorboard_to_mlflow(tensorboard_log_dir):
    """Import TensorBoard event files to MLflow metrics."""
    try:
        print("--- Importing TensorBoard metrics to MLflow ---")
        
        # Try TensorBoard package first (lighter dependency, comes with Stable Baselines3)
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            event_files = glob.glob(os.path.join(tensorboard_log_dir, "**", "events.out.tfevents.*"), recursive=True)
            
            if not event_files:
                print("No TensorBoard event files found!")
                return
            
            total_metrics = 0
            for event_file in event_files:
                try:
                    ea = EventAccumulator(event_file)
                    ea.Reload()
                    
                    # Get scalar summaries
                    scalar_tags = ea.Tags().get('scalars', [])
                    
                    for tag in scalar_tags:
                        scalar_events = ea.Scalars(tag)
                        for event in scalar_events:
                            metric_name = f"tb_{tag.replace('/', '_')}"
                            mlflow.log_metric(metric_name, event.value, step=event.step)
                            total_metrics += 1
                            
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")
                    continue
                        
            print(f"--- Successfully imported {total_metrics} TensorBoard metrics from {len(event_files)} files ---")
            
        except ImportError:
            print("TensorBoard package not available for metric import.")
            print("To enable TensorBoard metric import, install: pip install tensorboard")
            
    except Exception as e:
        print(f"Warning: Could not import TensorBoard metrics: {e}")
        print("TensorBoard artifacts will still be available in MLflow for manual viewing.")

# ==================================================================================
# == Main Execution Block                                                         ==
# ==================================================================================
if __name__ == '__main__':
    
    # ==============================================================================
    # == LOAD CONFIGURATION                                                       ==
    # ==============================================================================
    CONFIG = load_config("config.json")
    print("--- Configuration loaded successfully ---")
    
    # ==============================================================================
    
    # --- Runtime Options ---
    print("--- Runtime Configuration ---")
    run_name_input = input("Enter a name for this training run (or press Enter for default): ")
    print("-----------------------------")

    # --- Generate Run Name ---
    run_name = run_name_input if run_name_input else "PPO_training"

    # MLFLOW: Start an MLflow Run. All subsequent logging will be associated with this run.
    with mlflow.start_run(run_name=run_name):
        print(f"--- MLflow Run Started (Run ID: {mlflow.active_run().info.run_id}) ---")
        
        # MLFLOW: Log tags for quick filtering in the UI
        mlflow.set_tag("sb3.policy", CONFIG["training"]["ppo_policy"])
        mlflow.set_tag("env.id", "CustomRobotEnv")

        # MLFLOW: Log the entire configuration dictionary as parameters
        flat_config = flatten_dict(CONFIG)
        mlflow.log_params(flat_config)
        
        # MLFLOW: Also save the full config as a JSON artifact for easy viewing
        temp_dir = os.path.join("C:/temp/artifacts", f"run_{mlflow.active_run().info.run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        config_path = os.path.join(temp_dir, f"config_{run_name}.json")
        with open(config_path, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        mlflow.log_artifact(config_path)
        os.unlink(config_path)  # Clean up temp file

        MONITOR_KEYWORDS = ("is_success", "is_collision", "is_invalid_move", "grip_attempts", "failed_grips", "difficulty", "visible_steps")
        
        # --- Environment Setup ---
        env_config = CONFIG["environment"].copy()
        # Pass curriculum config to environment for initial difficulty setting
        env_config["curriculum"] = CONFIG["callbacks"]["curriculum_trainer"]

        def make_env():
            env = CustomRobotEnv(render_mode=None, config=env_config)
            env = Monitor(env, info_keywords=MONITOR_KEYWORDS)
            return env
        train_env = DummyVecEnv([make_env])
        
        # Specify which observation keys to normalize (only Box spaces, not MultiBinary)
        norm_obs_keys = [
            "ellipse_position_relative", "delta_ellipse_position_relative",
            "ellipse_major_axis_norm", "delta_ellipse_major_axis_norm",
            "ellipse_aspect_ratio", "delta_ellipse_aspect_ratio",
            "ellipse_orientation_2d", "delta_ellipse_orientation_2d",
            # "actuator_extensions", "delta_extensions",
            "relative_G1_position", "G1_velocity",
            "approach_vec", "radial_vec", "G1_angular_velocity"
        ]
        
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=norm_obs_keys)


        # --- Callback Setup ---
        custom_metrics_callback = CustomMetricsCallback(
            window_size=CONFIG["callbacks"]["custom_metrics"]["window_size"]
        )
        curriculum_callback = CurriculumTrainerCallback(
            metrics_callback=custom_metrics_callback,
            **CONFIG["callbacks"]["curriculum_trainer"]
        )
        active_callbacks = [custom_metrics_callback, curriculum_callback] 
        # active_callbacks = [custom_metrics_callback] #FROZEN curriculum for direct comparison

        callback_list = CallbackList(active_callbacks)

        # --- Agent Training ---
        print("--- Training the agent ---")
        
        # Set TensorBoard log path to temp directory to avoid OneDrive issues
        tensorboard_temp_path = "C:/temp/tensorboard_logs"
        
        # Use default PPO hyperparameters for simplicity
        ppo_kwargs = {
            "policy": CONFIG["training"]["ppo_policy"],
            "env": train_env,
            "verbose": CONFIG["training"].get("verbose", 1),
            "tensorboard_log": tensorboard_temp_path
        }
        
        model = PPO(**ppo_kwargs)
        
        model.learn(
            total_timesteps=CONFIG["training"]["total_timesteps"], 
            callback=callback_list, 
            progress_bar=True,
            tb_log_name=run_name
        )
        print("--- Training finished ---")

        # --- Save the Final Model ---
        temp_dir = os.path.join("C:/temp/artifacts", f"run_{mlflow.active_run().info.run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        model_save_path = os.path.join(temp_dir, f"ppo_model_{run_name}.zip")
        print(f"--- Saving the model to {model_save_path} ---")
        model.save(model_save_path)
        # Save VecNormalize statistics
        vecnorm_save_path = os.path.join(temp_dir, f"vecnormalize_{run_name}.pkl")
        train_env.save(vecnorm_save_path)
        # MLFLOW: Log the final model and the TensorBoard logs as artifacts
        print("--- Logging final artifacts to MLflow ---")
        mlflow.log_artifact(model_save_path, artifact_path="model")
        mlflow.log_artifact(vecnorm_save_path, artifact_path="model")
        # Clean up temporary model files
        os.unlink(model_save_path)
        os.unlink(vecnorm_save_path)

        # Log TensorBoard artifacts and import metrics to MLflow
        tensorboard_log_dir = None
        if os.path.exists(tensorboard_temp_path):
            # Search for directories that start with the run name (PPO adds suffixes like _1)
            for item in os.listdir(tensorboard_temp_path):
                item_path = os.path.join(tensorboard_temp_path, item)
                if os.path.isdir(item_path) and item.startswith(run_name):
                    tensorboard_log_dir = item_path
                    break
        
        if tensorboard_log_dir and os.path.exists(tensorboard_log_dir):
            mlflow.log_artifacts(tensorboard_log_dir, artifact_path="tensorboard")
            import_tensorboard_to_mlflow(tensorboard_log_dir)
        
        # Store the run ID for potential evaluation later
        current_run_id = mlflow.active_run().info.run_id
        
        # Clean up run-specific temp directory
        temp_dir = os.path.join("C:/temp/artifacts", f"run_{current_run_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        
        # Clean up model from memory
        del model
        
        train_env.close()
        
        print(f"--- MLflow Run Finished ---")
        print(f"Run ID: {current_run_id}")
        print(f"To view results, run 'mlflow ui --backend-store-uri file:///C:/temp/mlruns' in your terminal.")
