import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import imageio
import glob
from collections import deque
import json
import matplotlib.pyplot as plt

# MLFLOW: Import the mlflow library
import mlflow

# Set MLflow tracking URI to avoid OneDrive permission issues
mlflow.set_tracking_uri("file:///C:/temp/mlruns")

# Import custom modules for the robot simulation
from arm_ik_model import RobotKinematics, Ring
from collision_and_render_management import CollisionAndRenderManager
from ring_projector import RingProjector
from overview_render_manager import OverviewRenderManager
from ring_placement import set_random_pose_box_constraint, set_random_pose_box_constraint_fixed_rotation

# Import Stable Baselines3 components for reinforcement learning
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==================================================================================
# == Video Recording Callback                                                     ==
# ==================================================================================
class VideoRecorderCallback(BaseCallback):
    """A custom callback for recording and saving evaluation videos as MP4 files."""
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            overview_frames, robot_perspective_frames = [], []
            print(f"--- Recording performance at step {self.n_calls} ---")
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    # Since the env is wrapped in a Monitor, we need to access the underlying env for render
                    overview_frames.append(self.eval_env.env.render(camera_name='scene_overview'))
                    robot_perspective_frames.append(self.eval_env.env.render(camera_name='robot_perspective'))
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
            
            # Use temporary files that get deleted after MLflow logging
            temp_dir = "C:/temp/artifacts"
            os.makedirs(temp_dir, exist_ok=True)
            
            overview_vid_path = os.path.join(temp_dir, f"step_{self.n_calls}_overview.mp4")
            robot_vid_path = os.path.join(temp_dir, f"step_{self.n_calls}_robot.mp4")
            
            print(f"--- Saving overview video to {overview_vid_path} ---")
            imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
            print(f"--- Saving robot perspective video to {robot_vid_path} ---")
            imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')

            # MLFLOW: Log the generated videos as artifacts
            mlflow.log_artifact(overview_vid_path, artifact_path="replays")
            mlflow.log_artifact(robot_vid_path, artifact_path="replays")
            
            # Clean up temporary files
            os.unlink(overview_vid_path)
            os.unlink(robot_vid_path)
        return True

    def record_performance(self, model, n_episodes=5, prefix="final"):
        """A standalone method to record the performance of a trained model."""
        overview_frames, robot_perspective_frames = [], []
        obs, _ = self.eval_env.reset()
        for _ in range(n_episodes):
            done = False
            while not done:
                overview_frames.append(self.eval_env.env.render(camera_name='scene_overview'))
                robot_perspective_frames.append(self.eval_env.env.render(camera_name='robot_perspective'))
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
            obs, _ = self.eval_env.reset()
            
        # Use temporary files
        temp_dir = "C:/temp/artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        
        overview_vid_path = os.path.join(temp_dir, f"{prefix}_replay_overview.mp4")
        robot_vid_path = os.path.join(temp_dir, f"{prefix}_replay_robot.mp4")
        
        print(f"--- Saving overview video to {overview_vid_path} ---")
        imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
        print(f"--- Saving robot perspective video to {robot_vid_path} ---")
        imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')

        # MLFLOW: Log the final videos as artifacts
        mlflow.log_artifact(overview_vid_path, artifact_path="replays")
        mlflow.log_artifact(robot_vid_path, artifact_path="replays")
        
        # Clean up temporary files
        os.unlink(overview_vid_path)
        os.unlink(robot_vid_path)

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
        self.ellipse_calculable_prop_window = deque(maxlen=window_size)

    def _on_step(self) -> bool:
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

            # --- Ellipse Visible & Calculable Proportions ---
            visible_steps = info.get("visible_steps", 0)
            self.ellipse_visible_prop_window.append(visible_steps / ep_len if ep_len > 0 else 0)
            avg_visible_prop = np.mean(self.ellipse_visible_prop_window)
            self.logger.record("custom/ellipse_visible_proportion", avg_visible_prop)

            calculable_steps = info.get("calculable_steps", 0)
            self.ellipse_calculable_prop_window.append(calculable_steps / ep_len if ep_len > 0 else 0)
            avg_calculable_prop = np.mean(self.ellipse_calculable_prop_window)
            self.logger.record("custom/ellipse_calculable_proportion", avg_calculable_prop)

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
        self.ellipse_scale_factor = self.config["ellipse_scale_factor"]
        
        # Difficulty configuration
        difficulty_config = self.config["difficulty"]
        self.base_observation_noise = difficulty_config["base_observation_noise"]
        self.base_action_noise = difficulty_config["base_action_noise"]
        self.max_observation_noise = difficulty_config["max_observation_noise"]
        self.max_action_noise = difficulty_config["max_action_noise"]
        
        paths = self.config["paths"]
        camera_settings = self.config["camera"]
        overview_camera_settings = self.config["overview_camera"]

        # --- Define Observation and Action Spaces ---
        self.observation_space = spaces.Dict({
            "ellipse_position": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
            "delta_ellipse_position": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),

            "ellipse_semi_major_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "delta_ellipse_semi_major_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "ellipse_semi_minor_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "delta_ellipse_semi_minor_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),

            "ellipse_visible": spaces.MultiBinary(1),

            "actuator_extensions": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "delta_extensions": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),

            "E1_position": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "delta_E1": spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
            "E1_quaternion": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "delta_E1_quaternion": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),

            "last_move_successful": spaces.MultiBinary(1),

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
        self._update_difficulty_settings()

        
        self.ring_projector = RingProjector(self.robot, self.ring, vertical_fov_deg=camera_settings["fov"], image_width=camera_settings["width"], image_height=camera_settings["height"], method='custom')
        self.collision_render_manager = CollisionAndRenderManager(paths["gripper_col"], paths["gripper_col"], paths["ring_render"], paths["ring_col"], vertical_FOV=camera_settings["fov"], render_width=camera_settings["width"], render_height=camera_settings["height"])
        self.overview_render_manager = OverviewRenderManager(self.robot, self.ring, paths["ring_render"], width=overview_camera_settings["width"], height=overview_camera_settings["height"])
        
        self._setup_episode()

        self.ideal_E1, self.ideal_E1_quaternion = None, None
        self.E1_normalization_factor, self.quaternion_normalization_factor = 1.0, 1.0

    def _update_difficulty_settings(self):
        """Update observation noise, action noise based on current difficulty (0.0 to 1.0)"""
        self.observation_noise_std = self.base_observation_noise + (self.max_observation_noise - self.base_observation_noise) * self.difficulty
        self.action_noise_std = self.base_action_noise + (self.max_action_noise - self.base_action_noise) * self.difficulty

    def _setup_episode(self):
        # Set the robot to a random pose with difficulty from 0.0 to 1.0
        success, pose, actual_difficulty = set_random_pose_box_constraint_fixed_rotation(self.robot, self.ring, self.collision_render_manager, difficulty=self.difficulty)

        if not success:
            raise ValueError("Failed to set a valid random pose for the robot and ring. Please check the configuration and constraints.")

        
        # Use the actual calculated difficulty for logging purposes
        self.episode_difficulty = actual_difficulty

        self.ideal_E1 = self.robot.E1.copy()
        self.ideal_E1_quaternion = self.robot.E1_quaternion.copy()
        self.E1_normalization_factor = np.sum((self.robot.E1 - self.ideal_E1) ** 2) or 1.0
        self.quaternion_normalization_factor = 1.0  # Quaternions are already normalized

        self.robot.go_home()
        self.collision_render_manager.update_poses(self.robot, self.ring)
        self.ring_projector.update()


    def get_difficulty(self):
        """Get the current difficulty level (0.0 to 1.0)"""
        return self.difficulty

    def set_difficulty(self, new_difficulty):
        """Set the difficulty level (0.0 to 1.0) and update related parameters"""
        self.difficulty = np.clip(new_difficulty, 0.0, 1.0)
        self._update_difficulty_settings()
        if hasattr(self, 'verbose') and getattr(self, 'verbose', 0) > 0:
            print(f"DIFFICULTY SET TO {self.difficulty:.2f}: obs_noise={self.observation_noise_std:.3f}, action_noise={self.action_noise_std:.3f}")

    def adjust_difficulty(self, delta):
        """Adjust difficulty by a delta amount"""
        self.set_difficulty(self.difficulty + delta)

    def _add_observation_noise(self, obs):
        if self.observation_noise_std > 0:
            # Add noise to ellipse-related observations
            obs["ellipse_position"] += np.random.normal(0, self.observation_noise_std, size=obs["ellipse_position"].shape)
            obs["delta_ellipse_position"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_ellipse_position"].shape)
            obs["ellipse_semi_major_vector"] += np.random.normal(0, self.observation_noise_std, size=obs["ellipse_semi_major_vector"].shape)
            obs["delta_ellipse_semi_major_vector"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_ellipse_semi_major_vector"].shape)
            obs["ellipse_semi_minor_vector"] += np.random.normal(0, self.observation_noise_std, size=obs["ellipse_semi_minor_vector"].shape)
            obs["delta_ellipse_semi_minor_vector"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_ellipse_semi_minor_vector"].shape)
            
            # Add noise to robot state observations
            obs["actuator_extensions"] += np.random.normal(0, self.observation_noise_std, size=obs["actuator_extensions"].shape)
            obs["delta_extensions"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_extensions"].shape)
            obs["E1_position"] += np.random.normal(0, self.observation_noise_std, size=obs["E1_position"].shape)
            obs["delta_E1"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_E1"].shape)
            obs["E1_quaternion"] += np.random.normal(0, self.observation_noise_std, size=obs["E1_quaternion"].shape)
            obs["delta_E1_quaternion"] += np.random.normal(0, self.observation_noise_std, size=obs["delta_E1_quaternion"].shape)
        
        # Always clip all continuous observations to ensure they are within the defined space
        for key, space in self.observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                obs[key] = np.clip(obs[key], space.low, space.high)
        
        return obs

    def _add_action_noise(self, action):
        if self.action_noise_std > 0:
            noise = np.random.normal(0, self.action_noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def _get_obs(self):
        
        ellipse_details = self.ring_projector.projected_properties

        observations = {
            "ellipse_position":  ellipse_details['center_2d'],
            "delta_ellipse_position": ellipse_details['delta_center_2d'],
            "ellipse_semi_major_vector": ellipse_details['semi_major_vector'] * self.ellipse_scale_factor,
            "delta_ellipse_semi_major_vector": ellipse_details['delta_semi_major_vector'] * self.ellipse_scale_factor,
            "ellipse_semi_minor_vector": ellipse_details['semi_minor_vector'] * self.ellipse_scale_factor,
            "delta_ellipse_semi_minor_vector": ellipse_details['delta_semi_minor_vector'] * self.ellipse_scale_factor,
            "ellipse_visible": np.array([1] if ellipse_details.get('calculable', False) else [0], dtype=np.int8),

            "actuator_extensions": self.robot.extensions,
            "delta_extensions": self.robot.delta_extensions,
            "E1_position": self.robot.E1,
            "delta_E1": self.robot.delta_E1,
            "E1_quaternion": self.robot.E1_quaternion,
            "delta_E1_quaternion": self.robot.delta_E1_quaternion,
            "last_move_successful": np.array([1] if getattr(self.robot, 'last_solve_successful', True) else [0], dtype=np.int8)
        }

        return self._add_observation_noise(observations)

    def _get_info(self):
        # This is now only called when the episode ends
        return {
            "is_success": self.is_success,
            "is_collision": self.is_collision,
            "is_invalid_move": self.is_invalid_move,
            "grip_attempts": self.grip_attempts,
            "failed_grips": self.failed_grips,
            "difficulty": self.episode_difficulty,
            "visible_steps": self.visible_steps,
            "calculable_steps": self.calculable_steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Reset episode-specific trackers
        self.is_success = False
        self.is_collision = False
        self.is_invalid_move = 0
        self.grip_attempts = 0
        self.failed_grips = 0
        self.visible_steps = 0
        self.calculable_steps = 0
        
        # Reset previous distance variables for improvement rewards
        self.previous_dist_E1 = None
        self.previous_dist_quaternion = None
        
        self._setup_episode()

        return self._get_obs(), {} # Return empty info dict on reset

    def step(self, action):
        self.current_step += 1
        terminated = False
        reward = self.rewards["time"] * self.current_step
        
        action = self._add_action_noise(action)
        pose_deltas = action[0:4]
        should_grip = action[4] > 0.0

        if should_grip:
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

        if not terminated:
            self.robot.move_E1(
                pose_deltas[0] * self.multipliers["linear"], 
                pose_deltas[1] * self.multipliers["linear"], 
                pose_deltas[2] * self.multipliers["rx"], 
                pose_deltas[3] * self.multipliers["rz"]
            )
            self.ring_projector.update()
            
            # Update visibility and calculability counters
            ellipse_details = self.ring_projector.projected_properties
            if ellipse_details.get('visible', False):
                self.visible_steps += 1
            if ellipse_details.get('calculable', False):
                self.calculable_steps += 1

            self.collision_render_manager.update_poses(self.robot, self.ring)
            if self.collision_render_manager.check_collision():
                reward += self.rewards["collision"]
                self.is_collision = True
                terminated = True

        # Reward movement towards ideal position and orientation
        current_dist_E1 = np.sum((self.robot.E1 - self.ideal_E1) ** 2) / self.E1_normalization_factor
        # Calculate quaternion distance (1 - |dot product|) to measure rotational difference
        quaternion_dot = np.abs(np.dot(self.robot.E1_quaternion, self.ideal_E1_quaternion))
        current_dist_quaternion = 1.0 - quaternion_dot  # Distance metric for quaternions
        
        # Calculate improvement rewards (only after first step when previous distances exist)
        if self.previous_dist_E1 is not None:
            # Reward improvement (movement towards goal)
            improvement_E1 = self.previous_dist_E1 - current_dist_E1
            improvement_quaternion = self.previous_dist_quaternion - current_dist_quaternion
            
            # Scale the improvement rewards
            reward += improvement_E1 # Reward position improvement
            reward += improvement_quaternion # Reward orientation improvement
        
        # Store current distances for next step
        self.previous_dist_E1 = current_dist_E1
        self.previous_dist_quaternion = current_dist_quaternion

        if not self.robot.last_solve_successful:
            reward += self.rewards["fail_move"]
            self.is_invalid_move += 1

        truncated = self.current_step >= self.MAX_STEPS_PER_EPISODE
        
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
    record_while_training = input("Record performance videos during training? (y/n): ").lower() == 'y'
    render_final_video = input("Render a final performance video after training? (y/n): ").lower() == 'y'
    save_stats = input("Save VecNormalize normalisation stats plots to MLflow? (y/n): ").lower() == 'y'
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
        temp_dir = "C:/temp/artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        config_path = os.path.join(temp_dir, f"config_{run_name}.json")
        with open(config_path, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        mlflow.log_artifact(config_path)
        os.unlink(config_path)  # Clean up temp file

        MONITOR_KEYWORDS = ("is_success", "is_collision", "is_invalid_move", "grip_attempts", "failed_grips", "difficulty", "visible_steps", "calculable_steps")
        
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
            "ellipse_position", "delta_ellipse_position",
            "ellipse_semi_major_vector", "delta_ellipse_semi_major_vector", 
            "ellipse_semi_minor_vector", "delta_ellipse_semi_minor_vector",
            "actuator_extensions", "delta_extensions",
            "E1_position", "delta_E1", "E1_quaternion", "delta_E1_quaternion"
        ]
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=norm_obs_keys)

        eval_env = None
        needs_eval_env = record_while_training or render_final_video
        if needs_eval_env:
            def make_eval_env():
                env = CustomRobotEnv(render_mode='rgb_array', config=CONFIG["environment"])
                env = Monitor(env)
                return env
            eval_env = DummyVecEnv([make_eval_env])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., norm_obs_keys=norm_obs_keys)
            # Synchronize statistics
            eval_env.obs_rms = train_env.obs_rms

        # --- Callback Setup ---
        custom_metrics_callback = CustomMetricsCallback(
            window_size=CONFIG["callbacks"]["custom_metrics"]["window_size"]
        )
        curriculum_callback = CurriculumTrainerCallback(
            metrics_callback=custom_metrics_callback,
            **CONFIG["callbacks"]["curriculum_trainer"]
        )
        active_callbacks = [custom_metrics_callback, curriculum_callback]

        if needs_eval_env:
            video_callback = VideoRecorderCallback(
                eval_env=eval_env, 
                **CONFIG["callbacks"]["video_recorder"]
            )
            if record_while_training:
                active_callbacks.append(video_callback)

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
        temp_dir = "C:/temp/artifacts"
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
        
        if render_final_video:
            print("--- Recording final performance of the trained agent ---")
            # The record_performance method now automatically logs to MLflow
            video_callback.record_performance(
                model, 
                n_episodes=CONFIG["evaluation"]["n_video_episodes"], 
                prefix="final"
            )
        
        # Clean up model from memory
        del model
        
        train_env.close()
        if eval_env:
            eval_env.close()
        
        # Save VecNormalize stats plots if user requested
        if save_stats:
            def plot_and_save_stats(vecnorm, name_prefix):
                import numpy as np
                # For Dict observation spaces, obs_rms.mean and obs_rms.var are dictionaries
                obs_means = vecnorm.obs_rms.mean
                obs_vars = vecnorm.obs_rms.var
                
                if isinstance(obs_means, dict):
                    # Handle Dict observation space
                    fig, axes = plt.subplots(2, len(obs_means), figsize=(4*len(obs_means), 8))
                    if len(obs_means) == 1:
                        axes = axes.reshape(-1, 1)
                    
                    for i, (key, mean_vals) in enumerate(obs_means.items()):
                        if key in norm_obs_keys:  # Only plot normalized keys
                            var_vals = obs_vars[key]
                            std_vals = np.sqrt(var_vals)
                            
                            # Plot means
                            axes[0, i].bar(range(len(mean_vals)), mean_vals)
                            axes[0, i].set_title(f"{key} - Mean")
                            axes[0, i].tick_params(axis='x', rotation=45)
                            
                            # Plot standard deviations
                            axes[1, i].bar(range(len(std_vals)), std_vals)
                            axes[1, i].set_title(f"{key} - Std")
                            axes[1, i].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    obs_plot_path = os.path.join(temp_dir, f"{name_prefix}_obs_stats.png")
                    plt.savefig(obs_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(obs_plot_path, artifact_path="vecnormalize_stats")
                    os.unlink(obs_plot_path)
                else:
                    # Handle single array observation space (fallback)
                    obs_std = np.sqrt(obs_vars)
                    plt.figure(figsize=(10,4))
                    plt.subplot(1,2,1)
                    plt.title(f"{name_prefix} Obs Mean")
                    plt.bar(np.arange(len(obs_means)), obs_means)
                    plt.subplot(1,2,2)
                    plt.title(f"{name_prefix} Obs Std")
                    plt.bar(np.arange(len(obs_std)), obs_std)
                    plt.tight_layout()
                    obs_plot_path = os.path.join(temp_dir, f"{name_prefix}_obs_stats.png")
                    plt.savefig(obs_plot_path)
                    plt.close()
                    mlflow.log_artifact(obs_plot_path, artifact_path="vecnormalize_stats")
                    os.unlink(obs_plot_path)
                
                # Reward stats
                ret_mean = vecnorm.ret_rms.mean
                ret_std = np.sqrt(vecnorm.ret_rms.var)
                plt.figure(figsize=(6,4))
                plt.title(f"{name_prefix} Reward Mean/Std")
                plt.bar(["mean", "std"], [ret_mean, ret_std])
                plt.ylabel("Value")
                ret_plot_path = os.path.join(temp_dir, f"{name_prefix}_reward_stats.png")
                plt.savefig(ret_plot_path)
                plt.close()
                mlflow.log_artifact(ret_plot_path, artifact_path="vecnormalize_stats")
                os.unlink(ret_plot_path)
                
            plot_and_save_stats(train_env, "train_env")
            if needs_eval_env:
                plot_and_save_stats(eval_env, "eval_env")

        print(f"--- MLflow Run Finished ---")
        print(f"To view results, run 'mlflow ui --backend-store-uri file:///C:/temp/mlruns' in your terminal.")