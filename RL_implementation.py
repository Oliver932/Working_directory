import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import imageio
from collections import deque
from datetime import datetime

# Import custom modules for the robot simulation
from arm_ik_model import RobotKinematics, Ring
from collision_and_render_management import CollisionAndRenderManager
from ring_projector import RingProjector
from overview_render_manager import OverviewRenderManager

# Import Stable Baselines3 components for reinforcement learning
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# ==================================================================================
# == Video Recording Callback                                                     ==
# ==================================================================================
class VideoRecorderCallback(BaseCallback):
    """A custom callback for recording and saving evaluation videos as MP4 files."""
    def __init__(self, eval_env: gym.Env, render_path: str, eval_freq: int, n_eval_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.render_path = render_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        os.makedirs(self.render_path, exist_ok=True)

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
            
            overview_vid_path = os.path.join(self.render_path, f"replay_step_{self.n_calls}_overview.mp4")
            robot_vid_path = os.path.join(self.render_path, f"replay_step_{self.n_calls}_robot.mp4")
            
            print(f"--- Saving overview video to {overview_vid_path} ---")
            imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
            print(f"--- Saving robot perspective video to {robot_vid_path} ---")
            imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')
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
            
        overview_vid_path = os.path.join(self.render_path, f"{prefix}_replay_overview.mp4")
        robot_vid_path = os.path.join(self.render_path, f"{prefix}_replay_robot.mp4")
        
        print(f"--- Saving overview video to {overview_vid_path} ---")
        imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
        print(f"--- Saving robot perspective video to {robot_vid_path} ---")
        imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')

# ==================================================================================
# == Custom Metrics Callback                                                      ==
# ==================================================================================
class CustomMetricsCallback(BaseCallback):
    """
    A custom callback to log domain-specific metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_rate_window = deque(maxlen=100)
        self.collision_episode_window = deque(maxlen=100)
        self.invalid_move_prop_window = deque(maxlen=100)
        self.failed_grip_rate_window = deque(maxlen=100)
        self.avg_difficulty_window = deque(maxlen=100)
        self.ellipse_visible_prop_window = deque(maxlen=100)
        self.ellipse_calculable_prop_window = deque(maxlen=100)

    def _on_step(self) -> bool:
        if self.locals.get("infos")[0].get("episode"):
            info = self.locals.get("infos")[0]
            ep_len = info["episode"]["l"]
            
            # --- Success Rate: Proportion of successful episodes ---
            self.success_rate_window.append(info.get("is_success", 0))
            self.logger.record("custom/success_rate", np.mean(self.success_rate_window))

            # --- Collision Rate: Proportion of episodes with at least one collision ---
            had_collision = 1 if info.get("is_collision", 0) > 0 else 0
            self.collision_episode_window.append(had_collision)
            self.logger.record("custom/collision_rate", np.mean(self.collision_episode_window))

            # --- Invalid Move Proportion: Proportion of steps in an episode that were invalid moves ---
            num_invalid_moves = info.get("is_invalid_move", 0)
            invalid_move_prop = num_invalid_moves / ep_len if ep_len > 0 else 0
            self.invalid_move_prop_window.append(invalid_move_prop)
            self.logger.record("custom/invalid_move_proportion", np.mean(self.invalid_move_prop_window))

            # --- Failed Grip Rate: Proportion of grip attempts that fail ---
            total_grip_attempts = info.get("grip_attempts", 0)
            total_failed_grips = info.get("failed_grips", 0)
            failure_rate = total_failed_grips / total_grip_attempts if total_grip_attempts > 0 else 0.0
            self.failed_grip_rate_window.append(failure_rate)
            self.logger.record("custom/failed_grip_rate", np.mean(self.failed_grip_rate_window))

            # --- Average Difficulty: The average spawn difficulty of episodes ---
            self.avg_difficulty_window.append(info.get("difficulty", 0))
            self.logger.record("custom/avg_difficulty", np.mean(self.avg_difficulty_window))

            # --- Ellipse Visible & Calculable Proportions ---
            visible_steps = info.get("visible_steps", 0)
            self.ellipse_visible_prop_window.append(visible_steps / ep_len if ep_len > 0 else 0)
            self.logger.record("custom/ellipse_visible_proportion", np.mean(self.ellipse_visible_prop_window))

            calculable_steps = info.get("calculable_steps", 0)
            self.ellipse_calculable_prop_window.append(calculable_steps / ep_len if ep_len > 0 else 0)
            self.logger.record("custom/ellipse_calculable_proportion", np.mean(self.ellipse_calculable_prop_window))

        return True

# ==================================================================================
# == Curriculum Trainer Callback                                                  ==
# ==================================================================================
class CurriculumTrainerCallback(BaseCallback):
    """
    A callback to implement curriculum learning. It increases the environment's
    difficulty when the agent's performance meets a certain threshold.
    """
    def __init__(self, metrics_callback: CustomMetricsCallback, success_threshold: float, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.metrics_callback = metrics_callback
        self.success_threshold = success_threshold
        self.check_freq = check_freq
        self.last_difficulty_increase_step = 0

    def _on_step(self) -> bool:
        if (self.n_calls - self.last_difficulty_increase_step) > self.check_freq:
            if len(self.metrics_callback.success_rate_window) > 0:
                current_success_rate = np.mean(self.metrics_callback.success_rate_window)
                if current_success_rate >= self.success_threshold:
                    if self.verbose > 0:
                        print(f"--- Curriculum Threshold Met! Success rate {current_success_rate:.2f} >= {self.success_threshold} ---")
                    self.training_env.env_method('increase_difficulty')
                    self.last_difficulty_increase_step = self.n_calls
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
        self.curriculum_levels = self.config["curriculum"]
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
        self.robot = RobotKinematics(verbosity=0)
        self.ring = Ring()
        
        # --- Curriculum and Noise Setup ---
        self.difficulty_level = 0
        self.pose_range_min = self.curriculum_levels[0]["pose_range_min"]
        self.pose_range_max = self.curriculum_levels[0]["pose_range_max"]
        self.observation_noise_std = self.curriculum_levels[0]["obs_noise"]
        self.action_noise_std = self.curriculum_levels[0]["action_noise"]

        self._setup_episode()
        
        self.ring_projector = RingProjector(self.robot, self.ring, vertical_fov_deg=camera_settings["fov"], image_width=camera_settings["width"], image_height=camera_settings["height"], method='custom')
        self.collision_render_manager = CollisionAndRenderManager(paths["gripper_col"], paths["gripper_col"], paths["ring_render"], paths["ring_col"], vertical_FOV=camera_settings["fov"], render_width=camera_settings["width"], render_height=camera_settings["height"])
        self.overview_render_manager = OverviewRenderManager(self.robot, self.ring, paths["ring_render"], width=overview_camera_settings["width"], height=overview_camera_settings["height"])
        
        self.ideal_E1, self.ideal_E1_quaternion = None, None
        self.E1_normalization_factor, self.quaternion_normalization_factor = 1.0, 1.0

    def _setup_episode(self):
        # Set the robot to a random pose with a difficulty sampled uniformly
        # from the min to the max range for the current curriculum level.
        self.episode_difficulty = np.random.uniform(self.pose_range_min, self.pose_range_max)
        self.robot.set_random_e1_pose(difficulty=self.episode_difficulty)
        
        self.ring = self.robot.create_ring(ring=self.ring)
        self.ideal_E1 = self.robot.E1.copy()
        self.ideal_E1_quaternion = self.robot.E1_quaternion.copy()
        self.robot.go_home()
        self.E1_normalization_factor = np.sum((self.robot.E1 - self.ideal_E1) ** 2) or 1.0
        self.quaternion_normalization_factor = 1.0  # Quaternions are already normalized

    def increase_difficulty(self):
        if self.difficulty_level < len(self.curriculum_levels) - 1:
            self.difficulty_level += 1
            level_settings = self.curriculum_levels[self.difficulty_level]
            self.pose_range_min = level_settings["pose_range_min"]
            self.pose_range_max = level_settings["pose_range_max"]
            self.observation_noise_std = level_settings["obs_noise"]
            self.action_noise_std = level_settings["action_noise"]
            print(f"DIFFICULTY INCREASED TO LEVEL {self.difficulty_level}: pose_range=[{self.pose_range_min}, {self.pose_range_max}], obs_noise={self.observation_noise_std}, action_noise={self.action_noise_std}")
        else:
            print("MAX DIFFICULTY REACHED.")

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
            "E1_position": self.robot.E1 / (self.E1_normalization_factor or 400.0), # Normalize the position
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
        
        # Reset previous state variables for delta calculations
        self.previous_extensions = None
        self.previous_E1_position = None
        self.previous_quaternion = None
        
        # Reset previous distance variables for improvement rewards
        self.previous_dist_E1 = None
        self.previous_dist_quaternion = None
        
        self._setup_episode()
        self.ring_projector.ring = self.ring
        self.overview_render_manager.ring = self.ring
        self.ring_projector.update()
        self.collision_render_manager.update_poses(self.robot, self.ring)
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

# ==================================================================================
# == Main Execution Block                                                         ==
# ==================================================================================
if __name__ == '__main__':
    
    # ==============================================================================
    # == CONFIGURATION & HYPERPARAMETERS                                          ==
    # ==============================================================================
    CONFIG = {
        "training": {
            "total_timesteps": 1000000,
            "ppo_policy": "MultiInputPolicy",
            "tensorboard_log_path": "./outputs/tensorboard_logs/", 
        },
        "evaluation": {
            "n_eval_episodes": 100,
            "n_video_episodes": 4
        },
        "environment": {
            "max_steps": 500,
            "multipliers": { "linear": 4.0, "rx": 0.5, "rz": 0.5 },
            "ellipse_scale_factor": 0.5,  # Scale factor for ellipse vectors to fit in [-1, 1] range
            "rewards": {
                "success": 100.0, "fail_grip": -30.0, "collision": -100.0,
                "fail_move": -10.0, "time": -0.01,
            },
            "paths": {
                "gripper_col": "./Working_directory/meshes/gripper_collision_mesh.stl",
                "ring_render": "./Working_directory/meshes/ring_render_mesh.stl",
                "ring_col": "./Working_directory/meshes/ring_collision_mesh.stl",
            },
            "camera": { "fov": 60, "width": 960, "height": 544 },
            "overview_camera": { "width": 320, "height": 240 },
            "curriculum": [
                {"pose_range_min": 0.0, "pose_range_max": 0.2, "obs_noise": 0.0, "action_noise": 0.0},
                {"pose_range_min": 0.2, "pose_range_max": 0.6, "obs_noise": 0.01, "action_noise": 0.01},
                {"pose_range_min": 0.6, "pose_range_max": 1.0, "obs_noise": 0.05, "action_noise": 0.05},
            ]
        },
        "callbacks": {
            "video_recorder": { "eval_freq": 100000, "n_eval_episodes": 3 },
            "curriculum_trainer": { "check_freq": 10000, "success_threshold": 0.8 }
        }
    }
    # ==============================================================================
    
    # --- Runtime Options ---
    print("--- Runtime Configuration ---")
    run_name_input = input("Enter a name for this training run (or press Enter for default): ")
    record_while_training = input("Record performance videos during training? (y/n): ").lower() == 'y'
    evaluate_at_end = input("Evaluate the final model after training? (y/n): ").lower() == 'y'
    render_final_video = input("Render a final performance video after training? (y/n): ").lower() == 'y'
    print("-----------------------------")

    # --- Generate Unique Paths for This Run ---
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{run_name_input}_{time_str}" if run_name_input else f"PPO_{time_str}"
    
    run_output_dir = os.path.join("outputs", run_name)
    model_save_path = os.path.join(run_output_dir, "ppo_custom_robot.zip")
    replays_path = os.path.join(run_output_dir, "replays")
    os.makedirs(replays_path, exist_ok=True)

    MONITOR_KEYWORDS = ("is_success", "is_collision", "is_invalid_move", "grip_attempts", "failed_grips", "difficulty", "visible_steps", "calculable_steps")
    
    # --- Environment Setup ---
    train_env = CustomRobotEnv(render_mode=None, config=CONFIG["environment"])
    train_env = Monitor(train_env, info_keywords=MONITOR_KEYWORDS)
    
    print("--- Checking the custom environment ---")
    check_env(train_env)
    print("--- Environment check passed! ---")

    eval_env = None
    needs_eval_env = record_while_training or evaluate_at_end or render_final_video
    if needs_eval_env:
        eval_env = CustomRobotEnv(render_mode='rgb_array', config=CONFIG["environment"])
        eval_env = Monitor(eval_env)

    # --- Callback Setup ---
    custom_metrics_callback = CustomMetricsCallback()
    curriculum_callback = CurriculumTrainerCallback(
        metrics_callback=custom_metrics_callback,
        **CONFIG["callbacks"]["curriculum_trainer"]
    )
    active_callbacks = [custom_metrics_callback, curriculum_callback]

    if needs_eval_env:
        video_callback = VideoRecorderCallback(
            eval_env=eval_env, 
            render_path=replays_path,
            **CONFIG["callbacks"]["video_recorder"]
        )
        if record_while_training:
            active_callbacks.append(video_callback)

    callback_list = CallbackList(active_callbacks)

    # --- Agent Training ---
    print("--- Training the agent ---")
    model = PPO(
        CONFIG["training"]["ppo_policy"], 
        train_env, 
        verbose=1, 
        tensorboard_log=CONFIG["training"]["tensorboard_log_path"]
    )
    
    model.learn(
        total_timesteps=CONFIG["training"]["total_timesteps"], 
        callback=callback_list, 
        progress_bar=True,
        tb_log_name=run_name
    )
    print("--- Training finished ---")

    # --- Save the Final Model ---
    print(f"--- Saving the model to {model_save_path} ---")
    model.save(model_save_path)
    del model

    # --- Load and Evaluate the Trained Model ---
    print(f"--- Loading the model from {model_save_path} ---")
    loaded_model = PPO.load(model_save_path)

    if evaluate_at_end:
        print("--- Evaluating policy ---")
        mean_reward, std_reward = evaluate_policy(
            loaded_model, 
            eval_env, 
            n_eval_episodes=CONFIG["evaluation"]["n_eval_episodes"], 
            deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if render_final_video:
        print("--- Recording final performance of the trained agent ---")
        video_callback.record_performance(
            loaded_model, 
            n_episodes=CONFIG["evaluation"]["n_video_episodes"], 
            prefix="final"
        )
    
    train_env.close()
    if eval_env:
        eval_env.close()