import shap
import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
import imageio
import mlflow
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import custom modules for the robot simulation
from RL_training import CustomRobotEnv

# Constants
TEMP_DIR = "C:/temp/artifacts"
MLFLOW_URI = "file:///C:/temp/mlruns"


def evaluate_with_SHAP(run_id, model, eval_env, n_background=100, n_eval=200, MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Evaluate model using SHAP and generate feature importance heatmaps for each action dimension.
    
    Args:
        run_id: MLflow run ID for logging artifacts
        model: Trained PPO model
        eval_env: Evaluation environment
        n_background: Number of background samples for SHAP baseline
        n_eval: Number of evaluation samples for SHAP explanations
        MLFLOW_URI: MLflow tracking URI
        TEMP_DIR: Base temporary directory
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Use run-specific temp directory to avoid conflicts
    temp_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    with mlflow.start_run(run_id=run_id):
        # Use all keys from the observation space, in order
        obs_keys = list(eval_env.observation_space.spaces.keys())

        print(f"Collecting background dataset ({n_background} samples)...")
        # Collect background dataset (observations from random rollouts)
        obs_list = []
        obs_dict_list = []  # Store unflattened dict observations for SHAP
        for _ in range(n_background):
            obs = eval_env.reset()
            done = [False]
            while not done[0]:
                # Store both flattened and dict format
                if isinstance(obs, dict):
                    flat_obs = np.concatenate([obs[k].flatten() for k in obs_keys if k in obs])
                    obs_dict_list.append(obs.copy())  # Store original dict format
                else:
                    flat_obs = obs.flatten()
                    obs_dict_list.append({'obs': obs})  # Wrap single obs in dict format
                obs_list.append(flat_obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
                if len(obs_list) >= n_background:
                    break
            if len(obs_list) >= n_background:
                break

        # Stack the collected background observations into numpy arrays
        background = np.stack(obs_list[:n_background])
        background_dict = obs_dict_list[:n_background]

        # Save background datasets as MLflow artifacts
        background_path = os.path.join(temp_dir, "shap_background.npy")
        np.save(background_path, background)
        mlflow.log_artifact(background_path, artifact_path="shap")
        os.unlink(background_path)
        
        background_dict_path = os.path.join(temp_dir, "shap_background_dict.npy")
        np.save(background_dict_path, background_dict, allow_pickle=True)
        mlflow.log_artifact(background_dict_path, artifact_path="shap")
        os.unlink(background_dict_path)
        print(f"Saved background datasets (flattened: {background.shape}, dict format)")

        print(f"Collecting evaluation dataset ({n_eval} samples)...")
        eval_obs_list = []
        for _ in range(n_eval):
            obs = eval_env.reset()
            done = [False]
            while not done[0]:
                if isinstance(obs, dict):
                    flat_obs = np.concatenate([obs[k].flatten() for k in obs_keys if k in obs])
                else:
                    flat_obs = obs.flatten()
                eval_obs_list.append(flat_obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
                if len(eval_obs_list) >= n_eval:
                    break
            if len(eval_obs_list) >= n_eval:
                break
        eval_data = np.stack(eval_obs_list[:n_eval])

        # 3. Define model prediction function for SHAP
        def model_predict(X):
            # X shape: (n_samples, n_features)
            # Reconstruct obs dict with batched arrays for Dict obs space
            if hasattr(eval_env.observation_space, 'spaces'):
                obs_dict = {}
                idx = 0
                for k in obs_keys:
                    space = eval_env.observation_space.spaces[k]
                    size = np.prod(space.shape)
                    obs_dict[k] = X[:, idx:idx+size].reshape((X.shape[0],) + space.shape)
                    idx += size
                obs_batch = obs_dict
            else:
                obs_batch = X
            actions, _ = model.predict(obs_batch, deterministic=True)
            return actions

        print("Computing SHAP values...")
        # Run SHAP KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(eval_data)

        print("Generating SHAP feature importance heatmaps...")
        # Generate and save SHAP heatmaps (one per action/output)

        # Prepare feature names and heatmap layout info
        obs_heatmap_xlabels = []  # column labels (keys)
        obs_heatmap_shape = []    # number of rows for each key
        flat_idx = 0
        for k in obs_keys:
            space = eval_env.observation_space.spaces[k]
            dim = np.prod(space.shape)
            obs_heatmap_xlabels.append(k)
            obs_heatmap_shape.append(dim)
            flat_idx += dim
        max_rows = max(obs_heatmap_shape)
        n_cols = len(obs_heatmap_xlabels)

        # Prepare SHAP values array: (n_eval, n_features, n_outputs)
        if isinstance(shap_values, list):
            # List of arrays, one per output: each (n_eval, n_features)
            shap_arr = np.stack(shap_values, axis=-1)  # (n_eval, n_features, n_outputs)
        else:
            shap_arr = shap_values  # already (n_eval, n_features, n_outputs)


        # Average over samples for each output (feature importance = mean absolute SHAP)
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)  # (n_features, n_outputs)

        # Create a figure with 5 heatmaps (one per output/action)
        n_outputs = mean_abs_shap.shape[1]
        n_heatmaps = min(5, n_outputs)
        action_labels = ['Delta X', 'Delta Y', 'Delta RX', 'Delta RZ', 'Grip Signal']
        fig, axes = plt.subplots(n_heatmaps, 1, figsize=(8, 2.5 * n_heatmaps))
        if n_heatmaps == 1:
            axes = [axes]

        # For each output/action, build a 2D heatmap (rows: max_rows, cols: n_cols)
        for i in range(n_heatmaps):
            heatmap_data = np.full((max_rows, n_cols), np.nan)
            feature_idx = 0
            for col, n_row in enumerate(obs_heatmap_shape):
                for row in range(n_row):
                    heatmap_data[row, col] = mean_abs_shap[feature_idx, i]
                    feature_idx += 1
            ax = axes[i]
            vmax = np.max(mean_abs_shap)
            label = action_labels[i] if i < len(action_labels) else f"Action {i}"
            img = ax.imshow(heatmap_data, aspect='auto', cmap='Reds', vmin=0, vmax=vmax)
            ax.set_title(f"{label} Feature Importance", color=plt.cm.tab10(i))
            cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.01, aspect=20)
            cbar.set_label('Mean |SHAP| (Importance)')
            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(obs_heatmap_xlabels, rotation=30, ha='right')
            ax.set_yticks(np.arange(max_rows))
            ax.set_yticklabels([str(j) for j in range(max_rows)])
        fig.suptitle('SHAP Feature Importance Heatmaps (per Action)', fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # Save and log the figure
        shap_heatmap_path = os.path.join(temp_dir, "shap_action_heatmaps.png")
        plt.savefig(shap_heatmap_path)
        mlflow.log_artifact(shap_heatmap_path, artifact_path="shap")
        plt.close(fig)
        os.unlink(shap_heatmap_path)

    print("SHAP analysis completed and logged to MLflow")

def record_final_performance(run_id, model, eval_env, n_episodes=5, prefix="final", MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Record evaluation videos of the trained model and log them to MLflow.
    
    Args:
        run_id: MLflow run ID for logging artifacts
        model: Trained PPO model
        eval_env: Evaluation environment
        n_episodes: Number of episodes to record
        prefix: Prefix for video filenames
        MLFLOW_URI: MLflow tracking URI
        TEMP_DIR: Base temporary directory
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Use run-specific temp directory to avoid conflicts
    temp_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    with mlflow.start_run(run_id=run_id):
        overview_frames, robot_perspective_frames = [], []
        print(f"Recording {prefix} performance videos ({n_episodes} episodes)...")

        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = [False]
            while not done[0]:
                overview_frames.append(eval_env.envs[0].unwrapped.render(camera_name='scene_overview'))
                robot_perspective_frames.append(eval_env.envs[0].unwrapped.render(camera_name='robot_perspective'))
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)

        # Save videos to temporary files
        overview_vid_path = os.path.join(temp_dir, f"{prefix}_replay_overview.mp4")
        robot_vid_path = os.path.join(temp_dir, f"{prefix}_replay_robot.mp4")

        print(f"Saving overview video...")
        imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
        print(f"Saving robot perspective video...")
        imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')

        # Log videos as MLflow artifacts
        mlflow.log_artifact(overview_vid_path, artifact_path="replays")
        mlflow.log_artifact(robot_vid_path, artifact_path="replays")

        # Clean up temporary files
        os.unlink(overview_vid_path)
        os.unlink(robot_vid_path)

    print(f"Performance videos logged to MLflow successfully")

def plot_and_save_stats(run_id, vecnorm, name_prefix, MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Generate and save VecNormalize statistics as a CSV file and log it to MLflow.
    Handles both dict and array observation spaces.
    
    Args:
        run_id: MLflow run ID for logging artifacts
        vecnorm: VecNormalize object containing statistics
        name_prefix: Prefix for the output CSV filename
        MLFLOW_URI: MLflow tracking URI
        TEMP_DIR: Base temporary directory
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Use run-specific temp directory to avoid conflicts
    temp_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    with mlflow.start_run(run_id=run_id):
        obs_rms = vecnorm.obs_rms
        summary_rows = []
        
        # Extract normalized observation keys from VecNormalize object
        if isinstance(obs_rms, dict):
            norm_obs_keys = list(obs_rms.keys())
        else:
            # Fallback: if it's not a dict obs space, use a generic key
            norm_obs_keys = ['obs']
        
        print(f"Generating VecNormalize statistics for: {name_prefix}")
        
        # Process observation statistics based on observation space type
        if isinstance(obs_rms, dict):
            keys = [k for k in obs_rms.keys() if k in norm_obs_keys]
            if not keys:
                print(f"Warning: No normalized observation keys found for {name_prefix}. Skipping summary table.")
                return
            for key in keys:
                rms = obs_rms[key]
                mean_vals = getattr(rms, 'mean', None)
                var_vals = getattr(rms, 'var', None)
                if mean_vals is None or var_vals is None:
                    print(f"Warning: Missing mean/var for key '{key}'. Skipping.")
                    continue
                std_vals = np.sqrt(var_vals)
                for idx in range(len(mean_vals)):
                    summary_rows.append({
                        'obs_key': key,
                        'dim': idx,
                        'mean': mean_vals[idx],
                        'std': std_vals[idx],
                        'min': mean_vals[idx] - std_vals[idx],
                        'max': mean_vals[idx] + std_vals[idx]
                    })
        else:
            # Fallback: single array obs space
            mean_vals = getattr(obs_rms, 'mean', None)
            var_vals = getattr(obs_rms, 'var', None)
            if mean_vals is None or var_vals is None:
                print(f"Warning: obs_rms missing mean/var for {name_prefix}. Skipping summary table.")
                return
            std_vals = np.sqrt(var_vals)
            for idx in range(len(mean_vals)):
                summary_rows.append({
                    'obs_key': 'obs',
                    'dim': idx,
                    'mean': mean_vals[idx],
                    'std': std_vals[idx],
                    'min': mean_vals[idx] - std_vals[idx],
                    'max': mean_vals[idx] + std_vals[idx]
                })

        # Add reward statistics
        ret_mean = getattr(vecnorm.ret_rms, 'mean', None)
        ret_var = getattr(vecnorm.ret_rms, 'var', None)
        if ret_mean is not None and ret_var is not None:
            ret_std = np.sqrt(ret_var)
            summary_rows.append({
                'obs_key': 'reward',
                'dim': 0,
                'mean': ret_mean,
                'std': ret_std,
                'min': ret_mean - ret_std,
                'max': ret_mean + ret_std
            })
        else:
            print(f"Warning: reward statistics missing for {name_prefix}.")

        # Save and log statistics if data is available
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(temp_dir, f"{name_prefix}_vecnormalize_summary.csv")
            df.to_csv(summary_path, index=False)
            mlflow.log_artifact(summary_path, artifact_path="vecnormalize_stats")
            os.unlink(summary_path)
            print(f"VecNormalize statistics saved to MLflow successfully")
        else:
            print(f"No statistics data found for {name_prefix}")
    

# VERY USEFUL FUNCTION FOR LOADING ANY MODEL AND ENVIRONMENTS FROM MLFLOW
 

def load_trained_model_and_envs(run_id, MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Load a trained model, config, and VecNormalize wrapper from MLflow artifacts and create evaluation environments.
    Returns the config, model, evaluation environment, and normalized observation keys.
    """
    
    # Set MLflow tracking URI and start run context
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Verify run exists and display basic information
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    run_name = run.data.tags.get('mlflow.runName', 'unknown_run')
    print(f"Loading artifacts for run: {run_name} (ID: {run_id})")
    
    with mlflow.start_run(run_id=run_id):
        # Create run-specific temporary directory to avoid conflicts between different runs
        temp_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
        if os.path.exists(temp_dir):
            print(f"Cleaning existing temporary directory...")
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download all artifacts from MLflow to the run-specific directory
        print("Downloading artifacts from MLflow...")
        artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_dir)
        
        # Load configuration file
        config_files = glob.glob(os.path.join(artifact_dir, "config_*.json"))
        if not config_files:
            raise FileNotFoundError("No config_*.json file found in MLflow artifacts.")
        
        print(f"Loading configuration from: {os.path.basename(config_files[0])}")
        with open(config_files[0], "r") as f:
            config = json.load(f)
        
        # Locate model directory and verify contents
        model_path = os.path.join(artifact_dir, "model")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model directory not found in MLflow artifacts")
        
        # Scan for model and VecNormalize files
        model_files = []
        vecnorm_files = []
        
        for file in os.listdir(model_path):
            if file.endswith('.zip') and 'ppo_model' in file.lower():
                model_files.append(file)
            elif file.endswith('.pkl') and 'vecnormalize' in file.lower():
                vecnorm_files.append(file)
        
        if not model_files or not vecnorm_files:
            # If no files found, show what's available for debugging
            print(f"Available files in model directory:")
            for file in os.listdir(model_path):
                file_path = os.path.join(model_path, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else "directory"
                print(f"  {file} ({size} bytes)")
            raise FileNotFoundError(f"Could not find model or VecNormalize files. Found: {os.listdir(model_path)}")
        
        # Select files that match the run name, with fallback to first available
        model_file = None
        vecnorm_file = None
        
        # Prefer files that contain the run name (handle spaces by converting to underscores)
        run_name_normalized = run_name.replace(' ', '_')
        
        for file in model_files:
            if run_name_normalized in file or run_name in file:
                model_file = os.path.join(model_path, file)
                break
        
        if not model_file:
            model_file = os.path.join(model_path, model_files[0])
            print(f"Note: Using first available model file: {model_files[0]}")
        
        for file in vecnorm_files:
            if run_name_normalized in file or run_name in file:
                vecnorm_file = os.path.join(model_path, file)
                break
        
        if not vecnorm_file:
            vecnorm_file = os.path.join(model_path, vecnorm_files[0])
            print(f"Note: Using first available VecNormalize file: {vecnorm_files[0]}")
        
        # Load model and VecNormalize wrapper
        print(f"Loading PPO model: {os.path.basename(model_file)}")
        print(f"Loading VecNormalize wrapper: {os.path.basename(vecnorm_file)}")
        
        # Create evaluation environment
        def make_eval_env():
            env = CustomRobotEnv(render_mode='rgb_array', config=config["environment"])
            env = Monitor(env)
            return env

        eval_env = DummyVecEnv([make_eval_env])

        # Load VecNormalize wrapper and model
        eval_env = VecNormalize.load(vecnorm_file, eval_env)

        # CRITICAL: Set VecNormalize to evaluation mode to freeze statistics
        eval_env.training = False
        eval_env.norm_obs = True  # Keep observation normalization active
        eval_env.norm_reward = False  # Disable reward normalization during evaluation
        
        print("--- VecNormalize set to evaluation mode (statistics frozen) ---")

        model = PPO.load(model_file)
        print(f"--- Model loaded successfully from {os.path.basename(model_file)} ---")

        # Retrieve and set the last training difficulty from MLflow metrics
        try:
            metric_history = client.get_metric_history(run_id, "tb_curriculum_current_difficulty")
            if metric_history:
                last_difficulty = metric_history[-1].value
                eval_env.envs[0].unwrapped.set_difficulty(last_difficulty)
                print(f"Set evaluation environment difficulty to last training value: {last_difficulty:.3f}")
            else:
                # Fallback: prompt user for difficulty if not found in metrics
                user_input = input("Training difficulty not found in metrics. Enter difficulty value (0.0-1.0): ").strip()
                new_difficulty = float(user_input)
                eval_env.envs[0].unwrapped.set_difficulty(new_difficulty)
                print(f"Set evaluation environment difficulty to user input: {new_difficulty:.3f}")
        except Exception as e:
            print(f"Warning: Could not set evaluation difficulty: {e}")

        # Clean up temporary files
        shutil.rmtree(artifact_dir)
        if os.path.exists(temp_dir):
            try:
                import time
                time.sleep(0.1)  # Brief delay to ensure file handles are released
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Note: Could not clean up temporary directory: {e}")

    return config, model, eval_env

def evaluate_varying_difficulty(run_id, model, eval_env, config, n_episodes=500, 
                               MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Evaluate the model across varying difficulty levels using random curriculum difficulties and create 
    histograms of success rate and episode length against the actual difficulty (as returned by the environment).
    Saves the plots as MLflow artifacts.
    
    Args:
        run_id: MLflow run ID
        model: Trained PPO model
        eval_env: Evaluation environment (VecEnv)
        config: Configuration dictionary containing curriculum trainer settings
        n_episodes: Total number of episodes to run at random difficulties
        MLFLOW_URI: MLflow tracking URI
        TEMP_DIR: Temporary directory for saving artifacts
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Use run-specific temp directory to avoid conflicts
    temp_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    with mlflow.start_run(run_id=run_id):
        # Extract difficulty limits from config
        curriculum_config = config.get("callbacks", {}).get("curriculum_trainer", {})
        min_difficulty = curriculum_config.get("min_difficulty", 0.0)
        max_difficulty = curriculum_config.get("max_difficulty", 1.0)
        
        print(f"--- Evaluating across difficulty range: {min_difficulty} to {max_difficulty} with {n_episodes} episodes ---")
        
        # Run episodes at random curriculum difficulties
        episode_results = []
        episode_lengths = []
        curriculum_difficulties = []
        actual_difficulties = []
        collision_results = []
        truncated_results = []
        
        for episode in range(n_episodes):
            # Sample a random curriculum difficulty
            curriculum_difficulty = np.random.uniform(min_difficulty, max_difficulty)
            curriculum_difficulties.append(curriculum_difficulty)
            
            # Set the difficulty for this evaluation
            eval_env.envs[0].unwrapped.set_difficulty(curriculum_difficulty)
            
            # Run single episode
            obs = eval_env.reset()
            done = [False]
            step_count = 0
            
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = eval_env.step(action)
                step_count += 1
            
            # Extract episode info
            ep_info = info[0] if info and len(info) > 0 else {}
            is_success = ep_info.get("is_success", False)
            is_collision = ep_info.get("is_collision", False)
            # Check if episode was truncated (hit max steps)
            is_truncated = ep_info.get("is_truncated", False)
            actual_difficulty = ep_info.get("difficulty", curriculum_difficulty)  # Fall back to curriculum difficulty if not available
            
            episode_results.append(is_success)
            collision_results.append(is_collision)
            truncated_results.append(is_truncated)
            episode_lengths.append(step_count)
            actual_difficulties.append(actual_difficulty)
            
            if (episode + 1) % 50 == 0:
                print(f"  Completed {episode + 1}/{n_episodes} episodes")
        
        # Convert to numpy arrays for easier manipulation
        episode_results = np.array(episode_results)
        collision_results = np.array(collision_results)
        truncated_results = np.array(truncated_results)
        episode_lengths = np.array(episode_lengths)
        actual_difficulties = np.array(actual_difficulties)
        curriculum_difficulties = np.array(curriculum_difficulties)
        
        # Create histogram plots (only 2 plots now)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Define difficulty bins for histograms
        n_bins = 15
        difficulty_bins = np.linspace(min(actual_difficulties), max(actual_difficulties), n_bins)

        # Plot 1: Histogram of actual difficulties
        ax1.hist(actual_difficulties, bins=difficulty_bins, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Actual Difficulty Level')
        ax1.set_ylabel('Number of Episodes')
        ax1.set_title('Distribution of Actual Difficulty Levels')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Stacked bar chart showing success, collision, truncated, and other termination proportions by difficulty bins
        success_rates_binned = []
        collision_rates_binned = []
        truncated_rates_binned = []
        other_rates_binned = []
        difficulty_bin_centers = []

        for i in range(len(difficulty_bins) - 1):
            mask = (actual_difficulties >= difficulty_bins[i]) & (actual_difficulties < difficulty_bins[i + 1])
            if np.any(mask):
                total_episodes_in_bin = np.sum(mask)
                success_rate = np.sum(episode_results[mask]) / total_episodes_in_bin
                collision_rate = np.sum(collision_results[mask]) / total_episodes_in_bin
                truncated_rate = np.sum(truncated_results[mask]) / total_episodes_in_bin
                # Compute 'other termination' as those not success, collision, or truncated
                other_mask = ~(
                    episode_results[mask] |
                    collision_results[mask] |
                    truncated_results[mask]
                )
                other_rate = np.sum(other_mask) / total_episodes_in_bin

                success_rates_binned.append(success_rate)
                collision_rates_binned.append(collision_rate)
                truncated_rates_binned.append(truncated_rate)
                other_rates_binned.append(other_rate)
                difficulty_bin_centers.append((difficulty_bins[i] + difficulty_bins[i + 1]) / 2)

        # Create stacked bar chart
        width = np.diff(difficulty_bins)[0] * 0.8
        ax2.bar(difficulty_bin_centers, success_rates_binned, width=width,
            alpha=0.8, color='green', edgecolor='black', label='Success')
        ax2.bar(difficulty_bin_centers, collision_rates_binned, width=width,
            bottom=success_rates_binned, alpha=0.8, color='red', edgecolor='black', label='Collision')

        # Calculate bottom for truncated bars (success + collision)
        bottom_truncated = np.array(success_rates_binned) + np.array(collision_rates_binned)
        ax2.bar(difficulty_bin_centers, truncated_rates_binned, width=width,
            bottom=bottom_truncated, alpha=0.8, color='orange', edgecolor='black', label='Truncated')

        # Calculate bottom for other termination bars (success + collision + truncated)
        bottom_other = bottom_truncated + np.array(truncated_rates_binned)
        ax2.bar(difficulty_bin_centers, other_rates_binned, width=width,
            bottom=bottom_other, alpha=0.8, color='gray', edgecolor='black', label='Other Termination')

        ax2.set_xlabel('Actual Difficulty Level')
        ax2.set_ylabel('Proportion of Episodes')
        ax2.set_title('Episode Outcomes vs Actual Difficulty (Stacked)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()

        # Save the plot as an artifact
        plot_path = os.path.join(temp_dir, "difficulty_evaluation_histograms.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path, artifact_path="evaluation")
        plt.close(fig)
        os.unlink(plot_path)

        # Also log the raw data as a CSV
        results_df = pd.DataFrame({
        'episode': range(n_episodes),
        'curriculum_difficulty': curriculum_difficulties,
        'actual_difficulty': actual_difficulties,
        'is_success': episode_results,
        'is_collision': collision_results,
        'is_truncated': truncated_results,
        'episode_length': episode_lengths
        })

        csv_path = os.path.join(temp_dir, "difficulty_evaluation_data.csv")
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="evaluation")
        os.unlink(csv_path)
            
        # Log summary metrics
        overall_success_rate = np.mean(episode_results)
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = np.std(episode_lengths)
        
        mlflow.log_metric("eval_overall_success_rate", overall_success_rate)
        mlflow.log_metric("eval_mean_episode_length", mean_episode_length)
        mlflow.log_metric("eval_std_episode_length", std_episode_length)
        mlflow.log_metric("eval_min_episode_length", min(episode_lengths))
        mlflow.log_metric("eval_max_episode_length", max(episode_lengths))
        mlflow.log_metric("eval_n_episodes", n_episodes)
    
    print(f"--- Difficulty evaluation completed and saved to MLflow ---")
    print(f"Overall success rate: {overall_success_rate:.3f}")
    print(f"Episode length: {mean_episode_length:.1f}±{std_episode_length:.1f} (range: {min(episode_lengths)}-{max(episode_lengths)})")
    print(f"Actual difficulty range: {min(actual_difficulties):.3f} - {max(actual_difficulties):.3f}")
    print(f"Curriculum difficulty range: {min(curriculum_difficulties):.3f} - {max(curriculum_difficulties):.3f}")

if __name__ == '__main__':
    print("RL Evaluation Script")
    print("=" * 50)
    
    # Get run ID from user
    run_id_input = input("Enter the MLflow Run ID to evaluate: ").strip()
    if not run_id_input:
        print("No Run ID provided. Exiting.")
        exit(1)
    
    # Get evaluation options from user
    print("\nEvaluation Options:")
    render_video = input("Record performance video? (y/n): ").lower() == 'y'
    save_statistics = input("Save VecNormalize statistics? (y/n): ").lower() == 'y'
    run_shap = input("Run SHAP evaluation? (y/n): ").lower() == 'y'
    run_difficulty_eval = input("Run difficulty evaluation? (y/n): ").lower() == 'y'
    
    # Optional difficulty override
    difficulty_val = None
    set_difficulty = input("Override evaluation difficulty? (leave blank for training value): ").strip()
    if set_difficulty:
        try:
            difficulty_val = float(set_difficulty)
        except ValueError:
            print("Invalid difficulty value. Using training default.")
            difficulty_val = None
    
    # Get run information and start evaluation
    mlflow.set_tracking_uri(MLFLOW_URI)
    run = mlflow.get_run(run_id_input)
    run_name = run.data.tags.get('mlflow.runName', 'unknown_run')
    print(f"\nStarting evaluation for run: {run_name} (ID: {run_id_input})")
    
    # Load model and environment from MLflow artifacts
    config, model, eval_env = load_trained_model_and_envs(run_id_input)
    
    # Override difficulty if requested
    if difficulty_val is not None:
        try:
            eval_env.envs[0].unwrapped.set_difficulty(difficulty_val)
            print(f"Overrode evaluation difficulty to: {difficulty_val:.3f}")
        except Exception as e:
            print(f"Failed to override difficulty: {e}")
    
    # Execute selected evaluation tasks
    print("\nRunning evaluations...")
    
    if render_video:
        print("Recording performance videos...")
        record_final_performance(run_id_input, model, eval_env)
    
    if save_statistics:
        print("Generating VecNormalize statistics...")
        plot_and_save_stats(run_id_input, eval_env, "eval_env")
    
    if run_shap:
        print("Running SHAP feature importance analysis...")
        evaluate_with_SHAP(run_id_input, model, eval_env)
    
    if run_difficulty_eval:
        print("Running difficulty evaluation...")
        evaluate_varying_difficulty(run_id_input, model, eval_env, config)
    
    # Cleanup and completion message
    del model
    eval_env.close()
    
    print(f"\nEvaluation completed for run: {run_name}")
    print(f"View results: mlflow ui --backend-store-uri {MLFLOW_URI}")
    print("=" * 50)
