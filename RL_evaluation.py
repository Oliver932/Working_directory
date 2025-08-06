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
    Evaluate the model using SHAP and log feature importance plots to MLflow for each output dimension.
    Collects background and evaluation datasets, computes SHAP values, and logs grouped bar plots.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    with mlflow.start_run(run_id=run_id):
        # Use all keys from the observation space, in order
        obs_keys = list(eval_env.observation_space.spaces.keys())

        # 1. Collect background dataset (observations from random rollouts)
        obs_list = []
        for _ in range(n_background):
            obs = eval_env.reset()
            done = [False]
            while not done[0]:
                # Flatten dict obs if needed
                if isinstance(obs, dict):
                    flat_obs = np.concatenate([obs[k].flatten() for k in obs_keys if k in obs])
                else:
                    flat_obs = obs.flatten()
                obs_list.append(flat_obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
                if len(obs_list) >= n_background:
                    break
            if len(obs_list) >= n_background:
                break
        background = np.stack(obs_list[:n_background])

        # 2. Collect evaluation dataset (for SHAP explanation)
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

        # 4. Run SHAP KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(eval_data)

        # 5. Generate feature importance plot
        temp_dir = TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)

        # Prepare feature names
        feature_names = []
        for k in obs_keys:
            space = eval_env.observation_space.spaces[k]
            if np.prod(space.shape) == 1:
                feature_names.append(k)
            else:
                for j in range(np.prod(space.shape)):
                    feature_names.append(f"{k}[{j}]")

        # Compute mean absolute SHAP values for each feature and output
        if isinstance(shap_values, list):
            shap_arr = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_outputs)
        else:
            shap_arr = shap_values  # already (n_samples, n_features, n_outputs)
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)  # (n_features, n_outputs)

        # Create grouped bar chart
        n_features, n_outputs = mean_abs_shap.shape
        x = np.arange(n_features)
        bar_width = 0.8 / n_outputs

        plt.figure(figsize=(max(12, n_features // 2), 6))
        for i in range(n_outputs):
            plt.bar(x + i * bar_width, mean_abs_shap[:, i], width=bar_width, label=f'Output {i}')

        plt.xticks(x + bar_width * (n_outputs - 1) / 2, feature_names, rotation=90)
        plt.ylabel("Mean |SHAP value|")
        plt.title("Feature Importance per Output Dimension")
        plt.legend()
        plt.tight_layout()

        # Save and log plot
        shap_grouped_bar_path = os.path.join(temp_dir, "shap_grouped_bar.png")
        plt.savefig(shap_grouped_bar_path)
        mlflow.log_artifact(shap_grouped_bar_path, artifact_path="shap")
        plt.close()
        os.unlink(shap_grouped_bar_path)
    
    print("--- Grouped SHAP bar plot (per output) saved and logged to MLflow ---")

def record_final_performance(run_id, model, eval_env, n_episodes=5, prefix="final", MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Run the trained model in the evaluation environment, record videos, and log them to MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    with mlflow.start_run(run_id=run_id):
        overview_frames, robot_perspective_frames = [], []
        print(f"--- Recording {prefix} performance with {n_episodes} episodes ---")

        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = [False]
            while not done[0]:
                overview_frames.append(eval_env.envs[0].unwrapped.render(camera_name='scene_overview'))
                robot_perspective_frames.append(eval_env.envs[0].unwrapped.render(camera_name='robot_perspective'))
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)

        # Use temporary files
        temp_dir = TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)

        overview_vid_path = os.path.join(temp_dir, f"{prefix}_replay_overview.mp4")
        robot_vid_path = os.path.join(temp_dir, f"{prefix}_replay_robot.mp4")

        print(f"--- Saving overview video to {overview_vid_path} ---")
        imageio.mimwrite(overview_vid_path, [np.array(frame) for frame in overview_frames], fps=30, format='FFMPEG')
        print(f"--- Saving robot perspective video to {robot_vid_path} ---")
        imageio.mimwrite(robot_vid_path, [np.array(frame) for frame in robot_perspective_frames], fps=30, format='FFMPEG')

        # Log the final videos as artifacts to the active MLflow run
        mlflow.log_artifact(overview_vid_path, artifact_path="replays")
        mlflow.log_artifact(robot_vid_path, artifact_path="replays")

        # Clean up temporary files
        os.unlink(overview_vid_path)
        os.unlink(robot_vid_path)

    print(f"--- {prefix} performance videos logged to MLflow ---")

def plot_and_save_stats(run_id, vecnorm, name_prefix, MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Generate and save VecNormalize statistics as a CSV file and log it to MLflow.
    Handles both dict and array observation spaces.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    with mlflow.start_run(run_id=run_id):
        obs_rms = vecnorm.obs_rms
        summary_rows = []
        
        # Extract normalized observation keys from VecNormalize object
        if isinstance(obs_rms, dict):
            norm_obs_keys = list(obs_rms.keys())
        else:
            # Fallback: if it's not a dict obs space, use a generic key
            norm_obs_keys = ['obs']
        
        print(f"--- Detected normalized observation keys for stats: {norm_obs_keys} ---")
        
        # For Dict obs space with norm_obs_keys, obs_rms is a dict of RunningMeanStd objects
        if isinstance(obs_rms, dict):
            keys = [k for k in obs_rms.keys() if k in norm_obs_keys]
            if not keys:
                print(f"Warning: No normalized observation keys found in obs_rms for {name_prefix}. Skipping summary table.")
                return
            for key in keys:
                rms = obs_rms[key]
                mean_vals = getattr(rms, 'mean', None)
                var_vals = getattr(rms, 'var', None)
                if mean_vals is None or var_vals is None:
                    print(f"Warning: Missing mean/var for key '{key}' in obs_rms. Skipping.")
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

        # Reward stats
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
            print(f"Warning: ret_rms missing mean/var for {name_prefix}. Skipping reward stats in summary table.")

        if summary_rows:
            temp_dir = TEMP_DIR
            os.makedirs(temp_dir, exist_ok=True)
            
            df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(temp_dir, f"{name_prefix}_vecnormalize_summary.csv")
            df.to_csv(summary_path, index=False)
            mlflow.log_artifact(summary_path, artifact_path="vecnormalize_stats")
            os.unlink(summary_path)
    
    print(f"--- {name_prefix} VecNormalize stats saved to MLflow ---")

# VERY USEFUL FUNCTION FOR LOADING ANY MODEL AND ENVIRONMENTS FROM MLFLOW
def load_trained_model_and_envs(run_id, MLFLOW_URI=MLFLOW_URI, TEMP_DIR=TEMP_DIR):
    """
    Load a trained model, config, and VecNormalize wrapper from MLflow artifacts and create evaluation environments.
    Returns the config, model, evaluation environment, and normalized observation keys.
    """
    
    # Set MLflow tracking URI and start run context
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    with mlflow.start_run(run_id=run_id):
        # Setup temporary directory
        temp_dir = TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download all artifacts in one call
        artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_dir)
        
        # Load config
        config_files = glob.glob(os.path.join(artifact_dir, "config_*.json"))
        if not config_files:
            raise FileNotFoundError("No config_*.json file found in MLflow artifacts.")
        
        with open(config_files[0], "r") as f:
            config = json.load(f)
        
        # Find model and VecNormalize files in the model subdirectory
        model_path = os.path.join(artifact_dir, "model")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model directory not found in MLflow artifacts")
        
        # Find model and VecNormalize files
        model_file = None
        vecnorm_file = None
        
        for file in os.listdir(model_path):
            if file.endswith('.zip') and 'ppo_model' in file:
                model_file = os.path.join(model_path, file)
            elif file.endswith('.pkl') and 'vecnormalize' in file:
                vecnorm_file = os.path.join(model_path, file)
        
        if not model_file or not vecnorm_file:
            raise FileNotFoundError("Could not find model or VecNormalize files in MLflow artifacts")
        
        print(f"--- Loading model from {model_file} ---")
        print(f"--- Loading VecNormalize from {vecnorm_file} ---")
        
        # Create evaluation environment
        def make_eval_env():
            env = CustomRobotEnv(render_mode='rgb_array', config=config["environment"])
            env = Monitor(env)
            return env

        eval_env = DummyVecEnv([make_eval_env])

        # Load VecNormalize wrapper and model
        eval_env = VecNormalize.load(vecnorm_file, eval_env)
        model = PPO.load(model_file)

        # --- NEW: Pull last difficulty from MLflow metrics and set it ---
        client = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(run_id, "tb_curriculum_current_difficulty")
        if metric_history:
            last_difficulty = metric_history[-1].value
            try:
                eval_env.envs[0].unwrapped.set_difficulty(last_difficulty)
                print(f"Set evaluation environment difficulty to last trained value: {last_difficulty}")
            except Exception as e:
                print(f"Failed to set difficulty from MLflow metric: {e}")

        # Clean up downloaded artifacts
        shutil.rmtree(artifact_dir)

    return config, model, eval_env

if __name__ == '__main__':
    print("--- RL Evaluation Script ---")
    
    # Get run ID from user
    run_id_input = input("Enter the MLflow Run ID to evaluate: ").strip()
    if not run_id_input:
        print("No Run ID provided. Exiting.")
        exit(1)
    
    # Get evaluation options
    render_video = input("Render final performance video? (y/n): ").lower() == 'y'
    save_statistics = input("Save VecNormalize statistics? (y/n): ").lower() == 'y'
    run_shap = input("Run SHAP evaluation? (y/n): ").lower() == 'y'
    
    difficulty_val = None
    set_difficulty = input("Set evaluation environment difficulty? (leave blank for last training value): ").strip()
    if set_difficulty:
        try:
            difficulty_val = float(set_difficulty)
        except ValueError:
            print("Invalid difficulty value. Using default.")
            difficulty_val = None
    
    # Get run name for display
    mlflow.set_tracking_uri(MLFLOW_URI)
    run = mlflow.get_run(run_id_input)
    run_name = run.data.tags.get('mlflow.runName', 'unknown_run')
    print(f"--- Starting evaluation for run: {run_name} (ID: {run_id_input}) ---")
    
    # Load model and environments
    config, model, eval_env = load_trained_model_and_envs(run_id_input)
    
    # Set difficulty if requested
    if difficulty_val is not None:
        try:
            eval_env.envs[0].unwrapped.set_difficulty(difficulty_val)
            print(f"Set evaluation environment difficulty to {difficulty_val}")
        except Exception as e:
            print(f"Failed to set difficulty: {e}")
    
    # Run evaluations
    if render_video:
        print("--- Recording final performance of the trained agent ---")
        record_final_performance(run_id_input, model, eval_env)
    
    if save_statistics:
        print("--- Generating and saving VecNormalize statistics ---")
        plot_and_save_stats(run_id_input, eval_env, "eval_env")
    
    if run_shap:
        print("--- Running SHAP evaluation ---")
        evaluate_with_SHAP(run_id_input, model, eval_env)
    
    # Clean up
    del model
    eval_env.close()
    
    print(f"--- Evaluation completed for run: {run_name} ---")
    print(f"To view results, run 'mlflow ui --backend-store-uri {MLFLOW_URI}' in your terminal.")
