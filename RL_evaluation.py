import os
import json
import numpy as np
import pandas as pd
import imageio
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import custom modules for the robot simulation
from RL_implementation import CustomRobotEnv

def load_config(config_path="config.json"):
    """Load configuration from JSON file with error handling."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the configuration file exists.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing config file: {e}. Please check the JSON syntax in {config_path}.")

def record_final_performance(model, eval_env, train_env, config, n_episodes=5, prefix="final"):
    """Record the performance of a trained model and log videos to MLflow."""
    
    def _sync_difficulty():
        """Synchronize difficulty from training environment to evaluation environment."""
        if train_env is not None:
            # Get current difficulty from training environment
            current_difficulty = train_env.env_method('get_difficulty')[0]
            # Set the same difficulty in evaluation environment
            eval_env.env_method('set_difficulty', current_difficulty)
            print(f"--- Synced difficulty to evaluation environment: {current_difficulty:.2f} ---")
    
    # Sync difficulty from training to evaluation environment
    _sync_difficulty()
    
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
    temp_dir = "C:/temp/artifacts"
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

def plot_and_save_stats(vecnorm, name_prefix, norm_obs_keys):
    """Generate and save VecNormalize statistics as CSV to MLflow."""
    obs_rms = vecnorm.obs_rms
    summary_rows = []
    
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
        temp_dir = "C:/temp/artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(temp_dir, f"{name_prefix}_vecnormalize_summary.csv")
        df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="vecnormalize_stats")
        os.unlink(summary_path)
        print(f"--- {name_prefix} VecNormalize stats saved to MLflow ---")

def load_trained_model_and_envs(run_id, run_name, config):
    """Load a trained model and create environments for evaluation."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///C:/temp/mlruns")
    
    # Load the model and VecNormalize from MLflow artifacts
    temp_dir = "C:/temp/artifacts"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download model artifacts
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, 
        artifact_path="model",
        dst_path=temp_dir
    )
    
    # Find the model and vecnormalize files
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
    
    # Create environments
    MONITOR_KEYWORDS = ("is_success", "is_collision", "is_invalid_move", "grip_attempts", "failed_grips", "difficulty", "visible_steps", "calculable_steps")
    
    def make_train_env():
        env_config = config["environment"].copy()
        env_config["curriculum"] = config["callbacks"]["curriculum_trainer"]
        env = CustomRobotEnv(render_mode=None, config=env_config)
        env = Monitor(env, info_keywords=MONITOR_KEYWORDS)
        return env
    
    def make_eval_env():
        env = CustomRobotEnv(render_mode='rgb_array', config=config["environment"])
        env = Monitor(env)
        return env
    
    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Specify which observation keys to normalize (only Box spaces, not MultiBinary)
    norm_obs_keys = [
        "ellipse_position", "delta_ellipse_position",
        "ellipse_semi_major_vector", "delta_ellipse_semi_major_vector", 
        "ellipse_semi_minor_vector", "delta_ellipse_semi_minor_vector",
        "actuator_extensions", "delta_extensions",
        "E1_position", "delta_E1", "E1_quaternion", "delta_E1_quaternion"
    ]
    
    # Load VecNormalize
    train_env = VecNormalize.load(vecnorm_file, train_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., norm_obs_keys=norm_obs_keys)
    # Synchronize statistics
    eval_env.obs_rms = train_env.obs_rms
    
    # Load model
    model = PPO.load(model_file)
    
    # Clean up downloaded files
    os.unlink(model_file)
    os.unlink(vecnorm_file)
    os.rmdir(model_path)
    
    return model, train_env, eval_env, norm_obs_keys

def run_evaluation(run_id, render_final_video=True, save_stats=True):
    """Run evaluation for a trained model using its MLflow run ID."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///C:/temp/mlruns")
    
    # Get run information to retrieve the run name and config
    run = mlflow.get_run(run_id)
    run_name = run.data.tags.get('mlflow.runName', 'unknown_run')
    
    print(f"--- Starting evaluation for run: {run_name} (ID: {run_id}) ---")
    
    # Load configuration
    config = load_config("config.json")
    
    # Continue the existing MLflow run to add evaluation artifacts
    with mlflow.start_run(run_id=run_id):
        print(f"--- Resuming MLflow Run (Run ID: {run_id}) ---")
        
        # Load model and environments
        model, train_env, eval_env, norm_obs_keys = load_trained_model_and_envs(run_id, run_name, config)
        
        if render_final_video:
            print("--- Recording final performance of the trained agent ---")
            record_final_performance(
                model, 
                eval_env,
                train_env,
                config,
                n_episodes=config["evaluation"]["n_video_episodes"], 
                prefix="final"
            )
        
        # Save VecNormalize stats plots if requested
        if save_stats:
            print("--- Generating and saving VecNormalize statistics ---")
            plot_and_save_stats(train_env, "train_env", norm_obs_keys)
            plot_and_save_stats(eval_env, "eval_env", norm_obs_keys)
        
        # Clean up
        del model
        train_env.close()
        eval_env.close()
        
        print(f"--- Evaluation completed for run: {run_name} ---")

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
    
    try:
        run_evaluation(run_id_input, render_final_video=render_video, save_stats=save_statistics)
        print("--- Evaluation completed successfully ---")
        print(f"To view results, run 'mlflow ui --backend-store-uri file:///C:/temp/mlruns' in your terminal.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
