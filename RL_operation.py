from RL_training import CustomRobotEnv
from RL_evaluation import load_trained_model_and_envs

import serial
import time
import matplotlib.pyplot as plt
import numpy as np

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

    # Create RealRobot instance for managing robot communication
    real_robot = RealRobot(config, arduino)

    # 4. Reset environment
    obs = env.reset()
    done = False
    step_count = 0

    # synchronise the real robot
    print("Synchronizing real robot with environment...")
    real_robot.set_robot_position(env.envs[0].unwrapped.robot.extensions)

    # --- Setup live display windows ---
    plt.ion()
    # Get initial images for both cameras
    try:
        overview_img = env.envs[0].unwrapped.render(camera_name='scene_overview')
        robot_img = env.envs[0].unwrapped.render(camera_name='robot_perspective')
    except Exception:
        overview_img = robot_img = None

    fig, (ax_overview, ax_robot) = plt.subplots(1, 2, figsize=(10, 5))
    if overview_img is not None:
        img_overview = ax_overview.imshow(overview_img)
    else:
        img_overview = ax_overview.imshow([[0]])
    ax_overview.set_title('Overview Camera')
    if robot_img is not None:
        img_robot = ax_robot.imshow(robot_img)
    else:
        img_robot = ax_robot.imshow([[0]])
    ax_robot.set_title('Robot Perspective')
    plt.tight_layout()
    plt.show(block=False)

    # Ask user if manual confirmation is required after each step
    wait_for_user = input("Wait for user confirmation after each robot action? (y/N): ").strip().lower() == 'y'

    print("\n--- Running one episode ---")
    while not done:

        # Optionally wait for user confirmation before next step
        if wait_for_user:
            input("Press Enter to continue to the next step...")

        # 5. Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # 6. Step environment
        obs, reward, done, info = env.step(action)
        step_count += 1

        # --- Live update of camera views ---
        try:
            overview_img = env.envs[0].unwrapped.render(camera_name='scene_overview')
            robot_img = env.envs[0].unwrapped.render(camera_name='robot_perspective')
            img_overview.set_data(overview_img)
            img_robot.set_data(robot_img)
            fig.canvas.draw()
            fig.canvas.flush_events()

        except Exception as e:
            print(f"[Warning] Could not render camera views: {e}")

        
        # Send the extensions calculated by the robot kinematics to the real robot
        real_robot.set_robot_position(env.envs[0].unwrapped.robot.extensions)
        
        if (env.envs[0].unwrapped.robot.gripped and not real_robot.gripped):
            # If the robot is gripped in the environment but not in the real robot, grip it
            real_robot.set_gripper_state('GRIP')

        elif (not env.envs[0].unwrapped.robot.gripped and real_robot.gripped):
            # If the robot is not gripped in the environment but is in the real robot, ungrip it
            real_robot.set_gripper_state('UNGRIP')

    print("Episode finished. Closing serial connection.")
    real_robot.close()
    plt.ioff()
    plt.close(fig)



if __name__ == "__main__":
    main()


