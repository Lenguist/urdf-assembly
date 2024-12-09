# hillclimber.py

import os
import json
import pybullet as p
import pybullet_data
import time
import math
import random
from shutil import rmtree
from datetime import datetime

# ------------------------ Configuration Parameters ------------------------ #

# Simulation Parameters
SIMULATION_FPS = 240          # Simulation steps per second
RUN_DURATION = 10             # Duration of each run in seconds
TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
SETTLING_TIME = 0.5           # Settling time in seconds
SETTLING_STEPS = int(SIMULATION_FPS * SETTLING_TIME)

# Hill Climber Parameters
MAX_ITERATIONS = 1000       # Maximum number of iterations (adjust as needed)
MUTATION_RATE = 0.9           # Probability of each parameter being mutated
MUTATION_SCALE = 0.9         # Scale of mutation (5%)

# Logging Parameters
TRAINING_RUNS_DIR = "training_runs/hillclimber"  # Directory to store training runs (not backed up with git)
RESULTS_DIR = "results/hillclimber"             # Directory to store results (backed up with git)

# Gait Parameters Ranges (degrees)
A_MIN, A_MAX = 0, 10             # Range for 'a' parameters
B_MIN, B_MAX = 0, 10             # Range for 'b' parameters
C_MIN, C_MAX = 0, 2 * math.pi    # Range for 'c' parameters
OMEGA_MIN, OMEGA_MAX = 0.5, 10    # Range for 'omega' parameter

# Motor Force Limits (Adjusted for better movement)
HIP_MAX_FORCE = 100    # Increased from 150
KNEE_MAX_FORCE = 100   # Increased from 150

# Position and Velocity Gains (Fine-tuned for responsiveness)
POSITION_GAIN = 1      # Increased for better responsiveness
VELOCITY_GAIN = 1      # Increased for better responsiveness

# Joint Indices (as per your URDF)
HIP_JOINTS = [0, 2, 4, 6]
KNEE_JOINTS = [1, 3, 5, 7]
ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS

# ------------------------ Gait Parameters Class ------------------------ #

class GaitParameters:
    def __init__(self, a=None, b=None, c=None, omega=None):
        """
        Initialize gait parameters.
        Each joint has its own a, b, c parameters.
        Omega is shared across all joints.
        """
        self.a = a if a is not None else {joint: random.uniform(A_MIN, A_MAX) for joint in ALL_JOINTS}
        self.b = b if b is not None else {joint: random.uniform(B_MIN, B_MAX) for joint in ALL_JOINTS}
        self.c = c if c is not None else {joint: random.uniform(C_MIN, C_MAX) for joint in ALL_JOINTS}
        self.omega = omega if omega is not None else random.uniform(OMEGA_MIN, OMEGA_MAX)

    def mutate(self):
        """
        Mutate the gait parameters by a small random amount.
        Each parameter has a chance to be mutated based on MUTATION_RATE.
        """
        for joint in ALL_JOINTS:
            if random.random() < MUTATION_RATE:
                # Mutate 'a' parameter
                delta_a = self.a[joint] * MUTATION_SCALE
                self.a[joint] += random.uniform(-delta_a, delta_a)
                self.a[joint] = max(A_MIN, min(A_MAX, self.a[joint]))
                
            if random.random() < MUTATION_RATE:
                # Mutate 'b' parameter
                delta_b = self.b[joint] * MUTATION_SCALE
                self.b[joint] += random.uniform(-delta_b, delta_b)
                self.b[joint] = max(B_MIN, min(B_MAX, self.b[joint]))
                
            if random.random() < MUTATION_RATE:
                # Mutate 'c' parameter
                delta_c = MUTATION_SCALE * math.pi  # Scale mutation appropriately
                self.c[joint] += random.uniform(-delta_c, delta_c)
                self.c[joint] = (self.c[joint] + 2 * math.pi) % (2 * math.pi)  # Keep within [0, 2π]
        
        # Mutate 'omega' parameter
        if random.random() < MUTATION_RATE:
            delta_omega = (OMEGA_MAX - OMEGA_MIN) * MUTATION_SCALE
            self.omega += random.uniform(-delta_omega, delta_omega)
            self.omega = max(OMEGA_MIN, min(OMEGA_MAX, self.omega))

    def copy(self):
        """
        Create a deep copy of the current parameters.
        """
        return GaitParameters(a=self.a.copy(), b=self.b.copy(), c=self.c.copy(), omega=self.omega)

    def to_dict(self):
        """
        Convert parameters to a dictionary for logging.
        """
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "omega": self.omega
        }

    def __str__(self):
        """
        String representation for debugging.
        """
        return json.dumps(self.to_dict(), indent=4)

# ------------------------ Run Naming Function ------------------------ #

def generate_run_name():
    """
    Generate a unique run name using a randomly selected pair of words.
    
    Returns:
        str: The generated run name in the format 'word1_word2'.
    """
    adjectives = [
        "adorable", "adventurous", "agreeable", "alert", "amazing", "amusing", 
        "angelic", "artistic", "athletic", "awesome", "beautiful", "benevolent", 
        "bubbly", "captivating", "charming", "cheerful", "clever", "confident", 
        "courageous", "creative", "curious", "dazzling", "determined", "diligent", 
        "dynamic", "elegant", "empathetic", "energetic", "enthusiastic", "fantastic", 
        "friendly", "generous", "gentle", "gorgeous", "grateful", "harmonious", 
        "heroic", "hilarious", "hopeful", "imaginative", "independent", "ingenious", 
        "inspiring", "intelligent", "jovial", "joyful", "kindhearted", "legendary", 
        "lovable", "magnificent", "majestic", "marvelous", "mellow", "mindful", 
        "modest", "noble", "optimistic", "outgoing", "passionate", "peaceful", 
        "persevering", "philosophical", "playful", "poetic", "radiant", "resilient", 
        "resourceful", "romantic", "sensational", "serene", "spirited", "stunning", 
        "sympathetic", "tenacious", "thoughtful", "tranquil", "trustworthy", 
        "vibrant", "victorious", "vivacious", "wise", "wonderful", "zesty", 
        "zippy", "zealous"
    ]

    nouns = [
        "acorn", "aurora", "beach", "blossom", "breeze", "butterfly", "canyon", 
        "cliff", "comet", "coral", "crystal", "daisy", "desert", "dove", "dune", 
        "echo", "feather", "fern", "flame", "flower", "fog", "frost", "galaxy", 
        "gem", "glacier", "glow", "grass", "horizon", "iceberg", "jungle", 
        "lake", "light", "lily", "meadow", "mist", "nebula", "orchid", 
        "pebble", "petal", "phoenix", "pine", "prairie", "rainbow", "reef", 
        "ripple", "river", "rose", "sapphire", "savanna", "sea", "shadow", 
        "silhouette", "sky", "snowflake", "solstice", "spark", "starlight", 
        "stone", "stream", "sunbeam", "sunset", "thunder", "tide", "valley", 
        "violet", "volcano", "waterfall", "wave", "whisper", "wildflower", 
        "willow", "wind", "woods"
    ]

    
    word1 = random.choice(adjectives)
    word2 = random.choice(nouns)
    run_name = f"{word1}_{word2}"
    return run_name

# ------------------------ Initialization Functions ------------------------ #

def initialize_directories(run_name):
    """
    Initialize the directory structure for a given run.
    
    Args:
        run_name (str): The unique name of the training run.
    
    Returns:
        dict: Paths to the created directories.
    """
    training_run_dir = os.path.join(TRAINING_RUNS_DIR, run_name)
    results_run_dir = os.path.join(RESULTS_DIR, run_name)
    
    # Directories to be created
    dirs_to_create = [
        training_run_dir,
        os.path.join(training_run_dir, "logs"),
        results_run_dir,
        os.path.join(results_run_dir, "plots_best_run"),
        os.path.join(results_run_dir, "plots_training_run")
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "training_run": training_run_dir,
        "results_run": results_run_dir
    }

def save_hyperparameters(hyperparams, path):
    """
    Save hyperparameters to a JSON file.
    
    Args:
        hyperparams (dict): Dictionary of hyperparameters.
        path (str): Path to save the JSON file.
    """
    with open(path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Hyperparameters saved to '{path}'.")

def save_log(run_number, gait_params, distance, path):
    """
    Save the gait parameters and distance to a JSON file.
    
    Args:
        run_number (int): The current run number.
        gait_params (GaitParameters): The current gait parameters.
        distance (float): The distance traveled in this run.
        path (str): Path to save the JSON log file.
    """
    log_data = {
        "run_number": run_number,
        "distance_traveled_m": distance,
        "gait_parameters_deg": {
            joint: {
                "a": gait_params.a[joint],
                "b": gait_params.b[joint],
                "c": gait_params.c[joint]
            } for joint in ALL_JOINTS
        },
        "omega": gait_params.omega,
        "timestamp": datetime.now().isoformat()
    }
    with open(path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Log data saved to '{path}'.")

def save_best_run(best_run, path):
    """
    Save the best run's parameters to a JSON file.
    
    Args:
        best_run (dict): Dictionary containing the best run's data.
        path (str): Path to save the best run JSON file.
    """
    with open(path, 'w') as f:
        json.dump(best_run, f, indent=4)
    print(f"Best run data saved to '{path}'.")

def save_values_json(run_name, best_distance, best_params, directories):
    """
    Save the best run's values to a JSON file for visualization.
    
    Args:
        run_name (str): The unique name of the training run.
        best_distance (float): The best distance achieved.
        best_params (GaitParameters): The best gait parameters.
        directories (dict): Dictionary containing paths to training_run and results_run directories.
    """
    values_data = {
        "run_name": run_name,
        "best_distance_traveled_m": best_distance,
        "gait_parameters_deg": {
            joint: {
                "a": best_params.a[joint],
                "b": best_params.b[joint],
                "c": best_params.c[joint]
            } for joint in ALL_JOINTS
        },
        "omega": best_params.omega,
        "timestamp": datetime.now().isoformat()
    }
    values_json_path = os.path.join(directories["results_run"], "values.json")
    with open(values_json_path, 'w') as f:
        json.dump(values_data, f, indent=4)
    print(f"Values JSON saved to '{values_json_path}'.")

# ------------------------ Gait Evaluation Function ------------------------ #

def evaluate_gait(gait_params):
    """
    Evaluate the gait by simulating the robot's movement with the given gait parameters.
    Returns the net distance traveled along the X-axis after RUN_DURATION seconds.
    
    Args:
        gait_params (GaitParameters): The gait parameters to evaluate.
    
    Returns:
        float: Distance traveled in meters.
    """
    # Connect to PyBullet in DIRECT mode (no GUI)
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the plane
    plane_id = p.loadURDF("plane.urdf")

    # Rotate 90 degrees about X-axis (red axis)
    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # Drop the robot slightly above the ground
    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

    # Let the robot settle
    for _ in range(SETTLING_STEPS):
        p.stepSimulation()

    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    # Record initial position
    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]

    # Simulation loop
    for step in range(TOTAL_STEPS):
        t = step / SIMULATION_FPS  # Current time in seconds

        # Calculate and set joint angles based on gait parameters
        for joint in ALL_JOINTS:
            theta = gait_params.a[joint] + gait_params.b[joint] * math.sin(gait_params.omega * t + gait_params.c[joint])
            theta_rad = math.radians(theta)
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=HIP_MAX_FORCE if joint in HIP_JOINTS else KNEE_MAX_FORCE,
                positionGain=POSITION_GAIN,
                velocityGain=VELOCITY_GAIN
            )

        p.stepSimulation()

    # Record final position
    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]

    # Disconnect PyBullet
    p.disconnect()

    # Calculate distance traveled along X-axis
    distance_traveled = final_x - initial_x
    return distance_traveled

# ------------------------ Hill Climber Optimization ------------------------ #

def hill_climber_optimization(run_name, directories):
    """
    Perform hill climber optimization to find the gait parameters that maximize distance traveled.
    
    Args:
        run_name (str): The unique name of the training run.
        directories (dict): Dictionary containing paths to training_run and results_run directories.
    
    Returns:
        GaitParameters: The best gait parameters found.
        float: The best distance traveled.
    """
    # Initialize with random parameters
    current_params = GaitParameters()
    current_distance = evaluate_gait(current_params)
    run_number = 1
    log_path = os.path.join(directories["training_run"], "logs", f"log_{run_number}.json")
    save_log(run_number, current_params, current_distance, log_path)
    print(f"Run {run_number}: Distance Traveled = {current_distance:.3f} meters.")

    best_params = current_params.copy()
    best_distance = current_distance

    # Hill Climber Loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        # Create new parameters by mutating the current parameters
        new_params = current_params.copy()
        new_params.mutate()

        # Evaluate the new parameters
        new_distance = evaluate_gait(new_params)

        # Increment run number and log the new run
        run_number += 1
        log_path = os.path.join(directories["training_run"], "logs", f"log_{run_number}.json")
        save_log(run_number, new_params, new_distance, log_path)
        print(f"Run {run_number}: Distance Traveled = {new_distance:.3f} meters.")

        # Compare and decide to keep or discard
        if new_distance > best_distance:
            print(f"--> Improvement found! Updating best distance from {best_distance:.3f} to {new_distance:.3f} meters.")
            best_params = new_params.copy()
            best_distance = new_distance
            current_params = new_params.copy()
        else:
            print(f"--> No improvement. Keeping best distance at {best_distance:.3f} meters.")
            # Optionally, implement a mechanism to escape local maxima
            # For simplicity, we keep the current best

    print(f"\nOptimization Complete. Best Distance Traveled = {best_distance:.3f} meters.")

    return best_params, best_distance

# ------------------------ Visualization Function ------------------------ #

def visualize_best_gait(best_params):
    """
    Visualize the best gait in PyBullet's GUI mode.
    
    Args:
        best_params (GaitParameters): The best gait parameters to visualize.
    """
    print("\nStarting visualization of the best gait...")

    # Connect to PyBullet in GUI mode
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the plane
    plane_id = p.loadURDF("plane.urdf")

    # Rotate 90 degrees about X-axis (red axis)
    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    # Drop the robot slightly above the ground
    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

    # Let the robot settle
    for _ in range(SETTLING_STEPS):
        p.stepSimulation()
        time.sleep(1. / SIMULATION_FPS)

    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

    # Record initial position
    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]

    # Simulation loop for visualization (same duration as evaluation)
    for step in range(TOTAL_STEPS):
        t = step / SIMULATION_FPS  # Current time in seconds

        # Calculate and set joint angles based on best gait parameters
        for joint in ALL_JOINTS:
            theta = best_params.a[joint] + best_params.b[joint] * math.sin(best_params.omega * t + best_params.c[joint])
            theta_rad = math.radians(theta)
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=HIP_MAX_FORCE if joint in HIP_JOINTS else KNEE_MAX_FORCE,
                positionGain=POSITION_GAIN,
                velocityGain=VELOCITY_GAIN
            )

        p.stepSimulation()
        time.sleep(1. / SIMULATION_FPS)

    # Record final position
    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]

    # Calculate distance traveled
    distance_traveled = final_x - initial_x
    print(f"Visualization Complete. Distance Traveled = {distance_traveled:.3f} meters.")

    # Adjust the camera: Similar to 'hello_bullet.py'
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=50,     # Adjusted yaw for better perspective
        cameraPitch=-35,  # Adjusted pitch for better perspective
        cameraTargetPosition=[0, 0, 0]
    )

    print("Visualization of the best gait is now active. Close the window or interrupt the script to exit.")

    try:
        while True:
            p.stepSimulation()
            time.sleep(1. / SIMULATION_FPS)
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    finally:
        p.disconnect()

# ------------------------ Save Values to Results ------------------------ #

def save_values_json(run_name, best_distance, best_params, directories):
    """
    Save the best run's values to a JSON file for visualization.
    
    Args:
        run_name (str): The unique name of the training run.
        best_distance (float): The best distance achieved.
        best_params (GaitParameters): The best gait parameters.
        directories (dict): Dictionary containing paths to training_run and results_run directories.
    """
    values_data = {
        "run_name": run_name,
        "best_distance_traveled_m": best_distance,
        "gait_parameters_deg": {
            joint: {
                "a": best_params.a[joint],
                "b": best_params.b[joint],
                "c": best_params.c[joint]
            } for joint in ALL_JOINTS
        },
        "omega": best_params.omega,
        "timestamp": datetime.now().isoformat()
    }
    values_json_path = os.path.join(directories["results_run"], "values.json")
    with open(values_json_path, 'w') as f:
        json.dump(values_data, f, indent=4)
    print(f"Values JSON saved to '{values_json_path}'.")

# ------------------------ Main Execution Flow ------------------------ #

def main():
    # Generate a unique run name
    run_name = generate_run_name()
    print(f"Starting training run: '{run_name}'")

    # Initialize directories
    directories = initialize_directories(run_name)

    # Define initial hyperparameters
    initial_hyperparams = {
        "simulation_fps": SIMULATION_FPS,
        "run_duration": RUN_DURATION,
        "settling_time": SETTLING_TIME,
        "max_iterations": MAX_ITERATIONS,
        "mutation_rate": MUTATION_RATE,
        "mutation_scale": MUTATION_SCALE,
        "omega_min": OMEGA_MIN,
        "omega_max": OMEGA_MAX,
        "a_min": A_MIN,
        "a_max": A_MAX,
        "b_min": B_MIN,
        "b_max": B_MAX,
        "c_min": C_MIN,
        "c_max": C_MAX,
        "hip_max_force": HIP_MAX_FORCE,
        "knee_max_force": KNEE_MAX_FORCE,
        "position_gain": POSITION_GAIN,
        "velocity_gain": VELOCITY_GAIN,
        "joint_indices": {
            "hip_joints": HIP_JOINTS,
            "knee_joints": KNEE_JOINTS
        }
    }

    # Save hyperparameters to training_runs and results directories
    save_hyperparameters(initial_hyperparams, os.path.join(directories["training_run"], "training_hyperparameters.json"))
    save_hyperparameters(initial_hyperparams, os.path.join(directories["results_run"], "training_hyperparameters.json"))

    # Perform hill-climbing optimization
    best_params, best_distance = hill_climber_optimization(run_name, directories)

    # Save the best run's values to results directory
    save_values_json(run_name, best_distance, best_params, directories)

    # Save the best run's parameters in training_runs as best.json
    best_run_data = {
        "best_distance_traveled_m": best_distance,
        "gait_parameters_deg": {
            joint: {
                "a": best_params.a[joint],
                "b": best_params.b[joint],
                "c": best_params.c[joint]
            } for joint in ALL_JOINTS
        },
        "omega": best_params.omega,
        "timestamp": datetime.now().isoformat()
    }
    best_json_path = os.path.join(directories["training_run"], "best.json")
    save_best_run(best_run_data, best_json_path)

    # Visualize the best gait
    visualize_best_gait(best_params)

if __name__ == "__main__":
    main()
