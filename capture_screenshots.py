import pybullet as p
import pybullet_data
import time
import math
import os
import numpy as np
from PIL import Image
import json
from hillclimber import (
    GaitParameters,
    evaluate_gait,
    generate_run_name,
    initialize_directories,
    save_hyperparameters,
    save_values_json,
    save_best_run
)

def load_best_run(run_name, results_dir="results/hillclimber"):
    """Load the best run's data"""
    values_json_path = os.path.join(results_dir, run_name, "values.json")
    if not os.path.exists(values_json_path):
        print(f"Values JSON file '{values_json_path}' does not exist.")
        return None
    
    with open(values_json_path, 'r') as f:
        best_run = json.load(f)
    return best_run

def capture_screenshot(width, height, filename):
    """Capture and save a screenshot using PIL"""
    pixels = p.getCameraImage(
        width=width,
        height=height,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )[2]
    
    img_array = np.reshape(pixels, (height, width, 4))
    img_array = img_array[:, :, :3]
    
    im = Image.fromarray(img_array, 'RGB')
    im.save(filename)
    print(f"Saved {filename}")

def main():
    # Load best run data
    best_run = load_best_run("first_run")
    if best_run is None:
        print("Failed to load best run data. Exiting.")
        return
    
    gait_params = best_run['gait_parameters_deg']
    omega = best_run['omega']
    
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane and robot
    plane_id = p.loadURDF("plane.urdf")
    start_pos = [0, 0, 0.5]
    start_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Create directory for screenshots
    screenshots_dir = "figures/robot_screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Define joint indices
    HIP_JOINTS = [0, 2, 4, 6]
    KNEE_JOINTS = [1, 3, 5, 7]
    ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS
    
    # Let the robot settle
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Capture 10 screenshots over 10 seconds
    total_steps = 240 * 10  # 10 seconds at 240 FPS
    screenshot_interval = total_steps // 10
    
    for step in range(total_steps):
        t = step * (1.0/240.0)  # Current time in seconds
        
        # Apply joint movements
        for joint in ALL_JOINTS:
            a = gait_params[str(joint)]['a']
            b = gait_params[str(joint)]['b']
            c = gait_params[str(joint)]['c']
            
            theta = a + b * math.sin(omega * t + c)
            theta_rad = math.radians(theta)
            
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=int(joint),
                controlMode=p.POSITION_CONTROL,
                targetPosition=theta_rad,
                force=250,
                positionGain=0.1,
                velocityGain=0.1
            )
        
        p.stepSimulation()
        
        if step % screenshot_interval == 0:
            screenshot_num = step // screenshot_interval
            filename = os.path.join(screenshots_dir, f'screenshot_{screenshot_num:02d}.png')
            capture_screenshot(1024, 768, filename)
        
        time.sleep(1./240.)
    
    p.disconnect()
    print("Screenshot capture complete!")

if __name__ == "__main__":
    main()