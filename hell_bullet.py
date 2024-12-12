import pybullet as p
import pybullet_data
import time
import math

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a plane
plane_id = p.loadURDF("plane.urdf")

# Start position and orientation
start_pos = [0, 0, 0.5]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Load the robot
robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)

# Suspend the robot in the air
p.resetBasePositionAndOrientation(robot_id, [0, 0, 1], start_orientation)

# Let the robot settle for a bit
for i in range(240):
    p.stepSimulation()
    time.sleep(1./240.)

# Adjust the camera for better viewing
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=0,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 1]
)

# Helper function to set joint angles
def set_joint_angle(robot_id, joint_index, angle_deg):
    angle_rad = math.radians(angle_deg)
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle_rad,
        force=1000,
        positionGain=0.1,
        velocityGain=0.1
    )

# Joint indices for testing
joint_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Update if your robot has different joint indices

# Spin each joint one by one
for joint in joint_indices:
    # Forward 15 degrees
    set_joint_angle(robot_id, joint, 15)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)

    # Pause for 1 second
    time.sleep(0.2)

    # Reset to 0 degrees
    set_joint_angle(robot_id, joint, 0)
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1./240.)

# Disconnect from PyBullet
p.disconnect()
