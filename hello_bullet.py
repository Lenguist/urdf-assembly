import pybullet as p
import time
import pybullet_data
import math

# Connect to PyBullet GUI
p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
p.setGravity(0,0,-10)

# Load a plane
planeId = p.loadURDF("plane.urdf")

# Start the robot above the ground
cubeStartPos = [0,0,0.5]  
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robotID = p.loadURDF("urdf-assembly.urdf", cubeStartPos, cubeStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE)

p.setRealTimeSimulation(0)

# Fix the robot's base in the air
p.createConstraint(
    parentBodyUniqueId=robotID,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0,0,0],
    parentFramePosition=[0,0,0],
    childFramePosition=[0,0,0.5]
)

num_joints = p.getNumJoints(robotID)
for i in range(num_joints):
    info = p.getJointInfo(robotID, i)
    print("Joint index:", i, "Name:", info[1].decode("utf-8"))

joint_leg = 0   # right_back_leg
joint_knee = 1  # right_back_knee

# Reset joints
p.resetJointState(robotID, joint_leg, 0)
p.resetJointState(robotID, joint_knee, 0)

# Set camera to look from above (top-down view)
p.resetDebugVisualizerCamera(
    cameraDistance=1,   # a bit further to get a good view
    cameraYaw=0,          # doesn't matter too much from top view, can experiment
    cameraPitch=-60,      # straight down
    cameraTargetPosition=[0,0,0.05]
)

def frange(start, stop, step):
    if step > 0:
        val = start
        while val <= stop:
            yield val
            val += step
    else:
        val = start
        while val >= stop:
            yield val
            val += step

def move_joint_in_increments(robot_id, joint_index, start_deg, end_deg, step_deg, pause_sec=0.1):
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)
    step_rad = math.radians(step_deg)

    if start_rad < end_rad:
        angle_range = [a for a in frange(start_rad, end_rad, step_rad)]
    else:
        angle_range = [a for a in frange(start_rad, end_rad, -step_rad)]

    for angle in angle_range:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle,
            force=1000,
            positionGain=0.1,
            velocityGain=0.1
        )
        
        steps = int(240 * pause_sec)
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1./240.)

        js = p.getJointState(robot_id, joint_index)
        print(f"Joint {joint_index} angle (rad): {js[0]:.4f}")

# Move the lower leg joint
move_joint_in_increments(robotID, joint_leg, 0, 45, 15, pause_sec=0.2)
move_joint_in_increments(robotID, joint_leg, 45, 0, 15, pause_sec=0.2)

# Move the knee joint
move_joint_in_increments(robotID, joint_knee, 0, 45, 15, pause_sec=0.2)
move_joint_in_increments(robotID, joint_knee, 45, 0, 15, pause_sec=0.2)

cubePos, cubeOrn = p.getBasePositionAndOrientation(robotID)
print("Final base position:", cubePos, "orientation:", cubeOrn)

p.disconnect()
