import os
import json
import pybullet as p
import pybullet_data
import time
import math
import random
from datetime import datetime

# Joint Indices (as per your URDF)
HIP_JOINTS = [0, 2, 4, 6]
KNEE_JOINTS = [1, 3, 5, 7]
ALL_JOINTS = HIP_JOINTS + KNEE_JOINTS

# Parameter Ranges
A_MIN, A_MAX = 0, 10
B_MIN, B_MAX = 0, 10
C_MIN, C_MAX = 0, 2 * math.pi
OMEGA_MIN, OMEGA_MAX = 0.5, 10

HIP_MAX_FORCE = 100
KNEE_MAX_FORCE = 100
POSITION_GAIN = 1
VELOCITY_GAIN = 1

class GaitParameters:
    def __init__(self, a=None, b=None, c=None, omega=None):
        self.a = a if a is not None else {joint: random.uniform(A_MIN, A_MAX) for joint in ALL_JOINTS}
        self.b = b if b is not None else {joint: random.uniform(B_MIN, B_MAX) for joint in ALL_JOINTS}
        self.c = c if c is not None else {joint: random.uniform(C_MIN, C_MAX) for joint in ALL_JOINTS}
        self.omega = omega if omega is not None else random.uniform(OMEGA_MIN, OMEGA_MAX)
    
    def mutate(self, mutation_rate, mutation_scale):
        for joint in ALL_JOINTS:
            if random.random() < mutation_rate:
                delta_a = self.a[joint] * mutation_scale
                self.a[joint] += random.uniform(-delta_a, delta_a)
                self.a[joint] = max(A_MIN, min(A_MAX, self.a[joint]))
                
            if random.random() < mutation_rate:
                delta_b = self.b[joint] * mutation_scale
                self.b[joint] += random.uniform(-delta_b, delta_b)
                self.b[joint] = max(B_MIN, min(B_MAX, self.b[joint]))
                
            if random.random() < mutation_rate:
                delta_c = mutation_scale * math.pi
                self.c[joint] += random.uniform(-delta_c, delta_c)
                self.c[joint] = (self.c[joint] + 2 * math.pi) % (2 * math.pi)
        
        if random.random() < mutation_rate:
            delta_omega = (OMEGA_MAX - OMEGA_MIN) * mutation_scale
            self.omega += random.uniform(-delta_omega, delta_omega)
            self.omega = max(OMEGA_MIN, min(OMEGA_MAX, self.omega))
    
    def copy(self):
        return GaitParameters(a=self.a.copy(), b=self.b.copy(), c=self.c.copy(), omega=self.omega)

def evaluate_gait(gait_params, run_duration, sim_fps, settling_time):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    
    roll = math.radians(90)
    pitch = 0
    yaw = 0
    start_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    
    start_pos = [0, 0, 0.5]
    robot_id = p.loadURDF("urdf-assembly.urdf", start_pos, start_orientation)
    
    settling_steps = int(sim_fps * settling_time)
    for _ in range(settling_steps):
        p.stepSimulation()
    
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)
    initial_pos, _ = p.getBasePositionAndOrientation(robot_id)
    initial_x = initial_pos[0]
    
    total_steps = int(sim_fps * run_duration)
    for step in range(total_steps):
        t = step / sim_fps
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
    
    final_pos, _ = p.getBasePositionAndOrientation(robot_id)
    final_x = final_pos[0]
    p.disconnect()
    
    distance_traveled = final_x - initial_x
    return distance_traveled
