import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import time
from hillclimber import (
    GaitParameters,
    evaluate_gait,
    generate_run_name,
    initialize_directories,
    save_hyperparameters,
    save_values_json,
    save_best_run
)

# Simulation Parameters
SIMULATION_FPS = 240          # Simulation steps per second
RUN_DURATION = 10             # Duration of each run in seconds
SETTLING_TIME = 0.5           # Settling time in seconds

# Hill Climber Parameters
MAX_ITERATIONS = 1000         # Maximum number of iterations
MUTATION_RATE = 0.5           # Probability of each parameter being mutated
MUTATION_SCALE = 0.5          # Scale of mutation

# Motor Parameters
HIP_MAX_FORCE = 100    
KNEE_MAX_FORCE = 100   
POSITION_GAIN = 1      
VELOCITY_GAIN = 1      

def run_hill_climber(run_duration, save_dir):
    """Modified hill climber that uses different simulation durations"""
    
    # Modify simulation duration
    global RUN_DURATION, TOTAL_STEPS
    RUN_DURATION = run_duration
    TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
    
    # Initialize parameters
    best_params = GaitParameters()
    best_distance = evaluate_gait(best_params)
    
    start_time = time.time()
    
    for i in range(1, 1000 + 1):  # Fixed 1000 iterations
        candidate_params = best_params.copy()
        candidate_params.mutate()
        
        candidate_distance = evaluate_gait(candidate_params)
        
        if candidate_distance > best_distance:
            best_distance = candidate_distance
            best_params = candidate_params
            
        if i % 100 == 0:
            print(f"Iteration {i}: Best Distance = {best_distance:.4f} m")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    results = {
        "duration": run_duration,
        "best_distance": best_distance,
        "total_time": total_time,
        "parameters": {
            "omega": best_params.omega,
            "a": best_params.a,
            "b": best_params.b,
            "c": best_params.c
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"results_{run_duration}s.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return best_distance, total_time

def main():
    base_dir = "simulation_time_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Save hyperparameters
    hyperparams = {
        "simulation_fps": SIMULATION_FPS,
        "settling_time": SETTLING_TIME,
        "max_iterations": MAX_ITERATIONS,
        "mutation_rate": MUTATION_RATE,
        "mutation_scale": MUTATION_SCALE,
        "hip_max_force": HIP_MAX_FORCE,
        "knee_max_force": KNEE_MAX_FORCE,
        "position_gain": POSITION_GAIN,
        "velocity_gain": VELOCITY_GAIN
    }
    
    with open(os.path.join(base_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    durations = [1.25, 2.5, 5, 10, 20]  # Different simulation durations to test
    num_runs = 1  # Number of runs per duration
    
    results = {d: {"distances": [], "times": []} for d in durations}
    
    for duration in durations:
        print(f"\nTesting {duration}s simulation duration...")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            distance, runtime = run_hill_climber(
                duration,
                os.path.join(base_dir, f"duration_{duration}_run_{run}")
            )
            
            results[duration]["distances"].append(distance)
            results[duration]["times"].append(runtime)
    
    # Create summary tables
    performance_data = {
        "Duration (s)": durations,
        "Mean Distance (m)": [np.mean(results[d]["distances"]) for d in durations],
        "Std Distance (m)": [np.std(results[d]["distances"]) for d in durations],
        "Mean Time (s)": [np.mean(results[d]["times"]) for d in durations],
        "Std Time (s)": [np.std(results[d]["times"]) for d in durations]
    }
    
    df = pd.DataFrame(performance_data)
    df.to_csv(os.path.join(base_dir, "duration_results.csv"), index=False)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Plot performance vs duration
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.errorbar(durations, 
                [np.mean(results[d]["distances"]) for d in durations],
                yerr=[np.std(results[d]["distances"]) for d in durations],
                marker='o', capsize=5)
    plt.xlabel('Simulation Duration (s)')
    plt.ylabel('Distance Achieved (m)')
    plt.title('Performance vs Simulation Duration')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot computation time vs duration
    plt.subplot(1, 2, 2)
    plt.errorbar(durations,
                [np.mean(results[d]["times"]) for d in durations],
                yerr=[np.std(results[d]["times"]) for d in durations],
                marker='o', capsize=5)
    plt.xlabel('Simulation Duration (s)')
    plt.ylabel('Total Computation Time (s)')
    plt.title('Computation Time vs Simulation Duration')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "duration_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()