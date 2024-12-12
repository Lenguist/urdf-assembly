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

def run_hill_climber(fps, save_dir):
    """Modified hill climber that uses different FPS settings"""
    
    # Modify FPS in hillclimber.py parameters
    global SIMULATION_FPS
    SIMULATION_FPS = fps
    global TOTAL_STEPS
    TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
    
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
        "fps": fps,
        "best_distance": best_distance,
        "total_time": total_time
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"results_{fps}fps.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return best_distance, total_time

def main():
    base_dir = "fps_comparison_results"
    os.makedirs(base_dir, exist_ok=True)
    
    fps_values = [60, 120, 240, 480, 960]
    num_runs = 3
    
    results = {fps: {"distances": [], "times": []} for fps in fps_values}
    
    for fps in fps_values:
        print(f"\nTesting {fps} FPS...")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            distance, runtime = run_hill_climber(
                fps,
                os.path.join(base_dir, f"fps_{fps}_run_{run}")
            )
            
            results[fps]["distances"].append(distance)
            results[fps]["times"].append(runtime)
    
    # Create summary tables
    performance_data = {
        "FPS": fps_values,
        "Mean Distance (m)": [np.mean(results[fps]["distances"]) for fps in fps_values],
        "Std Distance (m)": [np.std(results[fps]["distances"]) for fps in fps_values],
        "Mean Time (s)": [np.mean(results[fps]["times"]) for fps in fps_values],
        "Std Time (s)": [np.std(results[fps]["times"]) for fps in fps_values]
    }
    
    df = pd.DataFrame(performance_data)
    df.to_csv(os.path.join(base_dir, "fps_results.csv"), index=False)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Plot performance vs FPS
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.errorbar(fps_values, 
                [np.mean(results[fps]["distances"]) for fps in fps_values],
                yerr=[np.std(results[fps]["distances"]) for fps in fps_values],
                marker='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Simulation FPS')
    plt.ylabel('Distance Achieved (m)')
    plt.title('Performance vs FPS')
    
    # Plot computation time vs FPS
    plt.subplot(1, 2, 2)
    plt.errorbar(fps_values,
                [np.mean(results[fps]["times"]) for fps in fps_values],
                yerr=[np.std(results[fps]["times"]) for fps in fps_values],
                marker='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Simulation FPS')
    plt.ylabel('Total Computation Time (s)')
    plt.title('Computation Time vs FPS')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fps_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()