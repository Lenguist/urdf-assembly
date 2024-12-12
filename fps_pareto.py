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

def run_hill_climber(iterations, fps, save_dir):
    """Hill climber with configurable iterations and FPS"""
    
    # Modify simulation parameters
    global SIMULATION_FPS
    SIMULATION_FPS = fps
    global TOTAL_STEPS
    TOTAL_STEPS = SIMULATION_FPS * RUN_DURATION
    
    best_params = GaitParameters()
    best_distance = evaluate_gait(best_params)
    
    start_time = time.time()
    
    for i in range(1, iterations + 1):
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
        "iterations": iterations,
        "fps": fps,
        "best_distance": best_distance,
        "total_time": total_time
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"results_iter{iterations}_fps{fps}.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return best_distance, total_time

def main():
    base_dir = "fps_pareto_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Configurations with approximately equal total computation time
    configs = [
        (2000, 120),   # 2000 iterations × 120 FPS
        (1000, 240),   # 1000 iterations × 240 FPS
        (500, 480),    # 500 iterations × 480 FPS
        (4000, 60),    # 4000 iterations × 60 FPS
        (250, 960),    # 250 iterations × 960 FPS
    ]
    
    results = {f"{iter}x{fps}": {"distances": [], "times": []} 
              for iter, fps in configs}
    
    # Run each configuration once
    for iterations, fps in configs:
        print(f"\nTesting {iterations} iterations with {fps} FPS...")
        
        distance, runtime = run_hill_climber(
            iterations,
            fps,
            os.path.join(base_dir, f"iter_{iterations}_fps_{fps}")
        )
        
        key = f"{iterations}x{fps}"
        results[key]["distances"].append(distance)
        results[key]["times"].append(runtime)
    
    # Create results table
    performance_data = {
        "Configuration": [],
        "Total Compute Time (s)": [],
        "Distance (m)": []
    }
    
    for key in results:
        performance_data["Configuration"].append(key)
        performance_data["Total Compute Time (s)"].append(np.mean(results[key]["times"]))
        performance_data["Distance (m)"].append(np.mean(results[key]["distances"]))
    
    df = pd.DataFrame(performance_data)
    df.to_csv(os.path.join(base_dir, "pareto_results.csv"), index=False)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Plot Pareto front
    plt.figure(figsize=(10, 6))
    plt.scatter([np.mean(results[k]["times"]) for k in results],
               [np.mean(results[k]["distances"]) for k in results])
    
    # Add labels for each point
    for i, key in enumerate(results):
        plt.annotate(key, 
                    (np.mean(results[key]["times"]), 
                     np.mean(results[key]["distances"])))
    
    plt.xlabel('Total Computation Time (s)')
    plt.ylabel('Distance Achieved (m)')
    plt.title('Pareto Front: Performance vs Computation Time (FPS)')
    
    plt.savefig(os.path.join(base_dir, "pareto_front.png"))
    plt.close()

if __name__ == "__main__":
    main()