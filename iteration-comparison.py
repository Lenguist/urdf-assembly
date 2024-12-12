
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

def run_hill_climber(max_iterations, save_dir):
    """Modified version of baseline hill climber that returns best distance"""
    
    # Initialize parameters
    best_params = GaitParameters()
    best_distance = evaluate_gait(best_params)
    
    # Run optimization
    for i in range(1, max_iterations + 1):
        # Mutation
        candidate_params = best_params.copy()
        candidate_params.mutate()
        
        # Evaluation
        candidate_distance = evaluate_gait(candidate_params)
        
        # Selection
        if candidate_distance > best_distance:
            best_distance = candidate_distance
            best_params = candidate_params
            
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Best Distance = {best_distance:.4f} m")
    
    # Save results
    results = {
        "iterations": max_iterations,
        "best_distance": best_distance,
        "parameters": {
            "omega": best_params.omega,
            "a": best_params.a,
            "b": best_params.b,
            "c": best_params.c
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"results_{max_iterations}.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return best_distance

def main():
    # Use same directory as before
    base_dir = "iteration_comparison_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Only test 2000 and 4000 iterations
    iterations_to_test = [2000]  # Will do 4000 after if 2000 succeeds
    num_runs = 3  # 3 runs for 2000, 1 run for 4000
    
    # Store results
    results = {n: [] for n in iterations_to_test}
    run_times = {n: [] for n in iterations_to_test}
    
    # Run experiments
    for n_iter in iterations_to_test:
        print(f"\nTesting {n_iter} iterations...")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            start_time = time.time()
            best_distance = run_hill_climber(
                n_iter, 
                os.path.join(base_dir, f"iter_{n_iter}_run_{run}")
            )
            end_time = time.time()
            
            results[n_iter].append(best_distance)
            run_times[n_iter].append(end_time - start_time)
    
    # After 2000 succeeds, try one run of 4000
    print("\nTesting 4000 iterations...")
    start_time = time.time()
    best_distance = run_hill_climber(
        4000,
        os.path.join(base_dir, "iter_4000_run_0")
    )
    end_time = time.time()
    results[4000] = [best_distance]
    run_times[4000] = [end_time - start_time]
    
    # Save these new results without overwriting previous data
    with open(os.path.join(base_dir, "additional_results.json"), "w") as f:
        json.dump({
            "2000": {
                "distances": results[2000],
                "times": run_times[2000]
            },
            "4000": {
                "distances": results[4000],
                "times": run_times[4000]
            }
        }, f, indent=4)

if __name__ == "__main__":
    main()
