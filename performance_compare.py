import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json  # Added this import
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
    # Create results directory
    base_dir = "iteration_comparison_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Configure experiments
    iterations_to_test = [250, 500, 1000, 2000, 4000]
    num_runs = 3
    
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
    
    # Create results table
    table_data = {
        "Iterations": [],
        "Mean Distance (m)": [],
        "Std Distance (m)": [],
        "Mean Time (s)": [],
        "Std Time (s)": []
    }
    
    for n_iter in iterations_to_test:
        table_data["Iterations"].append(n_iter)
        table_data["Mean Distance (m)"].append(np.mean(results[n_iter]))
        table_data["Std Distance (m)"].append(np.std(results[n_iter]))
        table_data["Mean Time (s)"].append(np.mean(run_times[n_iter]))
        table_data["Std Time (s)"].append(np.std(run_times[n_iter]))
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(base_dir, "results_summary.csv"), index=False)
    print("\nResults Summary:")
    print(df.to_string(index=False))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    means = [np.mean(results[n]) for n in iterations_to_test]
    stds = [np.std(results[n]) for n in iterations_to_test]
    
    plt.errorbar(iterations_to_test, means, yerr=stds, marker='o', capsize=5)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Distance Achieved (m)')
    plt.title('Performance vs Number of Iterations')
    
    # Save plot
    plt.savefig(os.path.join(base_dir, "performance_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()