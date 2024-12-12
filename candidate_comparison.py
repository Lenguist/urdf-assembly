```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
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

def run_hill_climber(max_iterations, num_candidates, save_dir):
    """Modified version of baseline hill climber that returns best distance and timing info"""
    
    # Initialize parameters
    best_params = GaitParameters()
    best_distance = evaluate_gait(best_params)
    
    # Timing arrays
    mutation_times = []
    evaluation_times = []
    selection_times = []
    
    # Run optimization
    for i in range(1, max_iterations + 1):
        # Mutation timing
        start_mutation = time.perf_counter()
        candidates = []
        for _ in range(num_candidates):
            candidate_params = best_params.copy()
            candidate_params.mutate()
            candidates.append(candidate_params)
        end_mutation = time.perf_counter()
        mutation_times.append(end_mutation - start_mutation)

        # Evaluation timing
        start_evaluation = time.perf_counter()
        candidate_distances = []
        for c in candidates:
            d = evaluate_gait(c)
            candidate_distances.append(d)
        end_evaluation = time.perf_counter()
        evaluation_times.append(end_evaluation - start_evaluation)

        # Selection timing
        start_selection = time.perf_counter()
        max_distance = max(candidate_distances)
        max_index = candidate_distances.index(max_distance)
        if max_distance > best_distance:
            best_distance = max_distance
            best_params = candidates[max_index]
        end_selection = time.perf_counter()
        selection_times.append(end_selection - start_selection)

        if i % 100 == 0:
            print(f"Iteration {i}: Best Distance = {best_distance:.4f} m")
    
    timing_data = {
        "mutation": np.mean(mutation_times),
        "evaluation": np.mean(evaluation_times),
        "selection": np.mean(selection_times),
        "total": np.mean(mutation_times) + np.mean(evaluation_times) + np.mean(selection_times)
    }
    
    return best_distance, timing_data

def main():
    # Create results directory
    base_dir = "candidate_comparison_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Configure experiments
    configs = [
        (1000, 1),  # iterations, candidates
        (500, 2),
        (250, 4),
        (125, 8)
    ]
    num_runs = 3
    
    # Store results
    results = {f"{iter}x{cand}": {"distances": [], "timing": []} for iter, cand in configs}
    
    # Run experiments
    for iterations, candidates in configs:
        print(f"\nTesting {iterations} iterations with {candidates} candidates...")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            best_distance, timing_data = run_hill_climber(
                iterations, 
                candidates,
                os.path.join(base_dir, f"iter_{iterations}_cand_{candidates}_run_{run}")
            )
            
            config_key = f"{iterations}x{candidates}"
            results[config_key]["distances"].append(best_distance)
            results[config_key]["timing"].append(timing_data)
    
    # Create performance table
    performance_data = {
        "Configuration": [],
        "Mean Distance (m)": [],
        "Std Distance (m)": [],
    }
    
    # Create timing table
    timing_data = {
        "Configuration": [],
        "Mutation (s)": [],
        "Evaluation (s)": [],
        "Selection (s)": [],
        "Total (s)": []
    }
    
    for config in results:
        # Performance data
        distances = results[config]["distances"]
        performance_data["Configuration"].append(config)
        performance_data["Mean Distance (m)"].append(np.mean(distances))
        performance_data["Std Distance (m)"].append(np.std(distances))
        
        # Timing data
        timing = results[config]["timing"]
        timing_data["Configuration"].append(config)
        timing_data["Mutation (s)"].append(np.mean([t["mutation"] for t in timing]))
        timing_data["Evaluation (s)"].append(np.mean([t["evaluation"] for t in timing]))
        timing_data["Selection (s)"].append(np.mean([t["selection"] for t in timing]))
        timing_data["Total (s)"].append(np.mean([t["total"] for t in timing]))
    
    # Save tables
    pd.DataFrame(performance_data).to_csv(os.path.join(base_dir, "performance_summary.csv"), index=False)
    pd.DataFrame(timing_data).to_csv(os.path.join(base_dir, "timing_summary.csv"), index=False)
    
    print("\nPerformance Summary:")
    print(pd.DataFrame(performance_data).to_string(index=False))
    print("\nTiming Summary:")
    print(pd.DataFrame(timing_data).to_string(index=False))
    
    # Create plots
    plt.figure(figsize=(10, 6))
    configs = [k for k in results.keys()]
    means = [np.mean(results[k]["distances"]) for k in configs]
    stds = [np.std(results[k]["distances"]) for k in configs]
    
    plt.bar(configs, means, yerr=stds, capsize=5)
    plt.xlabel('Configuration (iterations x candidates)')
    plt.ylabel('Distance Achieved (m)')
    plt.title('Performance vs Candidate Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "candidate_comparison.png"))
    plt.close()

if __name__ == "__main__":
    main()
```