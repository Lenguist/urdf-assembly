import json
import os
import numpy as np
import matplotlib.pyplot as plt

def collect_results(base_dir="figures/iteration_comparison_results"):
    iterations = [250, 500, 1000, 2000, 4000]
    results = {i: [] for i in iterations}
    
    # Collect results from each run
    for iter_num in iterations:
        # Skip 4000 if it only has one run
        max_runs = 1 if iter_num == 4000 else 3
        
        for run in range(max_runs):
            result_file = os.path.join(base_dir, f"iter_{iter_num}_run_{run}", f"results_{iter_num}.json")
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results[iter_num].append(data['best_distance'])
            except:
                print(f"Could not load {result_file}")
                continue
    
    # Calculate means and standard deviations
    means = []
    stds = []
    for iter_num in iterations:
        if len(results[iter_num]) > 0:  # Only include if we have results
            means.append(np.mean(results[iter_num]))
            stds.append(np.std(results[iter_num]) if len(results[iter_num]) > 1 else 0)
    
    return iterations, means, stds

def plot_iteration_comparison():
    iterations, means, stds = collect_results()
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations, means, yerr=stds, marker='o', capsize=5, 
                markersize=8, linewidth=2, elinewidth=2, capthick=2)
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Distance Achieved (m)', fontsize=12)
    plt.title('Performance vs Number of Iterations', fontsize=14)
    
    # Add value labels
    for x, y in zip(iterations, means):
        plt.annotate(f'{y:.2f}m', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('figures/iteration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print numerical results
    print("\nNumerical Results:")
    print("Iterations | Mean Distance (m) | Std Dev")
    print("-" * 45)
    for i, m, s in zip(iterations, means, stds):
        print(f"{i:9d} | {m:14.3f} | {s:8.3f}")

if __name__ == "__main__":
    plot_iteration_comparison()