#!/usr/bin/env python3

import time
import json
import os
import matplotlib.pyplot as plt
from hillclimber import (
    GaitParameters,
    evaluate_gait,
    generate_run_name,
    initialize_directories,
    save_hyperparameters,
    save_values_json,
    save_best_run
)

# Configurable parameters
MAX_ITERATIONS = 2000
NUM_CANDIDATES = 1  # Basic hill climber: 1 candidate per iteration (can adjust if needed)

def benchmarked_baseline_hill_climber():
    run_name = generate_run_name()
    directories = initialize_directories(run_name)

    # Hyperparameters dictionary (same as in hillclimber.py)
    initial_hyperparams = {
        "simulation_fps": 240,
        "run_duration": 10,
        "settling_time": 0.5,
        "max_iterations": MAX_ITERATIONS,
        "mutation_rate": 0.5,
        "mutation_scale": 0.5,
        "omega_min": 0.5,
        "omega_max": 10,
        "a_min": 0,
        "a_max": 10,
        "b_min": 0,
        "b_max": 10,
        "c_min": 0,
        "c_max": 2,
        "hip_max_force": 100,
        "knee_max_force": 100,
        "position_gain": 1,
        "velocity_gain": 1,
        "joint_indices": {
            "hip_joints": [0, 2, 4, 6],
            "knee_joints": [1, 3, 5, 7]
        }
    }

    # Save hyperparameters
    save_hyperparameters(initial_hyperparams, os.path.join(directories["training_run"], "training_hyperparameters.json"))
    save_hyperparameters(initial_hyperparams, os.path.join(directories["results_run"], "training_hyperparameters.json"))

    # Initialize best parameters
    best_params = GaitParameters()
    start_eval = time.perf_counter()
    best_distance = evaluate_gait(best_params)
    end_eval = time.perf_counter()

    # Timing arrays
    mutation_times = []
    evaluation_times = []
    selection_times = []

    # Run the hill climber
    for i in range(1, MAX_ITERATIONS + 1):
        # Mutation timing
        start_mutation = time.perf_counter()
        candidates = []
        for _ in range(NUM_CANDIDATES):
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
        # Find best candidate
        max_distance = max(candidate_distances)
        max_index = candidate_distances.index(max_distance)
        best_candidate_params = candidates[max_index]
        if max_distance > best_distance:
            best_distance = max_distance
            best_params = best_candidate_params
        end_selection = time.perf_counter()
        selection_times.append(end_selection - start_selection)

        # Optionally print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Best Distance = {best_distance:.4f} m")

    # Save best results
    save_values_json(run_name, best_distance, best_params, directories)
    best_run_data = {
        "best_distance_traveled_m": best_distance,
        "gait_parameters_deg": {
            joint: {
                "a": best_params.a[joint],
                "b": best_params.b[joint],
                "c": best_params.c[joint]
            } for joint in best_params.a
        },
        "omega": best_params.omega
    }
    best_json_path = os.path.join(directories["training_run"], "best.json")
    save_best_run(best_run_data, best_json_path)

    # Produce plots
    # Plot timing data: iteration vs time
    iterations = list(range(1, MAX_ITERATIONS + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mutation_times, label='Mutation Time (s)', alpha=0.7)
    plt.plot(iterations, evaluation_times, label='Evaluation Time (s)', alpha=0.7)
    plt.plot(iterations, selection_times, label='Selection Time (s)', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('Benchmarking Hill Climber Steps')
    plt.legend()
    plot_path = os.path.join(directories["results_run"], "timing_plot.png")
    plt.savefig(plot_path)
    print(f"Timing plot saved to {plot_path}")

if __name__ == "__main__":
    benchmarked_baseline_hill_climber()
