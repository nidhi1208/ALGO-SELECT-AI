"""
Comprehensive Benchmark Runner & Complexity Telemetry Suite.
Executes candidate algorithms on identical instances, validates sorting correctness,
measures exact wall-clock microseconds, comparisons, swaps, and speedup ratios.
"""

import time
import math
from typing import List, Dict, Any, Tuple
from ..algorithms.sorting import benchmark_single_sort, SORTING_PORTFOLIO
from ..ml.predictor import predictor

def run_sorting_benchmark(arr: List[Any], predicted_algo: str = None) -> Dict[str, Any]:
    """
    Runs all applicable sorting algorithms on the array, records execution times,
    determines the empirical winner, and compares against the AI prediction.
    """
    if predicted_algo is None:
        pred_res = predictor.predict_sorting(arr)
        predicted_algo = pred_res["predicted_algorithm"]

    results = []
    fastest_time = float('inf')
    empirical_winner = None
    default_algo = "QuickSort"
    default_time = None
    predicted_time = None

    for algo_name in SORTING_PORTFOLIO.keys():
        res = benchmark_single_sort(algo_name, arr, repeats=1)
        res_dict = res.to_dict()
        results.append(res_dict)

        if res.is_correct and res.duration_us < fastest_time:
            fastest_time = res.duration_us
            empirical_winner = algo_name

        if algo_name == default_algo:
            default_time = res.duration_us

        if algo_name == predicted_algo:
            predicted_time = res.duration_us

    # Fallbacks if timing fails
    if default_time is None or default_time == 0:
        default_time = 1.0
    if predicted_time is None or predicted_time == 0:
        predicted_time = fastest_time if fastest_time < float('inf') else 1.0

    speedup_vs_default = round(default_time / predicted_time, 2) if predicted_time > 0 else 1.0
    time_saved_us = max(0.0, default_time - predicted_time)
    time_saved_pct = round((time_saved_us / default_time) * 100.0, 1) if default_time > 0 else 0.0

    # Sort results by duration ascending
    results.sort(key=lambda x: x["duration_us"])

    return {
        "array_size": len(arr),
        "empirical_winner": empirical_winner,
        "predicted_algorithm": predicted_algo,
        "ai_prediction_match": empirical_winner == predicted_algo,
        "default_algorithm": default_algo,
        "default_time_us": default_time,
        "predicted_time_us": predicted_time,
        "speedup_vs_default": speedup_vs_default,
        "time_saved_us": round(time_saved_us, 2),
        "time_saved_pct": time_saved_pct,
        "results": results
    }


def generate_complexity_curve(distribution_type: str = "random_uniform_int", max_n: int = 4000) -> Dict[str, Any]:
    """
    Generates data points across multiple array sizes to plot empirical curves
    and visualize crossover points (e.g. where QuickSort overtakes InsertionSort).
    """
    from ..ml.data_generator import generate_sorting_instance
    sizes = [15, 30, 60, 120, 250, 500, 1000, 2000, max_n]
    algorithms = ["InsertionSort", "QuickSort", "MergeSort", "TimSort"]
    
    if distribution_type in ["small_range_int", "random_uniform_int"]:
        algorithms.append("CountingSort")

    curve_data = {algo: [] for algo in algorithms}

    for n in sizes:
        arr = generate_sorting_instance(n, distribution_type)
        for algo in algorithms:
            # Insertion sort gets slow for N > 2000, so skip or cap
            if algo == "InsertionSort" and n > 2000 and distribution_type not in ["nearly_sorted", "completely_sorted"]:
                curve_data[algo].append({"n": n, "time_ms": None})
                continue
            res = benchmark_single_sort(algo, arr, repeats=1)
            curve_data[algo].append({
                "n": n,
                "time_ms": round(res.duration_us / 1000.0, 4),
                "comparisons": res.comparisons
            })

    return {
        "sizes": sizes,
        "distribution_type": distribution_type,
        "curves": curve_data
    }
