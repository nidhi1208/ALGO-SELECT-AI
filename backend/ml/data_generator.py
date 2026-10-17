"""
Synthetic Instance Generator and Benchmarking Dataset Generator.
Synthesizes a broad spectrum of problem instances across distinct distributions,
measures empirical wall-clock times across portfolio algorithms, and labels the ground-truth optimal algorithm.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from ..algorithms.sorting import benchmark_single_sort, SORTING_PORTFOLIO
from ..features.sorting_features import extract_sorting_features, FEATURE_NAMES

def generate_sorting_instance(n: int, distribution_type: str) -> List[Any]:
    """Generates an array of size n with a specific distribution pattern."""
    if distribution_type == "random_uniform_int":
        return [random.randint(0, n * 10) for _ in range(n)]

    elif distribution_type == "small_range_int":
        # Crucial for CountingSort dominance: Range K is small (0 to n // 2 or 100)
        max_k = max(5, n // 2)
        return [random.randint(0, max_k) for _ in range(n)]

    elif distribution_type == "nearly_sorted":
        # 95% sorted with small random local perturbations (favors InsertionSort / TimSort)
        arr = list(range(n))
        num_swaps = max(1, int(n * 0.03))
        for _ in range(num_swaps):
            i = random.randint(0, n - 2)
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
        return arr

    elif distribution_type == "reverse_sorted":
        return list(range(n, 0, -1))

    elif distribution_type == "completely_sorted":
        return list(range(n))

    elif distribution_type == "few_unique":
        # Heavy duplicates (e.g., 3-5 unique values)
        unique_vals = [random.randint(1, 20) for _ in range(random.randint(2, 6))]
        return [random.choice(unique_vals) for _ in range(n)]

    elif distribution_type == "gaussian_floats":
        # Float values: invalidates non-comparison integer algorithms
        return [round(random.gauss(100.0, 25.0), 4) for _ in range(n)]

    elif distribution_type == "exponential":
        return [int(random.expovariate(1.0 / (n / 4))) for _ in range(n)]

    elif distribution_type == "large_range_int":
        # Range is massive (e.g. 10^8), CountingSort memory limit is exceeded, Radix or QuickSort wins
        return [random.randint(0, 10_000_000) for _ in range(n)]

    else:
        return [random.randint(0, 1000) for _ in range(n)]


def build_sorting_dataset(samples_per_distribution: int = 50) -> pd.DataFrame:
    """
    Generates instances, extracts features, benchmarks all algorithms,
    and labels the best performing algorithm.
    """
    distributions = [
        "random_uniform_int",
        "small_range_int",
        "nearly_sorted",
        "reverse_sorted",
        "completely_sorted",
        "few_unique",
        "gaussian_floats",
        "exponential",
        "large_range_int"
    ]

    # Mix of small sizes (where InsertionSort wins), medium sizes, and large sizes
    size_ranges = [12, 20, 28, 50, 100, 250, 600, 1200, 2500, 5000]

    records = []
    algorithms = list(SORTING_PORTFOLIO.keys())

    print(f"Generating benchmark dataset with {len(distributions) * len(size_ranges) * samples_per_distribution} instances...")

    for dist in distributions:
        for size in size_ranges:
            for _ in range(max(1, samples_per_distribution // 5)):
                # If small size and nearly sorted or random, insertion sort has high chance
                actual_n = max(5, int(size * random.uniform(0.85, 1.15)))
                arr = generate_sorting_instance(actual_n, dist)

                features = extract_sorting_features(arr)

                # Benchmark candidate algorithms
                best_algo = None
                min_time = float('inf')
                algo_times = {}

                for algo in algorithms:
                    res = benchmark_single_sort(algo, arr, repeats=1)
                    algo_times[f"time_{algo}"] = res.duration_us
                    if res.is_correct and res.duration_us < min_time:
                        min_time = res.duration_us
                        best_algo = algo

                if best_algo is not None and min_time < 999_000_000:
                    row = {**features, **algo_times}
                    row["best_algorithm"] = best_algo
                    row["min_time_us"] = min_time
                    row["distribution_type"] = dist
                    records.append(row)

    df = pd.DataFrame(records)
    print(f"Dataset generated successfully! Total rows: {len(df)}")
    print(df["best_algorithm"].value_counts())
    return df


if __name__ == "__main__":
    df = build_sorting_dataset(samples_per_distribution=30)
    df.to_csv("sorting_training_data.csv", index=False)
