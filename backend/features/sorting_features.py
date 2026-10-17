"""
Lightweight O(N) Meta-Feature Extractor for Sorting Instances.
Strict requirement: Must run significantly faster than the algorithms themselves (< 1ms).
Extracts statistical and structural characteristics to feed Machine Learning models.
"""

import math
import numpy as np
from typing import List, Dict, Any

FEATURE_NAMES = [
    "size",
    "log_size",
    "is_integer",
    "range_val",
    "range_to_size_ratio",
    "sortedness_ratio",
    "reverse_sortedness_ratio",
    "runs_ratio",
    "duplicate_ratio",
    "std_dev",
    "entropy_proxy"
]

def extract_sorting_features(arr: List[Any]) -> Dict[str, float]:
    """
    Extracts meta-features from an input array in strict linear O(N) time.
    """
    n = len(arr)
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    # Sampling for very large arrays to guarantee sub-millisecond feature extraction
    sample_size = min(n, 5000)
    if n > 5000:
        step = n // sample_size
        sample = [arr[i] for i in range(0, n, step)][:sample_size]
    else:
        sample = arr

    is_int_flag = 1.0 if all(isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) for x in sample[:100]) else 0.0

    min_v = float(sample[0])
    max_v = float(sample[0])
    consecutive_sorted = 0
    consecutive_reversed = 0
    runs = 1
    current_direction = 0 # 1 for up, -1 for down

    for i in range(1, len(sample)):
        v_prev = float(sample[i - 1])
        v_curr = float(sample[i])

        if v_curr < min_v:
            min_v = v_curr
        if v_curr > max_v:
            max_v = v_curr

        if v_prev <= v_curr:
            consecutive_sorted += 1
        if v_prev >= v_curr:
            consecutive_reversed += 1

        # Detect direction changes for run counting
        if v_curr > v_prev:
            if current_direction == -1:
                runs += 1
            current_direction = 1
        elif v_curr < v_prev:
            if current_direction == 1:
                runs += 1
            current_direction = -1

    sample_len = len(sample)
    sortedness = consecutive_sorted / max(1, sample_len - 1)
    reverse_sortedness = consecutive_reversed / max(1, sample_len - 1)
    runs_ratio = runs / sample_len

    # Cardinality / Duplicate estimation
    try:
        unique_cnt = len(set(sample))
        duplicate_ratio = unique_cnt / sample_len
    except Exception:
        duplicate_ratio = 1.0

    range_val = max(0.0, max_v - min_v)
    range_to_size = range_val / max(1, n)

    # Standard deviation and entropy proxy
    try:
        vals = [float(x) for x in sample]
        mean_v = sum(vals) / sample_len
        variance = sum((x - mean_v) ** 2 for x in vals) / sample_len
        std_dev = math.sqrt(variance)
    except Exception:
        std_dev = 0.0

    # Approximate entropy via 10-bin histogram
    try:
        if range_val > 0:
            bins = [0] * 10
            for x in sample:
                b_idx = min(9, int(((float(x) - min_v) / range_val) * 10))
                bins[b_idx] += 1
            entropy = 0.0
            for b in bins:
                if b > 0:
                    p = b / sample_len
                    entropy -= p * math.log2(p)
        else:
            entropy = 0.0
    except Exception:
        entropy = 0.0

    features = {
        "size": float(n),
        "log_size": math.log2(n) if n > 0 else 0.0,
        "is_integer": is_int_flag,
        "range_val": range_val,
        "range_to_size_ratio": range_to_size,
        "sortedness_ratio": round(sortedness, 4),
        "reverse_sortedness_ratio": round(reverse_sortedness, 4),
        "runs_ratio": round(runs_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "std_dev": round(std_dev, 4),
        "entropy_proxy": round(entropy, 4)
    }

    return features
