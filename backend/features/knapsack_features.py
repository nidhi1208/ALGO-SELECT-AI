"""
Knapsack Meta-Feature Extractor for DAA Algorithm Selection.
Extracts capacity ratio, correlation, and distribution features in O(N).
"""

from typing import List, Dict

KNAPSACK_FEATURE_NAMES = [
    "num_items",
    "capacity",
    "capacity_to_total_weight_ratio",
    "avg_weight",
    "avg_value",
    "weight_value_correlation",
    "capacity_to_n_ratio"
]

def extract_knapsack_features(weights: List[int], values: List[float], capacity: int) -> Dict[str, float]:
    n = max(1, len(weights))
    total_w = sum(weights) if weights else 1
    total_v = sum(values) if values else 0.0

    avg_w = total_w / n
    avg_v = total_v / n
    cap_ratio = capacity / total_w if total_w > 0 else 0.0
    cap_n_ratio = capacity / n

    # Correlation between weights and values
    correlation = 0.0
    if n > 1 and weights and values:
        var_w = sum((w - avg_w) ** 2 for w in weights)
        var_v = sum((v - avg_v) ** 2 for v in values)
        cov = sum((w - avg_w) * (v - avg_v) for w, v in zip(weights, values))
        denom = (var_w * var_v) ** 0.5
        if denom > 1e-9:
            correlation = cov / denom

    return {
        "num_items": float(n),
        "capacity": float(capacity),
        "capacity_to_total_weight_ratio": round(cap_ratio, 4),
        "avg_weight": round(avg_w, 2),
        "avg_value": round(avg_v, 2),
        "weight_value_correlation": round(correlation, 4),
        "capacity_to_n_ratio": round(cap_n_ratio, 2)
    }
