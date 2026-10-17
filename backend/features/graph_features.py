"""
Graph Meta-Feature Extractor for DAA Algorithm Selection.
Extracts structural graph features in O(V + E) time.
"""

from typing import List, Dict, Tuple, Any

GRAPH_FEATURE_NAMES = [
    "num_vertices",
    "num_edges",
    "density",
    "avg_degree",
    "max_degree",
    "has_negative_weights",
    "weight_variance",
    "sparsity_ratio"
]

def extract_graph_features(num_vertices: int, edges: List[Tuple[int, int, float]], is_directed: bool = False) -> Dict[str, float]:
    V = max(1, num_vertices)
    E = len(edges)

    max_possible_edges = V * (V - 1) if is_directed else (V * (V - 1)) / 2.0
    density = E / max_possible_edges if max_possible_edges > 0 else 0.0

    degrees = [0] * V
    weights = []
    has_negative = 0.0

    for u, v, w in edges:
        if 0 <= u < V:
            degrees[u] += 1
        if 0 <= v < V and not is_directed:
            degrees[v] += 1
        weights.append(w)
        if w < 0:
            has_negative = 1.0

    avg_degree = sum(degrees) / V
    max_degree = max(degrees) if degrees else 0

    if weights:
        mean_w = sum(weights) / len(weights)
        weight_var = sum((w - mean_w) ** 2 for w in weights) / len(weights)
    else:
        weight_var = 0.0

    return {
        "num_vertices": float(V),
        "num_edges": float(E),
        "density": round(density, 4),
        "avg_degree": round(avg_degree, 2),
        "max_degree": float(max_degree),
        "has_negative_weights": has_negative,
        "weight_variance": round(weight_var, 4),
        "sparsity_ratio": round(1.0 - min(1.0, density), 4)
    }
