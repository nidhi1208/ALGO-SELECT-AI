import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.features.sorting_features import extract_sorting_features, FEATURE_NAMES
from backend.features.graph_features import extract_graph_features
from backend.features.knapsack_features import extract_knapsack_features

def test_sorting_features():
    # Sorted array
    sorted_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    feats = extract_sorting_features(sorted_arr)
    assert feats["sortedness_ratio"] == 1.0
    assert feats["is_integer"] == 1.0
    assert feats["size"] == 10.0
    assert feats["range_val"] == 9.0

    # Reversed array
    rev_arr = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    rev_feats = extract_sorting_features(rev_arr)
    assert rev_feats["reverse_sortedness_ratio"] == 1.0

    # Small range array
    small_range = [2, 1, 2, 3, 1, 2, 3, 2, 1]
    sr_feats = extract_sorting_features(small_range)
    assert sr_feats["range_to_size_ratio"] < 1.0

def test_graph_features():
    # 4 vertices, 3 edges (spanning tree)
    edges = [(0, 1, 2.5), (1, 2, 1.0), (2, 3, 3.2)]
    g_feats = extract_graph_features(4, edges)
    assert g_feats["num_vertices"] == 4.0
    assert g_feats["num_edges"] == 3.0
    assert g_feats["has_negative_weights"] == 0.0
    assert g_feats["density"] == 3.0 / 6.0

def test_knapsack_features():
    weights = [10, 20, 30]
    values = [60.0, 100.0, 120.0]
    k_feats = extract_knapsack_features(weights, values, 50)
    assert k_feats["num_items"] == 3.0
    assert k_feats["capacity"] == 50.0
    assert abs(k_feats["capacity_to_total_weight_ratio"] - (50.0 / 60.0)) < 1e-3

if __name__ == "__main__":
    test_sorting_features()
    test_graph_features()
    test_knapsack_features()
    print("All feature extraction tests passed successfully!")
