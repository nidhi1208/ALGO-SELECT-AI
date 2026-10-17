import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import pytest
except ImportError:
    pytest = None
import random
from backend.algorithms.sorting import (
    quick_sort, merge_sort, heap_sort, insertion_sort, tim_sort, counting_sort, radix_sort
)
from backend.algorithms.graphs import (
    kruskal_mst, prim_heap_mst, prim_matrix_mst, dijkstra_shortest_path, bellman_ford_shortest_path
)
from backend.algorithms.knapsack import (
    knapsack_dp, knapsack_branch_and_bound, knapsack_meet_in_the_middle, knapsack_greedy
)

def test_sorting_correctness():
    test_cases = [
        [5, 2, 9, 1, 5, 6],
        list(range(20, 0, -1)),
        [7] * 10,
        [42],
        [],
        [100, 3, 25, 1, 89, 44, 12, 78, 9, 3]
    ]

    for arr in test_cases:
        expected = sorted(arr)
        
        # Comparison sorts
        assert quick_sort(arr)[0] == expected
        assert merge_sort(arr)[0] == expected
        assert heap_sort(arr)[0] == expected
        assert insertion_sort(arr)[0] == expected
        assert tim_sort(arr)[0] == expected

        # Non-comparison sorts (positive integers)
        if arr and all(isinstance(x, int) and x >= 0 for x in arr):
            assert counting_sort(arr)[0] == expected
            assert radix_sort(arr)[0] == expected

def test_graph_mst_correctness():
    # 4 vertices: 0-1 (1), 1-2 (2), 2-3 (3), 0-3 (4), 1-3 (5)
    # Expected MST edges: 0-1 (1), 1-2 (2), 2-3 (3) => Total weight = 6.0
    V = 4
    edges = [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 3.0),
        (0, 3, 4.0),
        (1, 3, 5.0)
    ]
    
    adj_list = {
        0: [(1, 1.0), (3, 4.0)],
        1: [(0, 1.0), (2, 2.0), (3, 5.0)],
        2: [(1, 2.0), (3, 3.0)],
        3: [(0, 4.0), (1, 5.0), (2, 3.0)]
    }

    INF = float('inf')
    adj_matrix = [
        [0.0, 1.0, INF, 4.0],
        [1.0, 0.0, 2.0, 5.0],
        [INF, 2.0, 0.0, 3.0],
        [4.0, 5.0, 3.0, 0.0]
    ]

    kruskal_w, _ = kruskal_mst(V, edges)
    prim_heap_w, _ = prim_heap_mst(V, adj_list)
    prim_mat_w, _ = prim_matrix_mst(V, adj_matrix)

    assert kruskal_w == 6.0
    assert prim_heap_w == 6.0
    assert prim_mat_w == 6.0

def test_shortest_path_dijkstra():
    V = 4
    adj_list = {
        0: [(1, 4.0), (2, 1.0)],
        1: [(3, 1.0)],
        2: [(1, 2.0), (3, 5.0)],
        3: []
    }
    # Shortest path from 0 to 3: 0 -> 2 -> 1 -> 3 = 1 + 2 + 1 = 4.0
    dist, _ = dijkstra_shortest_path(V, adj_list, source=0)
    assert dist[3] == 4.0

def test_knapsack_solvers():
    weights = [2, 3, 4, 5]
    values = [3.0, 4.0, 5.0, 6.0]
    capacity = 5
    # Optimal: weight 2 + 3 = 5, value = 3 + 4 = 7.0
    val_dp, _ = knapsack_dp(weights, values, capacity)
    val_bb, _ = knapsack_branch_and_bound(weights, values, capacity)
    val_mitm, _ = knapsack_meet_in_the_middle(weights, values, capacity)

    assert val_dp == 7.0
    assert val_bb == 7.0
    assert val_mitm == 7.0

if __name__ == "__main__":
    test_sorting_correctness()
    test_graph_mst_correctness()
    test_shortest_path_dijkstra()
    test_knapsack_solvers()
    print("All unit tests passed successfully!")
