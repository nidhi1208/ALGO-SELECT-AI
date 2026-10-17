"""
Graph Algorithms Suite for DAA Project:
Includes Minimum Spanning Tree (Kruskal, Prim-Heap, Prim-Matrix)
and Shortest Path (Dijkstra, Bellman-Ford, Floyd-Warshall).
Instrumented with edge relaxations, cycle checks, and execution timing.
"""

import time
import heapq
from typing import List, Dict, Tuple, Any

class GraphResult:
    def __init__(self, algorithm: str, duration_us: float, cost_or_dist: Any, operations: int, is_correct: bool):
        self.algorithm = algorithm
        self.duration_us = duration_us
        self.cost_or_dist = cost_or_dist
        self.operations = operations
        self.is_correct = is_correct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "duration_us": round(self.duration_us, 2),
            "duration_ms": round(self.duration_us / 1000.0, 4),
            "cost_or_dist": self.cost_or_dist,
            "operations": self.operations,
            "is_correct": self.is_correct
        }


# ==========================================
# Disjoint Set Union (for Kruskal)
# ==========================================
class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


# ==========================================
# 1. Kruskal's MST Algorithm (O(E log E))
# Best for sparse graphs
# ==========================================
def kruskal_mst(num_vertices: int, edges: List[Tuple[int, int, float]]) -> Tuple[float, int]:
    # edges: list of (u, v, weight)
    operations = 0
    sorted_edges = sorted(edges, key=lambda x: x[2])
    operations += len(edges) # proxy for sort comparison work

    dsu = DisjointSet(num_vertices)
    mst_weight = 0.0
    edges_count = 0

    for u, v, w in sorted_edges:
        operations += 1
        if dsu.union(u, v):
            mst_weight += w
            edges_count += 1
            if edges_count == num_vertices - 1:
                break

    return mst_weight, operations


# ==========================================
# 2. Prim's MST Algorithm with Min-Heap (O(E log V))
# Best for intermediate/sparse graphs
# ==========================================
def prim_heap_mst(num_vertices: int, adj_list: Dict[int, List[Tuple[int, float]]]) -> Tuple[float, int]:
    operations = 0
    visited = [False] * num_vertices
    min_heap = [(0.0, 0)]  # (weight, vertex)
    total_weight = 0.0
    nodes_visited = 0

    while min_heap and nodes_visited < num_vertices:
        w, u = heapq.heappop(min_heap)
        operations += 1
        if visited[u]:
            continue

        visited[u] = True
        total_weight += w
        nodes_visited += 1

        for v, weight in adj_list.get(u, []):
            operations += 1
            if not visited[v]:
                heapq.heappush(min_heap, (weight, v))

    return total_weight, operations


# ==========================================
# 3. Prim's MST Algorithm with Adjacency Matrix (O(V^2))
# Best for dense graphs (E ~ V^2)
# ==========================================
def prim_matrix_mst(num_vertices: int, adj_matrix: List[List[float]]) -> Tuple[float, int]:
    operations = 0
    INF = float('inf')
    key = [INF] * num_vertices
    visited = [False] * num_vertices
    key[0] = 0.0
    total_weight = 0.0

    for _ in range(num_vertices):
        min_k = INF
        u = -1
        # Find minimum key node not visited
        for v in range(num_vertices):
            operations += 1
            if not visited[v] and key[v] < min_k:
                min_k = key[v]
                u = v

        if u == -1:
            break

        visited[u] = True
        total_weight += min_k

        for v in range(num_vertices):
            operations += 1
            w = adj_matrix[u][v]
            if w < INF and not visited[v] and w < key[v]:
                key[v] = w

    return total_weight, operations


# ==========================================
# 4. Dijkstra's Shortest Path (O((V + E) log V))
# Non-negative edge weights
# ==========================================
def dijkstra_shortest_path(num_vertices: int, adj_list: Dict[int, List[Tuple[int, float]]], source: int = 0) -> Tuple[List[float], int]:
    operations = 0
    INF = float('inf')
    dist = [INF] * num_vertices
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        operations += 1
        if d > dist[u]:
            continue

        for v, w in adj_list.get(u, []):
            operations += 1
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist, operations


# ==========================================
# 5. Bellman-Ford Shortest Path (O(V * E))
# Handles negative weights, detects negative cycles
# ==========================================
def bellman_ford_shortest_path(num_vertices: int, edges: List[Tuple[int, int, float]], source: int = 0) -> Tuple[List[float], int]:
    operations = 0
    INF = float('inf')
    dist = [INF] * num_vertices
    dist[source] = 0.0

    for _ in range(num_vertices - 1):
        updated = False
        for u, v, w in edges:
            operations += 1
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    # Negative cycle check
    for u, v, w in edges:
        operations += 1
        if dist[u] != INF and dist[u] + w < dist[v]:
            # Negative cycle detected
            return [], operations

    return dist, operations


# ==========================================
# 6. Floyd-Warshall All-Pairs Shortest Path (O(V^3))
# Best for small dense graphs when all pairs needed
# ==========================================
def floyd_warshall_all_pairs(num_vertices: int, adj_matrix: List[List[float]]) -> Tuple[List[List[float]], int]:
    operations = 0
    V = num_vertices
    dist = [row.copy() for row in adj_matrix]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                operations += 1
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist, operations


GRAPH_PORTFOLIO = {
    "MST": {
        "Kruskal": {
            "fn": "kruskal",
            "complexity": "O(E log E)",
            "space": "O(V + E)",
            "optimal_when": "Sparse graphs (E << V^2) and edge list format"
        },
        "Prim_Heap": {
            "fn": "prim_heap",
            "complexity": "O(E log V)",
            "space": "O(V + E)",
            "optimal_when": "Intermediate density with adjacency list representation"
        },
        "Prim_Matrix": {
            "fn": "prim_matrix",
            "complexity": "O(V^2)",
            "space": "O(V^2)",
            "optimal_when": "Dense graphs (E ≈ V^2 / 2) with adjacency matrix"
        }
    },
    "ShortestPath": {
        "Dijkstra": {
            "fn": "dijkstra",
            "complexity": "O((V + E) log V)",
            "space": "O(V + E)",
            "optimal_when": "Non-negative edge weights, sparse/medium graphs"
        },
        "Bellman_Ford": {
            "fn": "bellman_ford",
            "complexity": "O(V * E)",
            "space": "O(V)",
            "optimal_when": "Negative edge weights present or negative cycle detection needed"
        },
        "Floyd_Warshall": {
            "fn": "floyd_warshall",
            "complexity": "O(V^3)",
            "space": "O(V^2)",
            "optimal_when": "Small dense graphs (|V| < 200) needing all-pairs shortest paths"
        }
    }
}
