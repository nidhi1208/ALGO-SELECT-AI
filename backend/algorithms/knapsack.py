"""
0/1 Knapsack Algorithms Suite for DAA Project:
Includes Dynamic Programming, Branch and Bound, Meet-in-the-Middle, and Greedy Dantzig.
"""

import time
import bisect
from typing import List, Tuple, Dict, Any

# ==========================================
# 1. Dynamic Programming (O(N * W))
# Best when capacity W is small to moderate
# ==========================================
def knapsack_dp(weights: List[int], values: List[float], capacity: int) -> Tuple[float, int]:
    operations = 0
    n = len(weights)
    if capacity <= 0 or n == 0:
        return 0.0, 0

    dp = [0.0] * (capacity + 1)

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(capacity, w - 1, -1):
            operations += 1
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v

    return dp[capacity], operations


# ==========================================
# 2. Branch and Bound (Best-First Search with Fractional Upper Bound)
# Best when N is moderate, W is large, and pruning cuts deep
# ==========================================
def knapsack_branch_and_bound(weights: List[int], values: List[float], capacity: int) -> Tuple[float, int]:
    operations = 0
    n = len(weights)
    if capacity <= 0 or n == 0:
        return 0.0, 0

    # Sort items by value/weight ratio descending
    items = sorted(zip(weights, values), key=lambda x: x[1] / x[0] if x[0] > 0 else 0, reverse=True)
    w_sorted = [x[0] for x in items]
    v_sorted = [x[1] for x in items]

    def bound(level: int, current_w: int, current_v: float) -> float:
        nonlocal operations
        if current_w > capacity:
            return 0.0
        b = current_v
        w_acc = current_w
        for i in range(level, n):
            operations += 1
            if w_acc + w_sorted[i] <= capacity:
                w_acc += w_sorted[i]
                b += v_sorted[i]
            else:
                b += (capacity - w_acc) * (v_sorted[i] / w_sorted[i])
                break
        return b

    max_profit = 0.0
    
    # Priority Queue of states: (-bound, level, current_w, current_v)
    import heapq
    initial_bound = bound(0, 0, 0.0)
    pq = [(-initial_bound, 0, 0, 0.0)]

    steps = 0
    MAX_STEPS = 200_000 # guard against exponential explosion
    while pq and steps < MAX_STEPS:
        steps += 1
        neg_b, lvl, curr_w, curr_v = heapq.heappop(pq)
        cur_bound = -neg_b

        if cur_bound <= max_profit:
            continue

        if lvl == n:
            if curr_v > max_profit:
                max_profit = curr_v
            continue

        # Option 1: Take item[lvl]
        new_w = curr_w + w_sorted[lvl]
        new_v = curr_v + v_sorted[lvl]
        if new_w <= capacity:
            if new_v > max_profit:
                max_profit = new_v
            b_take = bound(lvl + 1, new_w, new_v)
            if b_take > max_profit:
                heapq.heappush(pq, (-b_take, lvl + 1, new_w, new_v))

        # Option 2: Do not take item[lvl]
        b_skip = bound(lvl + 1, curr_w, curr_v)
        if b_skip > max_profit:
            heapq.heappush(pq, (-b_skip, lvl + 1, curr_w, curr_v))

    return max_profit, operations + steps


# ==========================================
# 3. Meet-in-the-Middle (O(2^(N/2)))
# Best when N <= 36 and W is astronomical
# ==========================================
def knapsack_meet_in_the_middle(weights: List[int], values: List[float], capacity: int) -> Tuple[float, int]:
    operations = 0
    n = len(weights)
    if n > 32:
        # Avoid huge exponential explosion if N > 32
        return knapsack_dp(weights, values, min(capacity, 10000))

    mid = n // 2
    w1, v1 = weights[:mid], values[:mid]
    w2, v2 = weights[mid:], values[mid:]

    def generate_subsets(w_sub: List[int], v_sub: List[float]) -> List[Tuple[int, float]]:
        nonlocal operations
        subsets = [(0, 0.0)]
        for w, v in zip(w_sub, v_sub):
            new_sub = []
            for sw, sv in subsets:
                operations += 1
                if sw + w <= capacity:
                    new_sub.append((sw + w, sv + v))
            subsets.extend(new_sub)
        return subsets

    sub1 = generate_subsets(w1, v1)
    sub2 = generate_subsets(w2, v2)

    # Sort sub2 by weight
    sub2.sort(key=lambda x: x[0])
    
    # Filter sub2 so that increasing weight has strictly increasing value
    clean_sub2 = []
    max_v = -1.0
    for w, v in sub2:
        operations += 1
        if v > max_v:
            clean_sub2.append((w, v))
            max_v = v

    weights2 = [x[0] for x in clean_sub2]
    best_profit = 0.0

    for w, v in sub1:
        operations += 1
        rem = capacity - w
        idx = bisect.bisect_right(weights2, rem) - 1
        if idx >= 0:
            total_v = v + clean_sub2[idx][1]
            if total_v > best_profit:
                best_profit = total_v

    return best_profit, operations


# ==========================================
# 4. Greedy Dantzig Approximation (O(N log N))
# Fast heuristic approximation
# ==========================================
def knapsack_greedy(weights: List[int], values: List[float], capacity: int) -> Tuple[float, int]:
    operations = 0
    n = len(weights)
    items = sorted(range(n), key=lambda i: values[i] / weights[i] if weights[i] > 0 else 0, reverse=True)
    operations += n

    curr_w = 0
    curr_v = 0.0
    for i in items:
        operations += 1
        if curr_w + weights[i] <= capacity:
            curr_w += weights[i]
            curr_v += values[i]

    return curr_v, operations


KNAPSACK_PORTFOLIO = {
    "Dynamic_Programming": {
        "complexity": "O(N * W)",
        "space": "O(W)",
        "optimal_when": "Small to medium capacity W (pseudo-polynomial)"
    },
    "Branch_and_Bound": {
        "complexity": "O(2^N) worst-case, fast in practice",
        "space": "O(2^N) queue",
        "optimal_when": "Large W with correlated items and strong bounding pruning"
    },
    "Meet_in_the_Middle": {
        "complexity": "O(2^(N/2))",
        "space": "O(2^(N/2))",
        "optimal_when": "N <= 32 and capacity W is very large"
    },
    "Greedy_Dantzig": {
        "complexity": "O(N log N)",
        "space": "O(N)",
        "optimal_when": "Instant real-time approximation needed under tight deadline"
    }
}
