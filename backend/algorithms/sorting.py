"""
Sorting Algorithms Suite for DAA Project:
Includes QuickSort, MergeSort, HeapSort, InsertionSort, TimSort, CountingSort, RadixSort.
Instrumented to collect execution time, comparisons, swaps/moves, and memory allocation.
"""

import time
import math
from typing import List, Tuple, Dict, Any

class SortingResult:
    def __init__(self, algorithm: str, sorted_arr: List[Any], duration_us: float, comparisons: int, swaps: int, is_correct: bool):
        self.algorithm = algorithm
        self.sorted_arr = sorted_arr
        self.duration_us = duration_us
        self.comparisons = comparisons
        self.swaps = swaps
        self.is_correct = is_correct

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "duration_us": round(self.duration_us, 2),
            "duration_ms": round(self.duration_us / 1000.0, 4),
            "comparisons": self.comparisons,
            "swaps": self.swaps,
            "is_correct": self.is_correct,
            "sample_output": self.sorted_arr[:15] if len(self.sorted_arr) > 15 else self.sorted_arr
        }


# ==========================================
# 1. QuickSort (Median-of-Three Partitioning)
# ==========================================
def quick_sort(arr: List[float]) -> Tuple[List[float], int, int]:
    comparisons = 0
    swaps = 0
    a = arr.copy()
    n = len(a)

    def _sort(low: int, high: int):
        nonlocal comparisons, swaps
        if low < high:
            # Median-of-three pivot selection to prevent worst-case on sorted data
            mid = (low + high) // 2
            if a[mid] < a[low]:
                a[low], a[mid] = a[mid], a[low]
                swaps += 1
            comparisons += 1
            if a[high] < a[low]:
                a[low], a[high] = a[high], a[low]
                swaps += 1
            comparisons += 1
            if a[high] < a[mid]:
                a[mid], a[high] = a[high], a[mid]
                swaps += 1
            comparisons += 1
            
            a[mid], a[high] = a[high], a[mid]
            swaps += 1
            pivot = a[high]

            i = low - 1
            for j in range(low, high):
                comparisons += 1
                if a[j] <= pivot:
                    i += 1
                    a[i], a[j] = a[j], a[i]
                    swaps += 1
            a[i + 1], a[high] = a[high], a[i + 1]
            swaps += 1
            p = i + 1

            _sort(low, p - 1)
            _sort(p + 1, high)

    _sort(0, n - 1)
    return a, comparisons, swaps


# ==========================================
# 2. MergeSort
# ==========================================
def merge_sort(arr: List[float]) -> Tuple[List[float], int, int]:
    comparisons = 0
    moves = 0
    a = arr.copy()

    def _merge_sort(sub: List[float]) -> List[float]:
        nonlocal comparisons, moves
        if len(sub) <= 1:
            return sub
        mid = len(sub) // 2
        left = _merge_sort(sub[:mid])
        right = _merge_sort(sub[mid:])

        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            comparisons += 1
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
            moves += 1

        while i < len(left):
            merged.append(left[i])
            i += 1
            moves += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
            moves += 1
        return merged

    sorted_res = _merge_sort(a)
    return sorted_res, comparisons, moves


# ==========================================
# 3. HeapSort
# ==========================================
def heap_sort(arr: List[float]) -> Tuple[List[float], int, int]:
    comparisons = 0
    swaps = 0
    a = arr.copy()
    n = len(a)

    def heapify(length: int, i: int):
        nonlocal comparisons, swaps
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < length:
            comparisons += 1
            if a[left] > a[largest]:
                largest = left

        if right < length:
            comparisons += 1
            if a[right] > a[largest]:
                largest = right

        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            swaps += 1
            heapify(length, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        swaps += 1
        heapify(i, 0)

    return a, comparisons, swaps


# ==========================================
# 4. InsertionSort (Adaptive)
# ==========================================
def insertion_sort(arr: List[float]) -> Tuple[List[float], int, int]:
    comparisons = 0
    shifts = 0
    a = arr.copy()
    n = len(a)

    for i in range(1, n):
        key = a[i]
        j = i - 1
        while j >= 0:
            comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                shifts += 1
                j -= 1
            else:
                break
        a[j + 1] = key
        shifts += 1

    return a, comparisons, shifts


# ==========================================
# 5. TimSort (Hybrid Natural Run-Based Sort)
# ==========================================
def tim_sort(arr: List[float]) -> Tuple[List[float], int, int]:
    comparisons = 0
    moves = 0
    a = arr.copy()
    n = len(a)
    RUN = 32

    def binary_insertion_sort(sub_arr: List[float], left: int, right: int):
        nonlocal comparisons, moves
        for i in range(left + 1, right + 1):
            temp = sub_arr[i]
            l = left
            r = i - 1
            # Binary search for position
            while l <= r:
                mid = (l + r) // 2
                comparisons += 1
                if temp < sub_arr[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # Shift elements
            for j in range(i - 1, l - 1, -1):
                sub_arr[j + 1] = sub_arr[j]
                moves += 1
            sub_arr[l] = temp
            moves += 1

    def merge_runs(sub_arr: List[float], l: int, m: int, r: int):
        nonlocal comparisons, moves
        len1 = m - l + 1
        len2 = r - m
        left = sub_arr[l:m + 1]
        right = sub_arr[m + 1:r + 1]

        i = j = 0
        k = l
        while i < len1 and j < len2:
            comparisons += 1
            if left[i] <= right[j]:
                sub_arr[k] = left[i]
                i += 1
            else:
                sub_arr[k] = right[j]
                j += 1
            moves += 1
            k += 1

        while i < len1:
            sub_arr[k] = left[i]
            i += 1
            k += 1
            moves += 1
        while j < len2:
            sub_arr[k] = right[j]
            j += 1
            k += 1
            moves += 1

    for i in range(0, n, RUN):
        binary_insertion_sort(a, i, min(i + RUN - 1, n - 1))

    size = RUN
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min(left + 2 * size - 1, n - 1)
            if mid < right:
                merge_runs(a, left, mid, right)
        size = 2 * size

    return a, comparisons, moves


# ==========================================
# 6. CountingSort (O(N + K) Non-comparison)
# ==========================================
def counting_sort(arr: List[int]) -> Tuple[List[int], int, int]:
    # Returns (sorted_arr, comparisons=0, operations)
    if not arr:
        return [], 0, 0
    
    # Check if elements are integers
    min_val = min(arr)
    max_val = max(arr)
    k_range = int(max_val - min_val) + 1
    
    # Safety cap: don't allocate massive memory if range is unreasonably large (> 5,000,000)
    if k_range > 5_000_000:
        raise ValueError(f"CountingSort range {k_range} too large for practical memory allocation.")

    count = [0] * k_range
    ops = 0

    for x in arr:
        count[int(x - min_val)] += 1
        ops += 1

    output = []
    for i, c in enumerate(count):
        val = i + min_val
        for _ in range(c):
            output.append(val)
            ops += 1

    return output, 0, ops


# ==========================================
# 7. RadixSort (LSD Integer Sort)
# ==========================================
def radix_sort(arr: List[int]) -> Tuple[List[int], int, int]:
    if not arr:
        return [], 0, 0
    
    # Handle negatives by shifting
    min_val = min(arr)
    shift = 0
    if min_val < 0:
        shift = -min_val
        a = [int(x + shift) for x in arr]
    else:
        a = [int(x) for x in arr]
    
    max_val = max(a)
    ops = 0
    exp = 1

    def count_sort_exp(arr_in: List[int], exp_val: int) -> List[int]:
        nonlocal ops
        n = len(arr_in)
        out = [0] * n
        count = [0] * 10

        for num in arr_in:
            idx = (num // exp_val) % 10
            count[idx] += 1
            ops += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        for i in range(n - 1, -1, -1):
            idx = (arr_in[i] // exp_val) % 10
            out[count[idx] - 1] = arr_in[i]
            count[idx] -= 1
            ops += 1

        return out

    while max_val // exp > 0:
        a = count_sort_exp(a, exp)
        exp *= 10

    if shift > 0:
        a = [x - shift for x in a]

    return a, 0, ops


# Portfolio Registry
SORTING_PORTFOLIO = {
    "QuickSort": {
        "fn": quick_sort,
        "type": "comparison",
        "best_complexity": "O(N log N)",
        "avg_complexity": "O(N log N)",
        "worst_complexity": "O(N^2)",
        "space_complexity": "O(log N)",
        "stable": False,
        "description": "Divide-and-conquer with median-of-three pivot; fast cache locality on random data."
    },
    "MergeSort": {
        "fn": merge_sort,
        "type": "comparison",
        "best_complexity": "O(N log N)",
        "avg_complexity": "O(N log N)",
        "worst_complexity": "O(N log N)",
        "space_complexity": "O(N)",
        "stable": True,
        "description": "Consistent O(N log N) runtime with guaranteed worst-case bounds; stable."
    },
    "HeapSort": {
        "fn": heap_sort,
        "type": "comparison",
        "best_complexity": "O(N log N)",
        "avg_complexity": "O(N log N)",
        "worst_complexity": "O(N log N)",
        "space_complexity": "O(1)",
        "stable": False,
        "description": "In-place binary max-heap sort; zero extra memory overhead."
    },
    "InsertionSort": {
        "fn": insertion_sort,
        "type": "comparison",
        "best_complexity": "O(N)",
        "avg_complexity": "O(N^2)",
        "worst_complexity": "O(N^2)",
        "space_complexity": "O(1)",
        "stable": True,
        "description": "Extremely fast constant factor for tiny arrays (N < 30) or nearly sorted data."
    },
    "TimSort": {
        "fn": tim_sort,
        "type": "comparison_hybrid",
        "best_complexity": "O(N)",
        "avg_complexity": "O(N log N)",
        "worst_complexity": "O(N log N)",
        "space_complexity": "O(N)",
        "stable": True,
        "description": "Industrial hybrid algorithm exploiting existing ascending/descending runs."
    },
    "CountingSort": {
        "fn": counting_sort,
        "type": "non_comparison",
        "best_complexity": "O(N + K)",
        "avg_complexity": "O(N + K)",
        "worst_complexity": "O(N + K)",
        "space_complexity": "O(K)",
        "stable": True,
        "description": "Linear time when range K is proportional to or smaller than array size N."
    },
    "RadixSort": {
        "fn": radix_sort,
        "type": "non_comparison",
        "best_complexity": "O(d * (N + b))",
        "avg_complexity": "O(d * (N + b))",
        "worst_complexity": "O(d * (N + b))",
        "space_complexity": "O(N + b)",
        "stable": True,
        "description": "Digit-by-digit non-comparison integer sorting; linear when number of digits d is small."
    }
}


def benchmark_single_sort(algorithm_name: str, arr: List[Any], repeats: int = 1) -> SortingResult:
    """Benchmarks a single sorting algorithm with high-precision timing."""
    if algorithm_name not in SORTING_PORTFOLIO:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    meta = SORTING_PORTFOLIO[algorithm_name]
    fn = meta["fn"]

    # Check if algorithm requires integer inputs
    is_all_int = all(isinstance(x, (int, float)) and int(x) == x for x in arr[:100])
    if meta["type"] == "non_comparison" and not is_all_int:
        return SortingResult(algorithm_name, [], 999_999_999, 0, 0, False)

    # For CountingSort, check range safety
    if algorithm_name == "CountingSort":
        if arr:
            min_v, max_v = min(arr), max(arr)
            if (max_v - min_v) > 2_000_000:
                # Range too huge for counting sort, penalize
                return SortingResult(algorithm_name, [], 999_999_999, 0, 0, False)

    # Convert to appropriate type
    input_data = [int(x) for x in arr] if meta["type"] == "non_comparison" else [float(x) for x in arr]

    times = []
    sorted_out = []
    comparisons = 0
    swaps = 0

    for _ in range(repeats):
        data_copy = input_data.copy()
        t0 = time.perf_counter_ns()
        try:
            sorted_out, comparisons, swaps = fn(data_copy)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0) # microseconds
        except Exception:
            return SortingResult(algorithm_name, [], 999_999_999, 0, 0, False)

    avg_duration = sum(times) / len(times)
    
    # Correctness check: is it non-decreasing?
    is_correct = True
    for i in range(len(sorted_out) - 1):
        if sorted_out[i] > sorted_out[i + 1]:
            is_correct = False
            break

    return SortingResult(algorithm_name, sorted_out, avg_duration, comparisons, swaps, is_correct)
