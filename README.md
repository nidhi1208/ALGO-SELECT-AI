# ALGO·SELECT AI: AI-Based Algorithm Selection Using Time Complexity & Machine Learning
> **Design and Analysis of Algorithms (DAA) — Capstone Research Project**  
> *An End-to-End Meta-Learning Framework for Automated Algorithm Portfolio Selection*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-91.11%25-brightgreen.svg)]()
[![Speedup](https://img.shields.io/badge/Avg%20Speedup%20Achieved-8.01x-cyan.svg)]()

---

## 📌 Executive Summary

Traditional algorithm analysis relies on asymptotic worst-case bounds ($\mathcal{O}(N \log N)$, $\mathcal{O}(N^2)$), ignoring crucial runtime factors such as CPU instruction caches, memory hierarchy latency, and input topology (e.g., pre-sortedness, entropy, duplicate density, graph sparsity). Consequently, **no single general-purpose algorithm is universally optimal across all problem instances** (an empirical consequence of the No Free Lunch theorem).

**ALGO·SELECT AI** implements **Rice's Algorithm Selection Framework (1976)** using Machine Learning. It extracts lightweight meta-features $f(x)$ from input instances in sub-millisecond linear time ($\mathcal{O}(N)$ or $\mathcal{O}(V + E)$) and employs supervised learning models (Random Forest, Gradient Boosting, MLP Neural Network, SVM, k-NN) to automatically predict the fastest algorithm from a competitive portfolio.

---

## 🔬 Theoretical Framework: Rice's Algorithm Selection Theorem

In 1976, John R. Rice formalized the algorithm selection problem:

$$\text{Given an instance } x \in \mathcal{P}, \text{ find } S(f(x)) = A^* \in \mathcal{A} \text{ such that } m(A^*, x) = \min_{A \in \mathcal{A}} m(A, x)$$

Where:
- $\mathcal{P}$: The problem instance space (e.g., arrays, graphs, knapsack items).
- $f: \mathcal{P} \to \mathbb{R}^d$: The feature extraction mapping.
- $\mathcal{A} = \{A_1, A_2, \dots, A_k\}$: The algorithm portfolio.
- $m(A_i, x)$: The empirical performance metric (wall-clock CPU time in microseconds).
- $S$: The selection mapping learned via Machine Learning.

### ⚠️ The Fundamental Meta-Learning Constraint
Meta-learning is only practically viable if the computational cost of feature extraction $T(f(x))$ is strictly negligible compared to algorithm runtime:

$$T(f(x)) \ll \min_{A \in \mathcal{A}} T(A(x))$$

In ALGO·SELECT AI, array meta-features are extracted in **strict $\mathcal{O}(N)$ time (sub-sampled to $< 0.35\text{ ms}$)**, ensuring zero net computational overhead.

---

## 🧩 Algorithm Portfolios

### 1. Sorting Algorithms Portfolio (Flagship Domain)
| Algorithm | Best Case | Average Case | Worst Case | Space | Key Empirical Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **QuickSort** (Median-of-3) | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(\log N)$ | Low constant factor and optimal CPU cache line locality on random data. |
| **MergeSort** | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | Strict worst-case guarantees and stable sorting. |
| **HeapSort** | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(1)$ | In-place guaranteed $\mathcal{O}(N \log N)$ with zero memory allocation. |
| **InsertionSort** | $\mathcal{O}(N)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(1)$ | Dominates for small arrays ($N \le 24$) and nearly sorted arrays due to minimal instruction overhead ($c_1 \ll c_2$). |
| **TimSort** | $\mathcal{O}(N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N \log N)$ | $\mathcal{O}(N)$ | Exploits natural monotonic runs in linear time. |
| **CountingSort** | $\mathcal{O}(N + K)$ | $\mathcal{O}(N + K)$ | $\mathcal{O}(N + K)$ | $\mathcal{O}(K)$ | Bypasses comparison lower bound $\Omega(N \log N)$ when range $K \le 2N$. |
| **RadixSort** (LSD) | $\mathcal{O}(d(N + b))$ | $\mathcal{O}(d(N + b))$ | $\mathcal{O}(d(N + b))$ | $\mathcal{O}(N + b)$ | Non-comparison digit sorting for large arrays of moderate integers. |

### 2. Graph Algorithms Portfolio
- **Minimum Spanning Tree (MST):**
  - **Kruskal's Algorithm** ($\mathcal{O}(E \log E)$ with DSU): Optimal on sparse graphs ($D < 0.15$).
  - **Prim's Algorithm (Min-Heap)** ($\mathcal{O}(E \log V)$): Optimal on intermediate density.
  - **Prim's Algorithm (Adjacency Matrix)** ($\mathcal{O}(V^2)$): Dominates on dense graphs ($D > 0.60$) by avoiding heap allocation overhead.
- **Shortest Path:**
  - **Dijkstra's Algorithm** ($\mathcal{O}((V + E) \log V)$): Optimal for non-negative edge weights.
  - **Bellman-Ford Algorithm** ($\mathcal{O}(V \cdot E)$): Mandatory when negative edge weights or negative cycles exist.
  - **Floyd-Warshall Algorithm** ($\mathcal{O}(V^3)$): Optimal for all-pairs shortest paths on compact graphs ($|V| \le 80$).

### 3. 0/1 Knapsack Portfolio
- **Dynamic Programming** ($\mathcal{O}(N \cdot W)$): Optimal when capacity $W$ is small ($W \le 5,000$).
- **Branch and Bound** (Best-First Search + Continuous LP Relaxation): Dominates on large $N$ and large $W$ via branch pruning.
- **Meet-in-the-Middle** ($\mathcal{O}(2^{N/2})$): Optimal when $N \le 32$ and $W$ is astronomical ($W \gg 10^5$).
- **Greedy Dantzig Approximation** ($\mathcal{O}(N \log N)$): Ultra-fast approximation under strict real-time deadlines.

---

## 📊 Extracted Meta-Features ($\mathcal{O}(N)$)

| Feature | Complexity | Mathematical Definition | Significance |
| :--- | :---: | :--- | :--- |
| `size` ($N$) | $\mathcal{O}(1)$ | Array length $\|x\|$ | Captures crossover thresholds ($N \le 24$). |
| `range_to_size_ratio` | $\mathcal{O}(N)$ | $(\max(x) - \min(x)) / N$ | Triggers CountingSort when ratio $\le 1.5$. |
| `sortedness_ratio` | $\mathcal{O}(N)$ | $\sum [x_i \le x_{i+1}] / (N-1)$ | Identifies adaptive advantages for TimSort / InsertionSort. |
| `reverse_sortedness_ratio` | $\mathcal{O}(N)$ | $\sum [x_i \ge x_{i+1}] / (N-1)$ | Detects inverse patterns that degrade naive partitioners. |
| `runs_ratio` | $\mathcal{O}(N)$ | $\text{MonotonicRuns}(x) / N$ | Key metric for TimSort run-merging. |
| `duplicate_ratio` | $\mathcal{O}(N)$ | $\|\text{Unique}(x)\| / N$ | Indicates 3-way partition efficiency. |
| `entropy_proxy` | $\mathcal{O}(N)$ | $-\sum p_i \log_2(p_i)$ over 10 histogram bins | Measures information dispersion. |
| `is_integer` | $\mathcal{O}(1)$ | $\forall x_i, x_i \in \mathbb{Z}$ | Determines non-comparison eligibility. |

---

## 🤖 Machine Learning Model Benchmarks

Trained and evaluated on **720 empirically benchmarked instances** across 9 distinct distribution topologies:

| Model Architecture | Train Accuracy | Test Accuracy | 5-Fold CV Mean | CV Std Dev | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (120 Trees) | **100.0%** | **91.11%** | **88.52%** | **&plusmn; 0.028** | 🥇 **Active Best Model** |
| **MLP Neural Network** (64x32) | 94.07% | 91.11% | 89.44% | &plusmn; 0.015 | Top Deep Baseline |
| **Support Vector Machine (RBF)** | 92.96% | 90.00% | 89.81% | &plusmn; 0.017 | Strong Margin Classifier |
| **Gradient Boosting** (GBDT) | 100.0% | 89.44% | 87.41% | &plusmn; 0.011 | Robust Ensemble |
| **k-Nearest Neighbors** ($k=5$) | 100.0% | 88.89% | 85.74% | &plusmn; 0.022 | Instance-Based Baseline |

### 🚀 Empirical Impact
- **Average Speedup Achieved over Default (Always QuickSort):** **8.01x Faster**
- **Average Wall-Clock CPU Time Saved:** **55.1% Reduction**
- **Peak Speedup Observed:** **117.2x** (CountingSort vs QuickSort on compact integer sets)

---

## 📁 Repository Structure

```
DAA Project 2/
├── run.py                          # Master one-click startup script (detects port & boots app)
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive documentation & research theory
├── backend/
│   ├── app.py                      # FastAPI application with REST endpoints
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── sorting.py              # Quick, Merge, Heap, Insertion, Tim, Counting, Radix
│   │   ├── graphs.py               # Kruskal, Prim (Heap/Matrix), Dijkstra, Bellman-Ford, Floyd-Warshall
│   │   └── knapsack.py             # DP, Branch & Bound, Meet-in-the-Middle, Greedy
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sorting_features.py     # O(N) linear meta-feature extractor
│   │   ├── graph_features.py       # O(V + E) graph structural metrics
│   │   └── knapsack_features.py    # Knapsack capacity/correlation metrics
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data_generator.py       # Synthetic dataset generation across 9 distributions
│   │   ├── trainer.py              # ML training pipeline (RF, GBDT, MLP, SVM, kNN)
│   │   ├── predictor.py            # Sub-millisecond inference + XAI rationale engine
│   │   └── models/                 # Saved joblib weights & model_metrics.json
│   ├── benchmark/
│   │   ├── __init__.py
│   │   └── runner.py               # High-precision microsecond race engine & scalability curves
│   └── tests/
│       ├── test_algorithms.py      # Mathematical correctness test suite
│       └── test_features.py        # Meta-feature validation test suite
└── frontend/
    ├── index.html                  # Cyberpunk-academic glassmorphism dashboard
    ├── styles.css                  # Custom Vanilla CSS design system & animations
    └── app.js                      # Real-time race animator, Chart.js telemetry & API client
```

---

## ⚡ Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run the master script:
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
Interactive API Swagger documentation is available at:
```
http://127.0.0.1:8000/docs
```

### 4. Running Verification Unit Tests
To verify mathematical correctness and feature extraction bounds:
```bash
python backend/tests/test_algorithms.py
python backend/tests/test_features.py
```

---

## 🌐 Interactive Dashboard Features

1. **⚡ Sorting Arena:**
   - **Distribution Presets:** Small Range (CountingSort), Nearly Sorted (TimSort), Small Array (InsertionSort), Random Uniform (QuickSort), High Duplicates, Reverse Sorted, Gaussian Floats.
   - **Real-Time Feature HUD:** Displays Sortedness, Range/N ratio, Duplicate ratio, and Monotonic runs extracted in $< 0.3\text{ ms}$.
   - **Live Benchmark Race Arena:** Executes all 7 sorting algorithms on identical data with animated race tracks, wall-clock telemetry ($\mu s$), comparisons, swaps, and JSON export.
2. **🕸️ Graph Networks:**
   - Interactive sliders for Vertices $|V|$, Density ($0\% \to 100\%$), and Negative Edge Weight toggles.
   - Dynamic algorithm selection between Kruskal, Prim (Heap), Prim (Matrix), Dijkstra, Bellman-Ford, and Floyd-Warshall.
3. **🎒 0/1 Knapsack Studio:**
   - Explores crossover boundaries between Dynamic Programming ($\mathcal{O}(NW)$), Branch & Bound, and Meet-in-the-Middle ($\mathcal{O}(2^{N/2})$).
4. **📊 ML Model Analytics & XAI:**
   - Leaderboard comparing Random Forest, Gradient Boosting, MLP, SVM, and k-NN.
   - Horizontal Gini Feature Importance chart.
   - Confusion Matrix recall diagnostics.
5. **📈 Complexity Scaling Curves:**
   - Interactive multi-line Chart.js plot showing empirical execution times as $N$ scales from 15 to 2,500.
6. **📖 Theory & Math:**
   - Full mathematical derivation of Rice's Theorem, Big-O vs CPU constant factors, and reference notes.

---

## 📜 Academic References
1. Rice, J. R. (1976). *The Algorithm Selection Problem*. Advances in Computers, 15, 65-118.
2. Kotthoff, L. (2014). *Algorithm Selection for Combinatorial Search Problems: A Survey*. AI Magazine, 35(3), 48-60.
3. Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). *Automated Algorithm Selection: Survey and Perspectives*. Evolutionary Computation, 27(1), 3-45.
4. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.

---
*Created for DAA Project 2 — Design and Analysis of Algorithms.*
