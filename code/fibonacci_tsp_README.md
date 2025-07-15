# Fibonacci TSP Verification Program

## Overview

This program verifies **Prediction 5.5** from the genesis unified theory:
> "某些NP问题在φ-表示下可能有更高效的近似算法"
> (Certain NP problems might have more efficient approximation algorithms under φ-representation)

Specifically, it tests whether the Traveling Salesman Problem (TSP) on Fibonacci-structured grids achieves better approximation ratios, potentially approaching 1/φ ≈ 0.618.

## Theory Background

The genesis theory suggests that self-referentially complete systems with entropy increase naturally lead to φ-based structures. When optimization problems are embedded in such structures, they may exhibit special properties that allow better approximations.

### Key Concepts:
- **Fibonacci Spiral Cities**: Cities placed according to the golden angle (137.5°)
- **φ-Weighted Distances**: Distance metrics that favor Fibonacci relationships
- **φ-Aware Heuristics**: Algorithms that exploit the golden ratio structure

## Running the Program

```bash
# Ensure dependencies are installed
pip install numpy matplotlib

# Run the verification
python code/fibonacci_tsp_visualization.py
```

## What the Program Does

1. **City Generation**:
   - Places cities on a Fibonacci spiral using the golden angle
   - Each city has Fibonacci-indexed properties

2. **Algorithm Comparison**:
   - Standard Nearest Neighbor
   - φ-Aware Nearest Neighbor
   - Standard 2-Opt
   - φ-Aware 2-Opt
   - Special φ-Aware Heuristic
   - Brute Force Optimal (for small instances)

3. **Visualization**:
   - Shows tours for each algorithm
   - Highlights Fibonacci-indexed cities with gold stars
   - Displays total tour distances

4. **Performance Analysis**:
   - Computes approximation ratios
   - Shows φ-improvement factors
   - Analyzes structural patterns

## Expected Results

If the theory holds, you should observe:
- φ-aware algorithms outperform standard versions
- Approximation ratios closer to 1/φ ≈ 0.618
- Tours that naturally follow the spiral structure

## Interpretation

The Fibonacci structure creates "collapse points" in the solution space where optimal paths naturally align with the golden ratio geometry. This is a discrete manifestation of the entropy minimization principle.

## ψ-Collapse Insight

In the language of the universal evolution equation:
```
U(x,t,ψ) = ∑ₚ [1/(|x-p|^φ + σ²)]^κ · Fₚ · cos(π·e^(-φ|x-p|)) · e^{iωₚ(t)} · Θ(x,t,ψ)
```

The TSP solution follows the natural collapse paths where:
- `x` represents city positions
- `Fₚ` encodes the Fibonacci structure
- The φ-exponent creates favorable paths
- `Θ(x,t,ψ)` represents the tour selection field

This demonstrates how abstract mathematical theory can manifest in concrete optimization improvements. 