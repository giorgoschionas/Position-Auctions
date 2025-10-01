# Position Auctions: Budgeted Bipartite Matching Simulation

This repository implements and evaluates algorithms for the **Budgeted Bipartite Matching** problem, with a focus on position auction mechanisms. The codebase provides a comprehensive simulation environment to analyze algorithm performance under various valuation and cost distributions.

## Problem Overview

**Budgeted Bipartite Matching** is an optimization problem where:
- A set of **agents** (left nodes) want to match with a set of **positions** (right nodes)
- Each edge has a **value** (social welfare) and a **cost** (transaction size/fee)
- Each agent can be matched to at most one position (and vice versa)
- Total cost of selected edges must not exceed a **budget** constraint
- **Objective**: Maximize total value subject to budget and matching constraints

## Repository Structure

### Core Algorithms (`budgeted_bipartite_matching_algs.py`)

Implements three main algorithms:

1. **`create_edge_list(valuations, costs)`**
   - Converts valuation matrix and cost array into efficient edge list representation
   - Each edge: `(u, v, value, cost, density)` where density = value/cost

2. **`greedy_matching(edges, Budget)`**
   - Greedy algorithm with backtracking for budgeted matching
   - Sorts edges by density (value/cost ratio) in descending order
   - Supports edge replacement: can replace a matched edge if new edge has higher value and fits budget
   - Backtracking: when an edge is replaced, algorithm reconsiders previously skipped edges
   - Returns: `(matching, total_value)`

3. **`budgeted_bipartite_matching_solver(edges, Budget)`**
   - Optimal solver using Gurobi MIP (Mixed Integer Programming)
   - Provides exact optimal solution for comparison
   - Returns: `(matching_edges, total_value)`

### Simulation Environment (`sims.py`)

**Main simulation framework** for evaluating algorithm performance:

#### Key Components:

- **`simulate_block(dist, tx_size_dists, N, B, T)`**
  - Runs T simulation trials for a given configuration
  - Generates random instances from specified distributions
  - Compares greedy vs optimal solutions
  - Tracks approximation ratios (OPT/GREEDY)
  - Returns: `(sols, bounds, opts, approxs, max_approx)`

#### Experimental Setup:

- **Valuation Distributions:**
  - Light-tailed: Exponential, Rayleigh
  - Heavy-tailed: Pareto, Levy Stable

- **Cost Distributions:**
  - Uniform: baseline with similar costs
  - Pareto: heavy-tailed costs (cheap and expensive agents)
  - Log-uniform: exponentially distributed costs

- **Parameters:**
  - `N`: Number of agents (default: 150)
  - `num_positions`: Number of positions (default: 15)
  - `Bs`: Range of budget values to test
  - `T`: Number of trials per configuration (default: 15-20)

#### Output:

The simulation generates:
- **Social Welfare plots**: Greedy vs Optimal as function of budget
- **Approximation ratio plots**: OPT/SOL vs budget size
- **Maximum approximation ratios**: Worst-case performance across all trials
- Results saved as PDF files for each (distribution, cost) combination

### Examples

#### `example.py`
Demonstrates an adversarial instance with approximation ratio ~2.33:
- One "cheap giant" with high value and tiny cost
- One "decoy" with slightly higher value but huge cost
- Multiple "filler" agents with moderate values
- Shows how greedy's replacement mechanism can be exploited


## Key Insights

### Approximation Ratio Analysis

1. **Light-tailed distributions** (Exponential, Rayleigh):
   - Greedy performs near-optimally (ratio ≤ 1.06)
   - Valuations are relatively homogeneous, no structure to exploit

2. **Heavy-tailed distributions** (Pareto, Levy Stable):
   - Expected to produce higher approximation ratios
   - Extreme outliers can create adversarial scenarios
   - Combined with heavy-tailed costs, ratios may increase significantly

3. **Budget vs Cost relationship**:
   - Larger budget relative to costs → smaller approximation ratio
   - When budget fits most agents, greedy ≈ optimal
   - Tight budgets create hardest instances for greedy

4. **Adversarial instances**:
   - Handcrafted examples can achieve ratio ~2.33
   - Decoys that consume ~75% of budget create effective traps
   - Natural distributions rarely exhibit pathological structure

## Dependencies

```
numpy
scipy
matplotlib
networkx
gurobipy
```

## Usage

### Run Simulations

```bash
python sims.py
```

This will:
- Test all combinations of valuation and cost distributions
- Generate plots for social welfare and approximation ratios
- Print maximum approximation ratios for each configuration

### Run Examples

```bash
python example.py                 # Original adversarial example (ratio ~2.33)
```


