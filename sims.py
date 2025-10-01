"""
Simulation environment for budgeted bipartite matching algorithms.

This module runs simulations comparing greedy and optimal algorithms under
various valuation and cost distributions, and generates comprehensive
visualizations for theoretical analysis.
"""

import numpy as np
import scipy.stats as stats

from budgeted_bipartite_matching_algs import (
    create_edge_list,
    greedy_matching,
    budgeted_bipartite_matching_solver
)
from visualizations import (
    plot_distribution_pdfs,
    plot_social_welfare,
    plot_approximation_ratio,
    plot_approximation_histogram,
    plot_approximation_cdf,
    plot_ratio_vs_budget_normalized,
    print_approximation_summary
)


def simulate_block(dist, tx_size_dists, N, num_positions, B, T):
    """
    Run T simulation trials for a given configuration.

    Parameters:
    -----------
    dist : scipy.stats distribution
        Distribution for generating valuations
    tx_size_dists : scipy.stats distribution
        Distribution for generating costs
    N : int
        Number of agents (left nodes)
    num_positions : int
        Number of positions (right nodes)
    B : float
        Budget constraint
    T : int
        Number of trials to run

    Returns:
    --------
    tuple
        (sols, bounds, opts, approxs, max_approx, max_costs) where:
        - sols: greedy solution values
        - bounds: theoretical bounds
        - opts: optimal solution values
        - approxs: approximation ratios (OPT/SOL)
        - max_approx: maximum approximation ratio
        - max_costs: maximum cost in each trial
    """
    opts = np.zeros(T)
    sols = np.zeros(T)
    bounds = np.zeros(T)
    approxs = np.zeros(T)
    max_costs = np.zeros(T)
    max_approx = 0

    for t in range(T):
        # Generate random instance
        valuations = dist.rvs(size=(N, num_positions))
        costs = tx_size_dists.rvs(N)
        max_cost = np.max(costs)
        max_costs[t] = max_cost

        # Create edge list representation
        edges = create_edge_list(valuations=valuations, costs=costs)

        # Run greedy matching algorithm
        _, sol_value = greedy_matching(edges, Budget=B)
        sols[t] = sol_value

        # Calculate the optimal solution using the budgeted matching solver
        _, opt = budgeted_bipartite_matching_solver(edges=edges, Budget=B)
        opts[t] = opt if opt is not None else 0

        # Theoretical bound (placeholder - adjust based on your theory)
        bounds[t] = 6 * opts[t] if opt is not None else 0

        # Approximation ratio
        approxs[t] = opts[t] / sols[t] if sols[t] > 0 else 0
        max_approx = max(max_approx, approxs[t])

    return sols, bounds, opts, approxs, max_approx, max_costs


def main():
    """Main simulation routine."""

    # ========================================================================
    # SIMULATION PARAMETERS
    # ========================================================================

    # Problem parameters
    B_plus = 100
    B_minus = 0.1
    N = 150  # Number of agents
    num_positions = 15  # Number of positions

    # Budget range to test
    Bs = np.arange(50, 200, 15)

    # Number of simulation trials per configuration
    T = 15

    # ========================================================================
    # DISTRIBUTION SETUP
    # ========================================================================

    # Cost distributions
    cost_dists = [
        stats.uniform(loc=B_minus, scale=B_plus-B_minus),  # Uniform
        stats.pareto(b=2, scale=B_minus),                   # Heavy-tailed
        stats.loguniform(B_minus, B_plus)                   # Log-uniform
    ]
    cost_dist_labels = ["Uniform", "Pareto", "LogUniform"]

    # Use subset for faster testing (comment out to use all)
    cost_dists = cost_dists[:1]
    cost_dist_labels = cost_dist_labels[:1]

    # Valuation distributions
    light_dists = [stats.expon(scale=2.5), stats.rayleigh(scale=1)]
    light_dist_labels = ["Exponential", "Rayleigh"]

    heavy_dists = [stats.pareto(b=1.5), stats.levy_stable(alpha=1.5, beta=0)]
    heavy_dist_labels = ["Pareto", "LevyStable"]

    dists = light_dists + heavy_dists
    dist_labels = light_dist_labels + heavy_dist_labels

    # ========================================================================
    # PLOT DISTRIBUTION PDFs
    # ========================================================================

    print("="*70)
    print("BUDGETED BIPARTITE MATCHING SIMULATION")
    print("="*70)
    print(f"Agents: {N}")
    print(f"Positions: {num_positions}")
    print(f"Budget range: {Bs[0]} to {Bs[-1]}")
    print(f"Trials per configuration: {T}")
    print(f"Distributions: {dist_labels}")
    print(f"Cost distributions: {cost_dist_labels}")
    print("="*70)

    plot_distribution_pdfs(dists, dist_labels)

    # ========================================================================
    # RUN SIMULATIONS
    # ========================================================================

    print("\nRunning simulations...")

    # Initialize result arrays
    sols = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
    bounds = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
    opts = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
    approxs = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
    max_approxs = np.zeros((len(Bs), len(dists), len(cost_dists)))
    costs_data = {}  # Store max_cost for each trial

    # Run simulations for all combinations
    total_sims = len(cost_dists) * len(dists) * len(Bs)
    sim_count = 0

    for k, cost_dist in enumerate(cost_dists):
        for i, dist in enumerate(dists):
            for j, B in enumerate(Bs):
                sim_count += 1
                print(f"  Progress: {sim_count}/{total_sims} "
                      f"({dist_labels[i]}, {cost_dist_labels[k]}, B={B:.0f})")

                ret = simulate_block(dist, cost_dist, N, num_positions, B, T)
                sols[:, j, i, k] = ret[0]
                bounds[:, j, i, k] = ret[1]
                opts[:, j, i, k] = ret[2]
                approxs[:, j, i, k] = ret[3]
                max_approxs[j, i, k] = ret[4]

                # Store max_costs for scatter plot
                for t in range(T):
                    costs_data[(t, j, i, k)] = ret[5][t]

    print("\nSimulations completed!")

    # ========================================================================
    # COMPUTE STATISTICS
    # ========================================================================

    means = np.mean(sols, axis=0)
    stds = np.std(sols, axis=0)
    opts_mean = np.mean(opts, axis=0)
    approxs_means = np.mean(approxs, axis=0)
    approxs_std = np.std(approxs, axis=0)

    # ========================================================================
    # GENERATE STANDARD PLOTS
    # ========================================================================

    print("\n" + "="*70)
    print("Generating standard visualizations...")
    print("="*70)

    colors = ['coral', 'indigo', 'firebrick', 'mediumblue']

    # Social welfare plots
    print("\n1. Social welfare plots (Greedy vs Optimal)...")
    plot_social_welfare(Bs, means, stds, opts_mean, dist_labels, cost_dist_labels, colors)

    # Approximation ratio plots
    print("2. Approximation ratio plots (OPT/SOL vs Budget)...")
    plot_approximation_ratio(Bs, approxs_means, approxs_std, dist_labels, cost_dist_labels, colors)

    # Print summary
    print_approximation_summary(max_approxs, dist_labels, cost_dist_labels)

    # ========================================================================
    # GENERATE THEORETICAL ANALYSIS PLOTS
    # ========================================================================

    print("\n" + "="*70)
    print("Generating theoretical analysis visualizations...")
    print("="*70)

    # Histogram of approximation ratios
    print("\n1. Histograms of approximation ratios...")
    plot_approximation_histogram(approxs, dist_labels, cost_dist_labels, theoretical_bound=2.0)

    # CDF plots showing tail behavior
    print("2. CDFs of approximation ratios (tail behavior)...")
    plot_approximation_cdf(approxs, dist_labels, cost_dist_labels, theoretical_bound=2.0)

    # Scatter plot: ratio vs budget/max_cost
    print("3. Scatter plots (ratio vs budget tightness)...")
    plot_ratio_vs_budget_normalized(approxs, Bs, costs_data, dist_labels, cost_dist_labels)

    print("\n" + "="*70)
    print("All visualizations generated successfully!")
    print("Figures saved in: figures/")
    print("="*70)


if __name__ == "__main__":
    main()
