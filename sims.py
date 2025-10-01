import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


import math
import random

from budgeted_bipartite_matching_algs import create_edge_list, greedy_matching, budgeted_bipartite_matching_solver



def simulate_block(dist, tx_size_dists, N, B, T):
    opts = np.zeros(T)
    sols = np.zeros(T)
    bounds = np.zeros(T)
    approxs = np.zeros(T)
    max_costs = np.zeros(T)  # Track max_cost for each trial
    max_approx = 0
    for t in range(T):

        valuations = dist.rvs(size=(N, num_positions))
        costs = tx_size_dists.rvs(N)
        max_cost = np.max(costs)
        max_costs[t] = max_cost

        # Create edge list (more efficient than NetworkX graph)
        edges = create_edge_list(valuations=valuations, costs=costs)

        # Run greedy matching algorithm
        sol_matching, sol_value = greedy_matching(edges, Budget=B)
        sols[t] = sol_value

        # Calculate the bound: SOL >= (1/2 - max_cost/B) * OPT by Large Markets Lemma
        # bounds[t] = (1/(1/2 - max_cost/B)) * sols[t]


        # Calculate the optimal solution using the budgeted matching solver
        opt_matching, opt = budgeted_bipartite_matching_solver(edges=edges, Budget=B)
        opts[t] = opt if opt is not None else 0

        # Theoretical bound
        bounds[t] = 6*opts[t] if opt is not None else 0

        approxs[t] = opts[t] / sols[t] if sols[t] > 0 else 0
        max_approx = max(max_approx, approxs[t])

    return sols, bounds, opts, approxs, max_approx, max_costs


# Constants of the setting
B_plus = 100
B_minus = 0.1
N = 150
num_positions = 15
# Bs: block size
Bs = np.arange(50, 200, 15)
# Number of simulations
T= 15

# Cost distributions
cost_dists = [
    stats.uniform(loc=B_minus, scale=B_plus-B_minus),  # Uniform (current)
    stats.pareto(b=2, scale=B_minus),                   # Heavy-tailed costs
    stats.loguniform(B_minus, B_plus)                   # Log-uniform costs
]
cost_dists = cost_dists[:1]  # Only use uniform costs for faster testing


cost_dist_labels = ["Uniform", "Pareto", "LogUniform"]
cost_dist_labels = cost_dist_labels[:1]  # Only use uniform costs for faster testing

# Light-tailed distributions
light_dists = [stats.expon(scale=2.5), stats.rayleigh(scale=1)]
light_dist_labels = ["Exponential", "Rayleigh"]

# Heavy-tailed distributions
heavy_dists = [stats.pareto(b=1.5), stats.levy_stable(alpha=1.5, beta=0)]
heavy_dist_labels = ["Pareto", "LevyStable"]

dists = light_dists + heavy_dists
dist_labels = light_dist_labels + heavy_dist_labels


x = np.linspace(0, 10, 1000)

for i, dist in enumerate(dists):
    y = dist.pdf(x)
    plt.plot(x, y, label=dist_labels[i])
plt.legend()
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Probability Density Functions')
plt.show()
plt.savefig(f"distributions.pdf", dpi=300, bbox_inches='tight')


sols = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
theoretical_bounds = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
opts = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
approxs = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
max_approxs = np.zeros((len(Bs), len(dists), len(cost_dists)))
costs_data = {}  # Store max_cost for each trial

for k, cost_dist in enumerate(cost_dists):
    for i, dist in enumerate(dists):
        for j, B in enumerate(Bs):
            ret = simulate_block(dist, cost_dist, N, B, T)
            sols[:, j, i, k] = ret[0]
            theoretical_bounds[:, j, i, k] = ret[1]
            opts[:, j, i, k] = ret[2]
            approxs[:, j, i, k] = ret[3]
            max_approxs[j, i, k] = ret[4]
            # Store max_costs for scatter plot
            for t in range(T):
                costs_data[(t, j, i, k)] = ret[5][t]

means = np.mean(sols, axis=0)
meds = np.median(sols, axis=0)
stds = np.std(sols, axis=0)
theoretical_bounds = np.mean(theoretical_bounds, axis=0)
opts = np.mean(opts, axis=0)
approxs_means = np.mean(approxs, axis=0)
approxs_std = np.std(approxs, axis=0)

# plot results
colors = ['coral', 'indigo', 'firebrick', 'mediumblue']

# Main Loop - Plot for each valuation distribution and cost distribution
for k, cost_label in enumerate(cost_dist_labels):
    for i, label in zip(range(len(dists)), dist_labels):
        plt.figure(figsize=(8, 6))
        plt.plot(Bs, means[:, i, k], lw=2, label='Greedy', color=colors[i])
        plt.plot(Bs, opts[:, i, k], lw=2, label='Optimal', linestyle='--', color='black')
        plt.fill_between(Bs, means[:, i, k] - stds[:, i, k], means[:, i, k] + stds[:, i, k], alpha=0.5, color=colors[i])

        plt.plot(Bs, theoretical_bounds[:, i, k], lw=2, label='Theoretical Bound', linestyle=':', color='green')



        # Set plot attributes
        plt.yscale('log')
        plt.xlabel('Block Size (B)')
        plt.ylabel('Social Welfare')
        plt.title(f'Social Welfare vs Block Size ({label}, {cost_label} costs)')
        plt.legend()
        plt.grid(True)

        # Save and display the plot
        plt.savefig(f"SW-{dist_labels[i]}-{cost_label}.pdf", dpi=300, bbox_inches='tight')
        plt.show()


for k, cost_label in enumerate(cost_dist_labels):
    for i, label in zip(range(len(dists)), dist_labels):
        plt.figure(figsize=(8, 6))
        plt.plot(Bs, approxs_means[:, i, k], lw=2, label='Approximation ratio', color=colors[i])
        plt.fill_between(Bs, approxs_means[:, i, k] - approxs_std[:, i, k], approxs_means[:, i, k] + approxs_std[:, i, k], alpha=0.5, color=colors[i])

        # Set plot attributes
        plt.xlabel('Block Size (B)')
        plt.ylabel('OPT/SOL')
        plt.title(f'OPT/SOL vs Block Size ({label} valuations, {cost_label} costs)')
        plt.legend()
        plt.grid(True)

        # Save and display the plot
        plt.savefig(f"Approx-{dist_labels[i]}-{cost_label}.pdf", dpi=300, bbox_inches='tight')
        plt.show()

# Print maximum approximation ratios for each distribution and cost combination
print("\nMaximum Approximation Ratios (OPT/SOL) across all trials:")
for k, cost_label in enumerate(cost_dist_labels):
    print(f"\n{cost_label} costs:")
    for i, label in enumerate(dist_labels):
        max_ratio = np.max(max_approxs[:, i, k])
        print(f"  {label}: {max_ratio:.4f}")


# ============================================================================
# THEORETICAL ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_approximation_histogram(approxs, dist_labels, cost_dist_labels, theoretical_bound=None):
    """
    Plot histogram of approximation ratios across all trials.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    theoretical_bound : float, optional
        Theoretical worst-case bound to display as vertical line
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            # Flatten all approximation ratios for this (dist, cost) combination
            ratios = approxs[:, :, i, k].flatten()
            ratios = ratios[ratios > 0]  # Remove zeros

            plt.figure(figsize=(8, 6))
            plt.hist(ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

            # Add theoretical bound if provided
            if theoretical_bound is not None:
                plt.axvline(theoretical_bound, color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Bound = {theoretical_bound}')

            # Add mean and max lines
            plt.axvline(np.mean(ratios), color='green', linestyle='-', linewidth=2,
                       label=f'Mean = {np.mean(ratios):.3f}')
            plt.axvline(np.max(ratios), color='orange', linestyle='-', linewidth=2,
                       label=f'Max = {np.max(ratios):.3f}')

            plt.xlabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Distribution of Approximation Ratios\n({label} valuations, {cost_label} costs)', fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(f"Hist-Approx-{label}-{cost_label}.pdf", dpi=300, bbox_inches='tight')
            plt.show()


def plot_approximation_cdf(approxs, dist_labels, cost_dist_labels, theoretical_bound=None):
    """
    Plot empirical CDF of approximation ratios to show tail behavior.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    theoretical_bound : float, optional
        Theoretical worst-case bound to display as vertical line
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            # Flatten all approximation ratios for this (dist, cost) combination
            ratios = approxs[:, :, i, k].flatten()
            ratios = ratios[ratios > 0]  # Remove zeros

            # Sort for CDF
            sorted_ratios = np.sort(ratios)
            cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

            plt.figure(figsize=(8, 6))
            plt.plot(sorted_ratios, cdf, linewidth=2, color='steelblue')

            # Add theoretical bound if provided
            if theoretical_bound is not None:
                plt.axvline(theoretical_bound, color='red', linestyle='--', linewidth=2,
                           label=f'Theoretical Bound = {theoretical_bound}')

            # Highlight tail (e.g., 95th percentile)
            p95 = np.percentile(ratios, 95)
            plt.axvline(p95, color='orange', linestyle=':', linewidth=2,
                       label=f'95th percentile = {p95:.3f}')

            plt.xlabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.ylabel('Cumulative Probability', fontsize=12)
            plt.title(f'CDF of Approximation Ratios\n({label} valuations, {cost_label} costs)', fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(f"CDF-Approx-{label}-{cost_label}.pdf", dpi=300, bbox_inches='tight')
            plt.show()


def plot_ratio_vs_budget_normalized(approxs, Bs, costs_data, dist_labels, cost_dist_labels):
    """
    Scatter plot: Max approximation ratio vs normalized budget (B/max_cost).
    Shows relationship between budget tightness and algorithm performance.

    Parameters:
    -----------
    approxs : np.ndarray
        Array of shape (T, len(Bs), len(dists), len(cost_dists))
    Bs : np.ndarray
        Array of budget values
    costs_data : dict
        Dictionary storing max_cost for each trial, indexed by (trial, B_idx, dist_idx, cost_idx)
    dist_labels : list
        Labels for valuation distributions
    cost_dist_labels : list
        Labels for cost distributions
    """
    for k, cost_label in enumerate(cost_dist_labels):
        for i, label in enumerate(dist_labels):
            budget_ratios = []
            approx_ratios = []

            # Collect all (budget_ratio, approx_ratio) pairs
            for t in range(approxs.shape[0]):
                for j, B in enumerate(Bs):
                    if (t, j, i, k) in costs_data:
                        max_cost = costs_data[(t, j, i, k)]
                        budget_ratio = B / max_cost if max_cost > 0 else 0
                        approx_ratio = approxs[t, j, i, k]

                        if approx_ratio > 0 and budget_ratio > 0:
                            budget_ratios.append(budget_ratio)
                            approx_ratios.append(approx_ratio)

            if len(budget_ratios) == 0:
                continue

            plt.figure(figsize=(8, 6))
            plt.scatter(budget_ratios, approx_ratios, alpha=0.5, s=30, color='steelblue')

            # Add trend line
            if len(budget_ratios) > 1:
                z = np.polyfit(budget_ratios, approx_ratios, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(budget_ratios), max(budget_ratios), 100)
                plt.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend (degree 2)')

            plt.xlabel('Budget / Max Cost', fontsize=12)
            plt.ylabel('Approximation Ratio (OPT/SOL)', fontsize=12)
            plt.title(f'Ratio vs Budget Tightness\n({label} valuations, {cost_label} costs)', fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(f"Scatter-Ratio-Budget-{label}-{cost_label}.pdf", dpi=300, bbox_inches='tight')
            plt.show()


# ============================================================================
# Generate theoretical analysis plots
# ============================================================================

print("\n" + "="*70)
print("Generating theoretical analysis visualizations...")
print("="*70)

# 1. Histogram of approximation ratios
print("\n1. Plotting histograms of approximation ratios...")
plot_approximation_histogram(approxs, dist_labels, cost_dist_labels, theoretical_bound=2.0)

# 2. CDF plots showing tail behavior
print("2. Plotting CDFs of approximation ratios...")
plot_approximation_cdf(approxs, dist_labels, cost_dist_labels, theoretical_bound=2.0)

# 3. Scatter plot: ratio vs budget/max_cost
print("3. Plotting scatter plots of approximation ratio vs budget/max_cost...")
plot_ratio_vs_budget_normalized(approxs, Bs, costs_data, dist_labels, cost_dist_labels)

print("\nTheoretical analysis plots generated successfully!")
print("="*70)
















