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
    max_approx = 0
    for t in range(T):

        valuations = dist.rvs(size=(N, num_positions))
        costs = tx_size_dists.rvs(N)
        max_cost = np.max(costs)

        # Create edge list (more efficient than NetworkX graph)
        edges = create_edge_list(valuations=valuations, costs=costs)

        # Run greedy matching algorithm
        sol_matching, sol_value = greedy_matching(edges, Budget=B)
        sols[t] = sol_value

        # Calculate the bound: SOL >= (1/2 - max_cost/B) * OPT
        bounds[t] = (1/(1/2 - max_cost/B)) * sols[t]

        # Calculate the optimal solution using the budgeted matching solver
        opt_matching, opt = budgeted_bipartite_matching_solver(edges=edges, Budget=B)
        opts[t] = opt if opt is not None else 0

        approxs[t] = opts[t] / sols[t] if sols[t] > 0 else 0
        max_approx = max(max_approx, approxs[t])

    return sols, bounds, opts, approxs, max_approx


# Constants of the setting
B_plus = 15
B_minus = 0.1
N = 150
num_positions = 15
# Bs: block size
Bs = np.arange(15, 200, 15)
# Number of simulations
T= 15

# Cost distributions
cost_dists = [
    stats.uniform(loc=B_minus, scale=B_plus-B_minus),  # Uniform (current)
    stats.pareto(b=2, scale=B_minus),                   # Heavy-tailed costs
    stats.loguniform(B_minus, B_plus)                   # Log-uniform costs
]
cost_dist_labels = ["Uniform", "Pareto", "LogUniform"]

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
bounds = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
opts = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
approxs = np.zeros((T, len(Bs), len(dists), len(cost_dists)))
max_approxs = np.zeros((len(Bs), len(dists), len(cost_dists)))

for k, cost_dist in enumerate(cost_dists):
    for i, dist in enumerate(dists):
        for j, B in enumerate(Bs):
            ret = simulate_block(dist, cost_dist, N, B, T)
            sols[:, j, i, k] = ret[0]
            bounds[:, j, i, k] = ret[1]
            opts[:, j, i, k] = ret[2]
            approxs[:, j, i, k] = ret[3]
            max_approxs[j, i, k] = ret[4]

means = np.mean(sols, axis=0)
meds = np.median(sols, axis=0)
stds = np.std(sols, axis=0)
bounds = np.mean(bounds, axis=0)
opts = np.mean(opts, axis=0)
approxs_means = np.mean(approxs, axis=0)
approxs_std = np.std(approxs, axis=0)

# plot results
colors = ['coral', 'indigo', 'firebrick', 'mediumblue']

# Main Loop - Plot for each valuation distribution and cost distribution
for k, cost_label in enumerate(cost_dist_labels):
    for i, label in zip(range(len(dists)), dist_labels):
        bound = bounds[:, i, k]
        plt.figure(figsize=(8, 6))
        plt.plot(Bs, means[:, i, k], lw=2, label='Greedy', color=colors[i])
        plt.plot(Bs, opts[:, i, k], lw=2, label='Optimal', linestyle='--', color='black')
        plt.fill_between(Bs, means[:, i, k] - stds[:, i, k], means[:, i, k] + stds[:, i, k], alpha=0.5, color=colors[i])

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
        bound = bounds[:, i, k]
        plt.figure(figsize=(8, 6))
        plt.plot(Bs, approxs_means[:, i, k], lw=2, label='Approximation ratio', color=colors[i])
        plt.fill_between(Bs, approxs_means[:, i, k] - approxs_std[:, i, k], approxs_means[:, i, k] + approxs_std[:, i, k], alpha=0.5, color=colors[i])

        # Set plot attributes
        plt.xlabel('Block Size (B)')
        plt.ylabel('OPT/SOL')
        plt.title(f'OPT/SOL vs Block Size ({label}, {cost_label} costs)')
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
















