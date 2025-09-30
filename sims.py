import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


import math
import random

from utils_simple_graphs import greedy_matching_simple, budgeted_bipartite_matching_solver


def create_edge_list(valuations, costs):
    """
    Create a simple edge list representation for a bipartite graph with weights(values) and costs.

    This is more efficient than NetworkX Graph for the greedy matching algorithm.
    Each edge is represented as a tuple: (u, v, weight, cost, density)

    Parameters:
    -----------
    valuations : numpy.ndarray
        2D array of shape (n_left, n_right) containing edge values
    costs : numpy.ndarray or list
        1D array of length n_left containing costs for each left node
        (all edges from the same left node have the same cost)

    Returns:
    --------
    list of tuples
        Each tuple is (u, v, weight, cost, density) where:
        - u: left node identifier (e.g., "u1")
        - v: right node identifier (e.g., "v1")
        - weight: edge value
        - cost: edge cost
        - density: weight/cost ratio
    """
    edges = []
    n_left, n_right = valuations.shape

    for i, cost in enumerate(costs):
        u = f"u{i+1}"
        for j in range(n_right):
            weight = valuations[i, j]
            if weight > 0:
                density = weight / cost
                v = f"v{j+1}"
                edges.append((u, v, weight, cost, density))

    return edges



def simulate_block(dist, tx_size_dists, N, B, T):
    opts = np.zeros(T)
    sols = np.zeros(T)
    bounds = np.zeros(T)
    approxs = np.zeros(T)
    for t in range(T):
        valuations = dist.rvs(size=(N, num_positions))
        costs = tx_size_dists.rvs(N)
        max_cost = np.max(costs)

        # Create edge list (more efficient than NetworkX graph)
        edges = create_edge_list(valuations=valuations, costs=costs)

        # Run greedy matching algorithm
        sol_matching, sol_value = greedy_matching_simple(edges, Budget=B)
        sols[t] = sol_value

        # Calculate the bound: SOL >= (1/2 - max_cost/B) * OPT
        bounds[t] = (1/(1/2 - max_cost/B)) * sols[t]

        # Calculate the optimal solution using the budgeted matching solver
        opt = budgeted_bipartite_matching_solver(edges=edges, Budget=B)
        opts[t] = opt if opt is not None else 0

        approxs[t] = opts[t] / sols[t] if sols[t] > 0 else 0

    return sols, bounds, opts, approxs


# Constants of the setting
B_plus = 15
B_minus = 0.1
N = 150
num_positions = 15
# Bs: block size
Bs = np.arange(70, 500, 25)
# Number of simulations
T= 20

tx_size_dists = stats.uniform(loc=B_minus, scale=B_plus-B_minus)
dists = [stats.expon(scale=2.5), stats.lognorm(s=1, scale=np.exp(1)), stats.rayleigh(scale=1)]
dist_labels = ["Exponential", "LogNormal", "Rayleigh"]



# Light Distributions
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


sols = np.zeros((T, len(Bs), len(dists)))
bounds = np.zeros((T, len(Bs), len(dists)))
opts = np.zeros((T, len(Bs), len(dists)))
approxs = np.zeros((T, len(Bs), len(dists)))


for i, dist in enumerate(dists):
    for j, B in enumerate(Bs):
        ret = simulate_block(dist, tx_size_dists, N, B, T)
        sols[:, j, i] = ret[0]
        bounds[:, j, i] = ret[1]
        opts[:, j, i] = ret[2]
        approxs[:, j, i] = ret[3]

means = np.mean(sols, axis=0)
meds = np.median(sols, axis=0)
stds = np.std(sols, axis=0)
bounds = np.mean(bounds, axis=0)
opts = np.mean(opts, axis=0)
approxs_means = np.mean(approxs, axis=0)
approxs_std = np.std(approxs, axis=0)

# plot results
colors = ['coral', 'indigo', 'firebrick', 'mediumblue']

# Main Loop
for i, label in zip(range(len(dists)), dist_labels):
    bound = bounds[:, i]
    plt.figure(figsize=(8, 6))
    plt.plot(Bs, means[:, i], lw=2, label='Mean', color=colors[i])
    # plt.plot(Bs, opts[:, i], lw=2, label='Optimal', linestyle='--', color='black')
    plt.fill_between(Bs, means[:, i] - stds[:, i], means[:, i] + stds[:, i], alpha=0.5, color=colors[i])

    plt.plot(Bs, bound, lw=2, label='Bound', linestyle='--', color='black')

    # Set plot attributes
    plt.yscale('log')
    plt.xlabel('Block Size (B)')
    plt.ylabel('Social Welfare')
    plt.title(f'Social Welfare vs Block Size ({label})')
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig(f"SW-{dist_labels[i]}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


for i, label in zip(range(len(dists)), dist_labels):
    bound = bounds[:, i]
    plt.figure(figsize=(8, 6))
    plt.plot(Bs, approxs_means[:, i], lw=2, label='Approximation ratio', color=colors[i])
    plt.fill_between(Bs, approxs_means[:, i] - approxs_std[:, i], approxs_means[:, i] + approxs_std[:, i], alpha=0.5, color=colors[i])


    # plt.plot(Bs, bound, lw=2, label='Bound', linestyle='--', color='black')

    # Set plot attributes
    plt.xlabel('Block Size (B)')
    plt.ylabel('OPT/SOL')
    plt.title(f'OPT/SOL vs Block Size ({label})')
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig(f"Approx-{dist_labels[i]}.pdf", dpi=300, bbox_inches='tight')
    plt.show()
















