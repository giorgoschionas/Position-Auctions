import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


import math
import random

from utils_simple_graphs import create_bipartite_graph, greedy_matching_bipartite


class Transaction:
    def __init__(self, id, size, vals):
        self.id =id
        self.size = size
        self.vals = vals
        self.densities = [val/ size for val in vals]


def simulate_block(dist, tx_size_dists, N, B, T):
    sols = np.zeros(T)
    bounds = np.zeros(T)
    for t in range(T):
        valuations = dist.rvs(size=(N, num_positions))
        costs = tx_size_dists.rvs(N)
        max_cost = np.max(costs)
        G = create_bipartite_graph(valuations= valuations, costs=costs, budgeted=True)
        sol_matching = greedy_matching_bipartite(G, Budget=B)
        # Calculate the social welfare of the solution
        sols[t] = sum(G[u][v]['weight'] for u, v in sol_matching)
        # Calculate the bound: SOL >= (1/2 - max_cost/B) * OPT
        bounds[t] = (1/(1/2 - max_cost/B)) * sols[t]

    return sols, bounds


# Constants of the setting
B_plus = 10
B_minus = 0.1
N = 100
num_positions = 15
# Bs: block size
Bs = np.arange(20, 200, 5)
# Number of simulations
T= 100

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


for i, dist in enumerate(dists):
    for j, B in enumerate(Bs):
        ret = simulate_block(dist, tx_size_dists, N, B, T)
        sols[:, j, i] = ret[0]
        bounds[:, j, i] = ret[1]


means = np.mean(sols, axis=0)
meds = np.median(sols, axis=0)
stds = np.std(sols, axis=0)
bounds = np.mean(bounds, axis=0)

# plot results
colors = ['coral', 'indigo', 'firebrick', 'mediumblue']

# Main Loop
for i, label in zip(range(len(dists)), dist_labels):
    bound = bounds[:, i]
    plt.figure(figsize=(8, 6))
    plt.plot(Bs, means[:, i], lw=2, label='Mean', color=colors[i])
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













