import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from itertools import chain, combinations
from sortedcontainers import SortedList, SortedSet, SortedDict

def draw_bipartite_graph(G, U, V):
    # Compute a bipartite layout
    pos = nx.bipartite_layout(G, U)

    # Draw
    plt.figure(figsize=(6,4))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=['skyblue' if n in U else 'lightgreen' for n in G],
        edge_color='gray',
        node_size=600
    )
    # (optional) draw edge‐labels to show weights
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=nx.get_edge_attributes(G, 'weight'),
        font_color='red'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("bipartite_graph.png")
    # plt.show()

def generate_valuations(rows, cols, low, high):
    """
    Generates a matrix of a given size with elements drawn from a uniform distribution.
    
    Args:
        rows (int): The number of transactions (rows).
        cols (int): The number of positions (columns).
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        
    Returns:
        numpy.ndarray: A 2D array of shape (rows, cols).
    """
    return np.random.uniform(low=low, high=high, size=(rows, cols))


def create_bipartite_graph(valuations, budgeted=False):
    G = nx.Graph()
    U = ["left " + str(i+1) for i in range(valuations.shape[0])]  # “left” side (bidders)
    V = ["right " + str(i+1) for i in range(valuations.shape[1])]  # “right” side (positions)    

    # add nodes
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    # add edges 
    for i in range(valuations.shape[0]): 
        # special case of budgeted matching: each edge from the same node on the left has the same cost
        cost = random.uniform(0.01, 0.1) if budgeted else 0
        for j in range(valuations.shape[1]):
                if valuations[i, j] > 0:
                    G.add_edge(U[i], V[j], weight=valuations[i, j], cost=cost)
      
    print(G[U[0]][V[1]]['weight'])

    
    draw_bipartite_graph(G, U, V)
    maximum_matching = nx.max_weight_matching(G)
    print("value of the maximum matching: ", sum(G[u][v]['weight'] for u, v in maximum_matching))

    return G

# takes as input a bipartite graph G and returns a matching by greedily selecting edges
# based on the edge weights, ensuring no two edges share a node.
def greedy_matching(G, weight='weight'):
    bidders, positions = nx.bipartite.sets(G)


    edges=sorted(G.edges(data=True), key=lambda edge: edge[2].get('weight', 1), reverse=True)
    greedy_matching = set()
    used = set()
    for u, v, attrs in edges:
        if u not in used and v not in used:
            greedy_matching.add((u, v))
            used.update([u, v])
    print("value of greedy matching: ",  sum(G[u][v]['weight'] for u, v in greedy_matching))
    return greedy_matching

def greedy_matching_bipartite(G,Budget):
    left, right = nx.bipartite.sets(G)
    local_budgets = [0]*len(right)
    edges = sorted(G.edges(data=True), key=lambda edge: edge[2].get('density', 0), reverse=True)
    greedy_budgeted_matching = set()
    used = set()
    for u, v, attrs in edges:
        if u not in used:
            if v not in used:
                # SOS check this condition
                if attrs['cost'] + sum(local_budgets) <= Budget: 
                    greedy_budgeted_matching.add((u, v))
                    used.update([u, v])
                    local_budgets[list(right).index(v)] = attrs['cost']
                    Budget -= attrs['cost']
            else:
                if attrs['cost'] + local_budgets[list(right).index(v)] <= Budget:
                    current = next(t for t in greedy_budgeted_matching if t[0] == u)
                    if G[current[0]][current[1]]['weight'] < G[u][v]['weight']:
                        greedy_budgeted_matching.remove(current)
                        greedy_budgeted_matching.add((u, v))
                        local_budgets[list(right).index(v)] = attrs['cost']
                        Budget -= attrs['cost'] + G[current[0]][current[1]]['cost']
    print("value of greedy budgeted matching: ",  sum(G[u][v]['weight'] for u, v in greedy_budgeted_matching))

# Example
num_txs = 10
num_positions = 5
custom_valuations = np.array([[101, 100], [100, 0]])
G= create_bipartite_graph(valuations=custom_valuations, budgeted=False)

# G= create_bipartite_graph(valuations=generate_valuations(num_txs, num_positions, 0, 10))
greedy_matching(G)
