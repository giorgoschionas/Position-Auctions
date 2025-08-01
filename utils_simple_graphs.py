import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


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

def generate_valuations(val_dist, rows, cols):
    """ Generate a 2D array of valuations based on a given distribution.
        
    Returns:
        numpy.ndarray: A 2D array of shape (rows, cols).
    """
    return val_dist.rvs(size=(rows, cols))


def create_bipartite_graph(valuations, costs=None, budgeted=False):
    G = nx.Graph()
    U = ["left " + str(i+1) for i in range(valuations.shape[0])]  # “left” side (bidders)
    V = ["right " + str(i+1) for i in range(valuations.shape[1])]  # “right” side (positions)    

    # add nodes
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    # add edges 
    for i, cost in zip(range(valuations.shape[0]), list(costs)): 
        # special case of budgeted matching: each edge from the same node on the left has the same cost
        for j in range(valuations.shape[1]):
                if valuations[i, j] > 0:
                    G.add_edge(U[i], V[j], weight=valuations[i, j], cost=cost, density=valuations[i, j] / cost if budgeted else 0)
                    print
      


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
    bipartite_with_locals =  set()
    used = set()
    for u, v, attrs in edges:
        if u not in used:
            if v not in used:
                # SOS check this condition
                if attrs['cost'] <= Budget: 
                    greedy_budgeted_matching.add((u, v))
                    bipartite_with_locals.add((u, v))
                    used.update([u, v])
                    local_budgets[list(right).index(v)] = attrs['cost']
                    Budget -= attrs['cost']
            else:
                # current is currently in matched in v
                current = next((t for t in greedy_budgeted_matching if t[1] == v), None)
                if (local_budgets[list(right).index(v)] >= attrs['cost']):
                    bipartite_with_locals.add((u, v))

                if (G[current[0]][current[1]]['weight'] < G[u][v]['weight']) and (attrs['cost'] <= Budget +  G[current[0]][current[1]]['cost']):
                    greedy_budgeted_matching.remove(current)
                    greedy_budgeted_matching.add((u, v))
                    bipartite_with_locals.add((u, v))
                    Budget = Budget - attrs['cost'] + G[current[0]][current[1]]['cost']
                    local_budgets[list(right).index(v)] = attrs['cost']
    # print("value of greedy budgeted matching: ",  sum(G[u][v]['weight'] for u, v in greedy_budgeted_matching))
    # print("used budget: ", sum(local_budgets))

    new_G = nx.Graph()
    for u, v in bipartite_with_locals:
        if G.has_edge(u, v):
            # Get edge attributes from G
            attr = G.get_edge_data(u, v)
            # Add edge to H with attributes
            new_G.add_edge(u, v, **attr)
    

    return greedy_budgeted_matching

# Example
num_txs = 10
num_positions = 3

dists = [stats.expon(scale=2.5), stats.lognorm(s=1, scale=np.exp(1)), stats.rayleigh(scale=1)]


G = create_bipartite_graph(valuations=generate_valuations(dists[0], num_txs, num_positions), costs=np.random.uniform(low=5, high=25, size=num_txs), budgeted=True)
greedy_matching_bipartite(G, Budget=50)




