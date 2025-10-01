from budgeted_bipartite_matching_algs import create_edge_list, budgeted_bipartite_matching_solver, greedy_matching
import numpy as np

# Example: Budgeted Bipartite Matching
# ------------------------------------
# 6 left nodes (transactions/bidders) want to match with 5 right nodes (positions)
# Each edge has: value (weight), cost, and density (value/cost ratio)

# Valuations matrix (6 left nodes x 5 right nodes)
# Row i: valuations from left node u(i+1) to all right nodes
vals = np.array([[3,0,0,0,0],       # u1: high value (3) only for v1
                 [3.006,0,0,0,0],   # u2: slightly higher value (3.006) for v1, but much higher cost
                 [1,1,1,1,1],       # u3-u6: uniform value (1) for all positions
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1]])

# Costs for each left node (all edges from same left node have same cost)
costs = [0.001,   # u1: very cheap (density for v1 = 3/0.001 = 3000)
         3.002,   # u2: expensive (density for v1 = 3.006/3.002 ≈ 1.001)
         1,       # u3-u5: moderate cost (density = 1/1 = 1)
         1,
         1,
         0.999]   # u6: slightly cheaper (density = 1/0.999 ≈ 1.001)

# Create edge list representation
edges = create_edge_list(vals, costs)

# Print all edges
print("All edges (u, v, weight, cost, density):")
for edge in edges:
    print(edge)
print()

# Run greedy matching algorithm (sorts by density, uses backtracking for replacements)
greedy_matching, greedy_total_value = greedy_matching(edges=edges, Budget=4)

# Run optimal solver (Gurobi MIP)
optimal_matching, optimal_total_value = budgeted_bipartite_matching_solver(edges=edges, Budget=4)

# Display results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Greedy matching total value: {greedy_total_value}")
print(f"Greedy matching edges:")
for edge in greedy_matching:
    print(f"  {edge}")
print(f"\nOptimal matching total value: {optimal_total_value}")
print(f"Optimal matching edges:")
for edge in optimal_matching:
    print(f"  {edge}")
print(f"Approximation ratio (OPT/GREEDY): {optimal_total_value/greedy_total_value if greedy_total_value > 0 else 'N/A'}")