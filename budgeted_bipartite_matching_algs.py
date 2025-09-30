import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import gurobipy as gp
from gurobipy import GRB
import random



def greedy_matching_simple(edges, Budget):
    """
    Greedy algorithm for budgeted bipartite matching with backtracking.

    Uses simple edge list representation (no NetworkX) for maximum efficiency.

    Selects edges by density (value/cost ratio) in descending order, respecting:
    - Budget constraint: total cost of selected edges ≤ Budget
    - Matching constraint: each node matched at most once
    - Replacement rule: can replace a matched edge if new edge has higher value and fits budget
    - Backtracking: when a left node is freed, reconsider previously skipped edges

    Parameters:
    -----------
    edges : list of tuples
        Each tuple is (u, v, weight, cost, density)
    Budget : float
        Total budget constraint

    Returns:
    --------
    tuple: (matching, total_value)
        - matching: list of tuples (u, v, weight, cost, density) in the greedy matching
        - total_value: sum of weights in the matching
    """
    # Sort edges by density (value/cost ratio) in descending order
    edges_sorted = sorted(edges, key=lambda e: e[4], reverse=True)  # e[4] is density

    # Build edge lookup dictionary for quick access to edge attributes
    edge_dict = {(e[0], e[1]): e for e in edges_sorted}

    matching = []
    used_left = set()  # Left nodes already matched
    matched_to_right = {}  # Maps right node v -> (u, weight, cost) currently matched

    # Build edge index map for backtracking
    edge_index = {(e[0], e[1]): idx for idx, e in enumerate(edges_sorted)}

    i = 0
    while i < len(edges_sorted):
        u, v, weight, cost, density = edges_sorted[i]

        if u not in used_left:  # Only consider if left node u is unmatched

            if v not in matched_to_right:  # Case 1: Right node v is free
                if cost <= Budget:
                    matching.append((u, v, weight, cost, density))
                    used_left.add(u)
                    matched_to_right[v] = (u, weight, cost)
                    Budget -= cost

            else:  # Case 2: Right node v is already matched
                current_u, current_weight, current_cost = matched_to_right[v]
                current_edge = (current_u, v)

                # Replace current edge if:
                # 1. New edge has higher value
                # 2. New edge fits in budget after freeing current edge's cost
                if (current_weight < weight) and (cost <= Budget + current_cost):

                    # Remove old edge from matching
                    matching = [m for m in matching if not (m[0] == current_u and m[1] == v)]

                    # Free up current_u and update budget
                    used_left.remove(current_u)
                    Budget = Budget + current_cost

                    # Add new edge
                    matching.append((u, v, weight, cost, density))
                    used_left.add(u)
                    matched_to_right[v] = (u, weight, cost)
                    Budget = Budget - cost

                    # Backtrack: restart from where current_u was added
                    if current_edge in edge_index:
                        backtrack_idx = edge_index[current_edge]
                        i = backtrack_idx  # Restart from that position
                        continue

        i += 1

    # Calculate total value
    total_value = sum(e[2] for e in matching)  # e[2] is weight

    return matching, total_value


def budgeted_bipartite_matching_solver(edges, Budget):
    """
    Solve budgeted bipartite matching using Gurobi (optimal solution).

    Parameters:
    -----------
    edges : list of tuples
        Each tuple is (u, v, weight, cost, density)
    Budget : float
        Total budget constraint

    Returns:
    --------
    float or None
        Optimal objective value (total weight), or None if infeasible
    """
    # Extract unique left and right nodes
    U = set(e[0] for e in edges)
    V = set(e[1] for e in edges)

    # Build value and cost dictionaries
    value = {(e[0], e[1]): e[2] for e in edges}  # e[2] is weight
    cost = {(e[0], e[1]): e[3] for e in edges}   # e[3] is cost

    # --- Create model ---
    m = gp.Model("budgeted_bipartite_matching")
    m.setParam('OutputFlag', 0)

    # Binary decision x[i,j] == 1 if we match i–j
    x = m.addVars(value.keys(), vtype=GRB.BINARY, name="x")

    # --- Objective: maximize total value ---
    m.setObjective(
        gp.quicksum(value[i,j] * x[i,j] for i,j in x.keys()),
        GRB.MAXIMIZE
    )

    # --- Budget constraint ---
    m.addConstr(
        gp.quicksum(cost[i,j] * x[i,j] for i,j in x.keys()) <= Budget,
        name="budget"
    )

    # --- Matching constraints ---
    # Each u in U matched at most once
    for i in U:
        m.addConstr(
            gp.quicksum(x[i,j] for j in V if (i,j) in x) <= 1,
            name=f"match_left_{i}"
        )
    # Each v in V matched at most once
    for j in V:
        m.addConstr(
            gp.quicksum(x[i,j] for i in U if (i,j) in x) <= 1,
            name=f"match_right_{j}"
        )

    # --- Solve and retrieve solution ---
    m.optimize()

    if m.status == GRB.OPTIMAL:
        sol = [(i,j) for i,j in x.keys() if x[i,j].X > 0.5]
        # print("Optimal matching:", sol)
        # print("Total value:", m.ObjVal)
        # print("Total cost:", sum(cost[i,j] for (i,j) in sol))
        return m.ObjVal
    else:
        # print("No optimal solution found. Status code:", m.status)
        return None



U = [f"u{i}" for i in range(1, 51)]    # u1, u2, …, u50
V = [f"v{j}" for j in range(1, 11)]

# 2. Build a list of candidate edges with random weights/costs
#    Here we include all possible I×J edges; you can sparsify if you like.
random.seed(48)
edge_list = []
for u in U:
    for v in V:
        val  = random.uniform(1, 10)     # value in [1,10)
        cst  = random.uniform(1, 5)      # cost in [1,5)
        edge_list.append((u, v, {"value": val, "cost": cst}))

# 3. Convert to value and cost dicts
value = { (u,v): attrs["value"] for u, v, attrs in edge_list }
cost  = { (u,v): attrs["cost"]  for u, v, attrs in edge_list }

# 4. Set a budget
B = 0.3 * sum(cost.values())  # e.g. 30% of total possible cos



opt = budgeted_bipartite_matching_solver(U, V, value, cost, B)

print(f"Optimal value: {opt}")




