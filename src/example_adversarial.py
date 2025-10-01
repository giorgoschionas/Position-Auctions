from budgeted_bipartite_matching_algs import create_edge_list, budgeted_bipartite_matching_solver, greedy_matching
import numpy as np

# Adversarial Example: Pushing Ratio Higher Than 2.33
# -----------------------------------------------------
# Key insight from example.py: Decoy should consume ~75% of budget
# To push higher: Use multiple decoys, each consuming large fraction of their segment

# Strategy:
# - Multiple cheap giants (high value, tiny cost) for DIFFERENT positions
# - Multiple decoys (slightly higher value, huge cost ~75% of budget/k)
# - Fillers to pad the solution
# - Budget tight enough that decoys block fillers

# 8 left nodes, 5 right nodes
vals = np.array([
    [6, 0, 0, 0, 0],        # u1: cheap giant for v1
    [6.012, 0, 0, 0, 0],    # u2: decoy for v1 (slightly better)
    [0, 6, 0, 0, 0],        # u3: cheap giant for v2
    [0, 6.012, 0, 0, 0],    # u4: decoy for v2
    [1, 1, 1, 1, 1],        # u5-u8: fillers
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0.999, 0.999, 0.999, 0.999, 0.999]
])

costs = [
    0.001,   # u1: cheap (density = 6/0.001 = 6000)
    6.002,   # u2: decoy (density = 6.012/6.002 ≈ 1.0017), uses 75% of 8
    0.001,   # u3: cheap (density = 6000)
    6.002,   # u4: decoy (density ≈ 1.0017)
    1.0,     # u5-u7: fillers (density = 1)
    1.0,
    1.0,
    0.998    # u8: slightly cheaper filler (density ≈ 1.001)
]

Budget = 8.0  # 2 decoys use 12.004 (too much), OR 2 cheap + many fillers

# Create edge list representation
edges = create_edge_list(vals, costs)

# Run greedy matching algorithm
greedy_result, greedy_total_value = greedy_matching(edges=edges, Budget=Budget)

# Run optimal solver
optimal_result, optimal_total_value = budgeted_bipartite_matching_solver(edges=edges, Budget=Budget)

# Display results
print("="*70)
print("ADVERSARIAL EXAMPLE: Two Decoy Traps")
print("="*70)
print(f"Budget: {Budget}")
print(f"\nGreedy matching total value: {greedy_total_value:.3f}")
print(f"Greedy matching cost: {sum(e[3] for e in greedy_result):.3f}")
print(f"Greedy matching edges ({len(greedy_result)} edges):")
for edge in sorted(greedy_result, key=lambda e: -e[2]):
    print(f"  {edge[0]} -> {edge[1]}: value={edge[2]:.3f}, cost={edge[3]:.3f}")

print(f"\nOptimal matching total value: {optimal_total_value:.3f}")
print(f"Optimal matching cost: {sum(costs[int(e[0][1:])-1] for e in optimal_result):.3f}")
print(f"Optimal matching edges ({len(optimal_result)} edges):")
for edge in sorted(optimal_result):
    print(f"  {edge}")

approx_ratio = optimal_total_value/greedy_total_value if greedy_total_value > 0 else float('inf')
print(f"\n{'='*70}")
print(f"APPROXIMATION RATIO (OPT/GREEDY): {approx_ratio:.4f}")
print(f"{'='*70}")

# Analysis
print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("Expected Greedy (if replacement trap works):")
print("  • u1→v1 (value=6, cost=0.001), budget: 7.999")
print("  • u3→v2 (value=6, cost=0.001), budget: 7.998")
print("  • u5→v3, u6→v4, u7→v5 (fillers, value=3, cost=3), budget: 4.998")
print("  • u8→... (maybe 1 more)")
print("  • u2 tries to replace u1: u2.value (6.012) > u1.value (6) ✓")
print("    u2.cost (6.002) ≤ budget + u1.cost (4.998 + 0.001) = 4.999 ✗")
print("  • NO REPLACEMENT - budget too tight!")
print()
print("This won't work either. The budget needs to be higher to allow replacement")
print("but low enough that greedy can't fit many fillers after decoys.")
print()
print("The original example.py might already be near-optimal structure!")
print("="*70)
