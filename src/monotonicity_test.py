"""
Test monotonicity of the greedy matching algorithm.

In mechanism design, an allocation rule is monotone if:
When agent i increases their bid (keeping all other bids fixed),
the quality of their allocation does not decrease.

For position auctions with CTRs:
- Valuation[i,j] = CTR[i,j] * bid[i]
- Quality of allocation = CTR[i,j] if matched to position j, else 0
- Monotonicity: increasing bid[i] should not decrease the CTR of agent i's allocated position
"""

import numpy as np
import scipy.stats as stats
from budgeted_bipartite_matching_algs import create_edge_list, greedy_matching


def get_agent_allocation(matching, agent_id):
    """
    Extract which position an agent was matched to.

    Parameters:
    -----------
    matching : list of tuples
        Each tuple is (u, v, weight, cost, density)
    agent_id : str
        Agent identifier (e.g., "u1")

    Returns:
    --------
    tuple or None
        (position_id, valuation, cost) if matched, None otherwise
    """
    for u, v, weight, cost, density in matching:
        if u == agent_id:
            return (v, weight, cost)
    return None


def get_agent_ctr(ctrs, agent_idx, position_idx):
    """Get CTR for agent at position."""
    return ctrs[agent_idx, position_idx]


def test_monotonicity_single_instance(ctrs, bids, costs, Budget, test_agent_idx, bid_increase_factor=1.2):
    """
    Test monotonicity for a single agent on a single instance.

    Parameters:
    -----------
    ctrs : np.ndarray
        CTR matrix (N x M)
    bids : np.ndarray
        Bid vector (N,)
    costs : np.ndarray
        Cost vector (N,)
    Budget : float
        Budget constraint
    test_agent_idx : int
        Index of agent to test (0-indexed)
    bid_increase_factor : float
        Factor to increase bid by (default 1.2 = 20% increase)

    Returns:
    --------
    dict
        Results containing:
        - is_monotone: True if monotonicity holds
        - old_allocation: (position, ctr) or None
        - new_allocation: (position, ctr) or None
        - violation_details: string describing violation if any
    """
    N, M = ctrs.shape
    test_agent_id = f"u{test_agent_idx + 1}"

    # Run with original bids
    valuations_old = ctrs * bids.reshape(-1, 1)
    edges_old = create_edge_list(valuations_old, costs)
    matching_old, value_old = greedy_matching(edges_old, Budget)

    # Get agent's allocation
    alloc_old = get_agent_allocation(matching_old, test_agent_id)

    if alloc_old is None:
        old_position = None
        old_ctr = 0.0
        old_valuation = 0.0
    else:
        old_position, old_valuation, _ = alloc_old
        # Extract position index from "v{j+1}" format
        old_pos_idx = int(old_position[1:]) - 1
        old_ctr = get_agent_ctr(ctrs, test_agent_idx, old_pos_idx)

    # Increase agent's bid
    bids_new = bids.copy()
    bids_new[test_agent_idx] *= bid_increase_factor

    # Run with increased bid
    valuations_new = ctrs * bids_new.reshape(-1, 1)
    edges_new = create_edge_list(valuations_new, costs)
    matching_new, value_new = greedy_matching(edges_new, Budget)

    # Get agent's new allocation
    alloc_new = get_agent_allocation(matching_new, test_agent_id)

    if alloc_new is None:
        new_position = None
        new_ctr = 0.0
        new_valuation = 0.0
    else:
        new_position, new_valuation, _ = alloc_new
        new_pos_idx = int(new_position[1:]) - 1
        new_ctr = get_agent_ctr(ctrs, test_agent_idx, new_pos_idx)

    # Check monotonicity: new CTR should be >= old CTR
    is_monotone = new_ctr >= old_ctr - 1e-9  # Small tolerance for numerical errors

    violation_details = None
    if not is_monotone:
        violation_details = (
            f"Monotonicity VIOLATED for agent {test_agent_id}:\n"
            f"  Old bid: {bids[test_agent_idx]:.3f} -> Position: {old_position}, CTR: {old_ctr:.4f}\n"
            f"  New bid: {bids_new[test_agent_idx]:.3f} -> Position: {new_position}, CTR: {new_ctr:.4f}\n"
            f"  CTR decreased by: {old_ctr - new_ctr:.4f}"
        )

    return {
        'is_monotone': is_monotone,
        'old_allocation': (old_position, old_ctr) if old_position else None,
        'new_allocation': (new_position, new_ctr) if new_position else None,
        'old_bid': bids[test_agent_idx],
        'new_bid': bids_new[test_agent_idx],
        'violation_details': violation_details
    }


def run_monotonicity_tests(N=50, M=10, Budget=100, num_trials=100, bid_increase_factor=1.2):
    """
    Run multiple monotonicity tests with random instances.

    Parameters:
    -----------
    N : int
        Number of agents
    M : int
        Number of positions
    Budget : float
        Budget constraint
    num_trials : int
        Number of random instances to test
    bid_increase_factor : float
        Factor to increase bid by

    Returns:
    --------
    dict
        Summary statistics including:
        - num_violations: number of monotonicity violations
        - violation_rate: fraction of tests that violated monotonicity
        - violations: list of violation details
    """
    print("="*70)
    print("MONOTONICITY TEST FOR GREEDY MATCHING ALGORITHM")
    print("="*70)
    print(f"Parameters:")
    print(f"  Agents: {N}")
    print(f"  Positions: {M}")
    print(f"  Budget: {Budget}")
    print(f"  Trials: {num_trials}")
    print(f"  Bid increase factor: {bid_increase_factor}")
    print("="*70)

    num_violations = 0
    violations = []

    # Use fixed CTRs across all trials for consistency
    np.random.seed(42)
    ctrs = np.abs(np.random.normal(loc=1.0, scale=0.2, size=(N, M)))

    for trial in range(num_trials):
        # Generate random bids and costs
        bids = stats.expon(scale=2.5).rvs(N)
        costs = stats.uniform(loc=0.1, scale=100).rvs(N)

        # Pick a random agent to test
        test_agent_idx = np.random.randint(0, N)

        # Test monotonicity
        result = test_monotonicity_single_instance(
            ctrs, bids, costs, Budget, test_agent_idx, bid_increase_factor
        )

        if not result['is_monotone']:
            num_violations += 1
            violations.append({
                'trial': trial,
                'agent_idx': test_agent_idx,
                'result': result
            })

            if num_violations <= 5:  # Print first 5 violations
                print(f"\n{result['violation_details']}")

    violation_rate = num_violations / num_trials

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total tests: {num_trials}")
    print(f"Violations: {num_violations}")
    print(f"Violation rate: {violation_rate*100:.2f}%")

    if num_violations == 0:
        print("\n✓ Algorithm appears to be MONOTONE (no violations found)")
    else:
        print(f"\n✗ Algorithm is NOT MONOTONE ({num_violations} violations found)")
        print(f"\nFirst few violations shown above.")

    print("="*70)

    return {
        'num_violations': num_violations,
        'violation_rate': violation_rate,
        'violations': violations,
        'num_trials': num_trials
    }


def test_specific_scenario():
    """
    Test a specific scenario to understand monotonicity behavior.
    """
    print("\n" + "="*70)
    print("SPECIFIC SCENARIO TEST")
    print("="*70)

    # Small example for debugging
    N, M = 4, 3
    Budget = 50
    epsilon=0.001

    # Fixed CTRs and costs
    ctrs = np.array([
        [1.0, 0, 0],
        [0, 1, 1-epsilon],
        [0, 1, 0],
        [0, 1, 0]
    ])

    bids = np.array([2,1,1+epsilon, 1+3*epsilon])
    costs = np.array([1,1,1+10*epsilon,2])

    print(f"\nCTRs:\n{ctrs}")
    print(f"\nBids: {bids}")
    print(f"Costs: {costs}")
    print(f"Budget: {Budget}")

    # Test each agent with different bid increase factors
    for bid_inc_factor in [1.001, 1.01, 1.1, 1.2]:
        print(f"\n{'='*70}")
        print(f"Testing with bid_increase_factor = {bid_inc_factor}")
        print(f"{'='*70}")

        for agent_idx in range(N):
            result = test_monotonicity_single_instance(
                ctrs, bids, costs, Budget, agent_idx, bid_increase_factor=bid_inc_factor
            )

            if not result['is_monotone']:
                print(f"\n✗ VIOLATION for agent u{agent_idx+1}:")
                print(f"  {result['violation_details']}")

        # Check if all passed
        violations = [test_monotonicity_single_instance(
            ctrs, bids, costs, Budget, i, bid_increase_factor=bid_inc_factor
        ) for i in range(N)]

        if all(v['is_monotone'] for v in violations):
            print(f"  ✓ All agents monotone with factor {bid_inc_factor}")


if __name__ == "__main__":
    # Run large-scale tests
    results = run_monotonicity_tests(
        N=50,
        M=10,
        Budget=100,
        num_trials=1000,
        bid_increase_factor=1.2
    )

    # Run specific scenario for understanding
    test_specific_scenario()
