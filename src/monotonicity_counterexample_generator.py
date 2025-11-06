"""
Generalized counterexample generator for monotonicity violations.

This module provides tools to:
1. Generate parameterized families of counterexamples
2. Systematically search for violations with different structures
3. Characterize the space of monotonicity violations
"""

import numpy as np
import itertools
from budgeted_bipartite_matching_algs import create_edge_list, greedy_matching
from monotonicity_test import test_monotonicity_single_instance


class CounterexampleGenerator:
    """
    Generator for parameterized families of monotonicity counterexamples.
    """

    def __init__(self, epsilon=0.001):
        """
        Parameters:
        -----------
        epsilon : float
            Small perturbation parameter for creating tight competition
        """
        self.epsilon = epsilon

    def generate_base_example(self, N=4, M=3, Budget=50):
        """
        Generate the base counterexample from test_specific_scenario.

        Returns:
        --------
        dict with keys: ctrs, bids, costs, Budget, target_agent
        """
        epsilon = self.epsilon

        ctrs = np.array([
            [1.0, 0, 0],
            [0, 1, 1-epsilon],
            [0, 1, 0],
            [0, 1, 0]
        ])

        bids = np.array([2, 1, 1+epsilon, 1+3*epsilon])
        costs = np.array([1, 1, 1+10*epsilon, 2])

        return {
            'ctrs': ctrs,
            'bids': bids,
            'costs': costs,
            'Budget': Budget,
            'target_agent': 3,  # Agent u4 (0-indexed)
            'N': N,
            'M': M
        }

    def generate_scaled_example(self, N=6, M=4, Budget=100):
        """
        Generate a scaled-up version with more agents and positions.

        Structure:
        - First agent: monopoly on position 1
        - Next M-1 agents: compete for positions 2..M with slight preferences
        - Remaining agents: compete for middle positions
        - Last agent (target): vulnerable due to high cost
        """
        epsilon = self.epsilon
        ctrs = np.zeros((N, M))

        # Agent 1: monopoly on position 1
        ctrs[0, 0] = 1.0

        # Agents 2 to M: compete for positions 2..M
        for i in range(1, M):
            ctrs[i, 1:] = 1.0
            if i < M-1:
                ctrs[i, i] += epsilon  # Slight preference for own position

        # Remaining agents (including target): compete for middle positions
        for i in range(M, N):
            ctrs[i, 1:M-1] = 1.0

        # Bids: ε-separated values creating critical orderings
        bids = np.array([2.0] + [1.0 + i*epsilon for i in range(N-1)])

        # Costs: target agent (last) has highest cost
        costs = np.ones(N)
        costs[-1] = 2.0  # Target agent is vulnerable
        costs[M-1] = 1 + 10*epsilon

        return {
            'ctrs': ctrs,
            'bids': bids,
            'costs': costs,
            'Budget': Budget,
            'target_agent': N-1,
            'N': N,
            'M': M
        }

    def generate_minimal_example(self):
        """
        Try to find minimal counterexample (smallest N, M).

        Returns:
        --------
        dict or None if no violation found
        """
        # Try N=3, M=2
        for structure_type in ['type1', 'type2', 'type3']:
            instance = self._try_minimal_structure(N=3, M=2, structure_type=structure_type)
            if instance and self._check_violation(instance):
                return instance

        # Try N=4, M=2
        for structure_type in ['type1', 'type2']:
            instance = self._try_minimal_structure(N=4, M=2, structure_type=structure_type)
            if instance and self._check_violation(instance):
                return instance

        return None

    def _try_minimal_structure(self, N, M, structure_type):
        """Generate different CTR structures for minimal examples."""
        epsilon = self.epsilon

        if structure_type == 'type1':
            # Structure: competition for position 2
            ctrs = np.zeros((N, M))
            ctrs[0, 0] = 1.0
            ctrs[1:, 1] = 1.0

        elif structure_type == 'type2':
            # Structure: mixed competition
            ctrs = np.zeros((N, M))
            ctrs[0, 0] = 1.0
            ctrs[1, :] = 1.0
            ctrs[2:, 1] = 1.0

        elif structure_type == 'type3':
            # Structure: graduated preferences
            ctrs = np.zeros((N, M))
            ctrs[0, 0] = 1.0
            ctrs[1, 1] = 1.0
            ctrs[2:, 0] = 1-epsilon
            ctrs[2:, 1] = 1.0
        else:
            return None

        bids = np.array([2.0] + [1.0 + i*epsilon for i in range(N-1)])
        costs = np.ones(N)
        costs[-1] = 2.0

        # Try different budgets
        for budget_factor in [1.5, 2.0, 2.5, 3.0]:
            Budget = budget_factor * costs[-1]

            instance = {
                'ctrs': ctrs,
                'bids': bids,
                'costs': costs,
                'Budget': Budget,
                'target_agent': N-1,
                'N': N,
                'M': M
            }

            if self._check_violation(instance):
                return instance

        return None

    def _check_violation(self, instance):
        """Check if instance violates monotonicity for target agent."""
        result = test_monotonicity_single_instance(
            ctrs=instance['ctrs'],
            bids=instance['bids'],
            costs=instance['costs'],
            Budget=instance['Budget'],
            test_agent_idx=instance['target_agent'],
            bid_increase_factor=self.epsilon  # Use epsilon directly, not 1+epsilon
        )
        return not result['is_monotone']

    def search_parameter_space(self, N_range=(3, 6), M_range=(2, 4),
                                budget_factors=None, num_trials=100):
        """
        Systematic search over parameter space.

        Parameters:
        -----------
        N_range : tuple
            (min_N, max_N) range of agent counts
        M_range : tuple
            (min_M, max_M) range of position counts
        budget_factors : list or None
            Budget multipliers to try (default: [1.5, 2.0, 2.5, 3.0])
        num_trials : int
            Number of random instances per configuration

        Returns:
        --------
        dict with violations found
        """
        if budget_factors is None:
            budget_factors = [1.5, 2.0, 2.5, 3.0, 5.0]

        violations = []

        for N in range(N_range[0], N_range[1] + 1):
            for M in range(M_range[0], M_range[1] + 1):
                if M >= N:
                    continue  # Skip cases where positions >= agents

                for budget_factor in budget_factors:
                    # Try structured examples
                    instance = self.generate_scaled_example(N, M, budget_factor * N)
                    if self._check_violation(instance):
                        violations.append({
                            'type': 'structured',
                            'instance': instance
                        })

                    # Try random variations
                    for trial in range(num_trials // len(budget_factors)):
                        random_instance = self._generate_random_variation(N, M, budget_factor)
                        if random_instance and self._check_violation(random_instance):
                            violations.append({
                                'type': 'random',
                                'instance': random_instance
                            })

        return {
            'num_violations': len(violations),
            'violations': violations,
            'search_space': {
                'N_range': N_range,
                'M_range': M_range,
                'budget_factors': budget_factors,
                'num_trials': num_trials
            }
        }

    def _generate_random_variation(self, N, M, budget_factor):
        """Generate random variation around structured example."""
        epsilon = self.epsilon

        # Random CTRs with some structure
        ctrs = np.random.uniform(0, 1, (N, M))

        # Force some zero entries to create competition
        for i in range(N):
            num_zeros = np.random.randint(0, M//2)
            zero_positions = np.random.choice(M, num_zeros, replace=False)
            ctrs[i, zero_positions] = 0

        # Random bids with ε-separation
        bids = np.sort(np.random.uniform(1, 3, N))

        # Random costs with target agent having higher cost
        costs = np.random.uniform(0.5, 2, N)
        target_agent = np.random.randint(0, N)
        costs[target_agent] = max(costs) * 1.5

        Budget = budget_factor * np.sum(costs) / N

        return {
            'ctrs': ctrs,
            'bids': bids,
            'costs': costs,
            'Budget': Budget,
            'target_agent': target_agent,
            'N': N,
            'M': M
        }

    def analyze_violation_characteristics(self, instance):
        """
        Detailed analysis of why a violation occurs.

        Returns:
        --------
        dict with analysis details
        """
        ctrs = instance['ctrs']
        bids = instance['bids']
        costs = instance['costs']
        Budget = instance['Budget']
        target_idx = instance['target_agent']

        # Get matchings before and after
        valuations_before = ctrs * bids.reshape(-1, 1)
        edges_before = create_edge_list(valuations_before, costs)
        matching_before, value_before = greedy_matching(edges_before, Budget)

        bids_after = bids.copy()
        bids_after[target_idx] *= (1 + self.epsilon)

        valuations_after = ctrs * bids_after.reshape(-1, 1)
        edges_after = create_edge_list(valuations_after, costs)
        matching_after, value_after = greedy_matching(edges_after, Budget)

        # Compute densities
        densities_before = valuations_before / costs.reshape(-1, 1)
        densities_after = valuations_after / costs.reshape(-1, 1)

        # Find what changed
        agents_before = {m[0] for m in matching_before}
        agents_after = {m[0] for m in matching_after}

        new_agents = agents_after - agents_before
        removed_agents = agents_before - agents_after

        return {
            'target_agent': f"u{target_idx+1}",
            'matching_before': matching_before,
            'matching_after': matching_after,
            'value_before': value_before,
            'value_after': value_after,
            'agents_added': new_agents,
            'agents_removed': removed_agents,
            'density_change': densities_after[target_idx] - densities_before[target_idx],
            'budget_before': Budget - sum(m[3] for m in matching_before),
            'budget_after': Budget - sum(m[3] for m in matching_after)
        }


def demonstrate_generalization():
    """
    Demonstrate different generalizations of the counterexample.
    """
    print("="*70)
    print("MONOTONICITY COUNTEREXAMPLE GENERALIZATION")
    print("="*70)

    gen = CounterexampleGenerator(epsilon=0.001)

    # 1. Base example
    print("\n1. BASE EXAMPLE (N=4, M=3)")
    print("-"*70)
    base = gen.generate_base_example()
    result = test_monotonicity_single_instance(
        base['ctrs'], base['bids'], base['costs'], base['Budget'],
        base['target_agent'], bid_increase_factor=1 + gen.epsilon
    )
    print(f"Violates monotonicity: {not result['is_monotone']}")
    if not result['is_monotone']:
        print(result['violation_details'])

    # 2. Scaled example
    print("\n2. SCALED EXAMPLE (N=6, M=4)")
    print("-"*70)
    scaled = gen.generate_scaled_example(N=6, M=4, Budget=100)
    result = test_monotonicity_single_instance(
        scaled['ctrs'], scaled['bids'], scaled['costs'], scaled['Budget'],
        scaled['target_agent'], bid_increase_factor=1 + gen.epsilon
    )
    print(f"Violates monotonicity: {not result['is_monotone']}")
    if not result['is_monotone']:
        print(result['violation_details'])

    # 3. Search for minimal
    print("\n3. SEARCHING FOR MINIMAL EXAMPLE (N=3, M=2)")
    print("-"*70)
    minimal = gen.generate_minimal_example()
    if minimal:
        print(f"Found minimal example: N={minimal['N']}, M={minimal['M']}")
        result = test_monotonicity_single_instance(
            minimal['ctrs'], minimal['bids'], minimal['costs'], minimal['Budget'],
            minimal['target_agent'], bid_increase_factor=1 + gen.epsilon
        )
        print(result['violation_details'])
    else:
        print("No minimal example found with current search")

    # 4. Parameter space search
    print("\n4. PARAMETER SPACE SEARCH")
    print("-"*70)
    print("Searching N=[3,5], M=[2,3], various budgets...")
    results = gen.search_parameter_space(
        N_range=(3, 5),
        M_range=(2, 3),
        num_trials=20
    )
    print(f"Found {results['num_violations']} violations")

    if results['num_violations'] > 0:
        print(f"\nFirst few violations:")
        for i, v in enumerate(results['violations'][:3]):
            inst = v['instance']
            print(f"  {i+1}. N={inst['N']}, M={inst['M']}, "
                  f"Budget={inst['Budget']:.2f}, Type={v['type']}")


if __name__ == "__main__":
    demonstrate_generalization()
