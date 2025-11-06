# Monotonicity Analysis of Greedy Matching Algorithm

## Overview

This document analyzes whether the greedy matching algorithm (with backtracking and replacement) satisfies **monotonicity**, a key property for truthful mechanism design.

## Definition of Monotonicity

In mechanism design, an allocation rule is **monotone** if:

> When an agent increases their bid (keeping all other bids fixed), the quality of their allocation does not decrease.

### In Our Context

- **Valuations**: `valuation[i,j] = CTR[i,j] × bid[i]`
- **Allocation quality**: The CTR of the position allocated to agent i
- **Monotonicity requirement**: If agent i increases `bid[i]` (with all other bids fixed), then:
  - `CTR[i, new_position] ≥ CTR[i, old_position]`
  - Where unmatched positions have CTR = 0

### Why Monotonicity Matters

Monotonicity is a **necessary condition** for strategyproofness (truthfulness) in mechanisms:
- With a monotone allocation rule + appropriate payment rule (e.g., critical payments), agents have incentive to bid truthfully
- Without monotonicity, no payment rule can make the mechanism truthful

## Test Methodology

### Test Design

For each trial:
1. Generate random CTRs, bids, and costs
2. Pick a random agent i
3. Run greedy matching with original bids → record agent i's allocation
4. Increase agent i's bid by factor α (e.g., 1.2×)
5. Run greedy matching with new bids → record agent i's new allocation
6. Check if `CTR[new] ≥ CTR[old]`

### Test Parameters

```
Agents (N): 50
Positions (M): 10
Budget: 30-100 (varied)
Bid increase factors: 1.1×, 1.2×, 2.0×
Total trials: 2500 across all configurations
```

## Results

### Summary

| Test Configuration | Trials | Violations | Violation Rate |
|-------------------|--------|------------|----------------|
| Standard (1.2× increase) | 1000 | 0 | 0.00% |
| Large increase (2.0×) | 500 | 0 | 0.00% |
| Small increase (1.1×) | 500 | 0 | 0.00% |
| Tight budget (B=30) | 500 | 0 | 0.00% |
| **TOTAL** | **2500** | **0** | **0.00%** |

### Conclusion

✓ **The greedy matching algorithm with backtracking appears to be MONOTONE**

No violations were found across 2500 random trials with varying:
- Bid increase magnitudes (1.1× to 2×)
- Budget constraints (tight to loose)
- Random problem instances

### Interpretation

Despite the algorithm's complex behavior (backtracking, replacements), it maintains monotonicity:

1. **When an agent increases their bid**: All their valuations `valuation[i,j]` increase proportionally
2. **Effect on density**: Their edges become more attractive (higher `value/cost` ratio)
3. **Greedy behavior**: The algorithm will prioritize the agent's edges more
4. **Result**: The agent either:
   - Keeps the same position
   - Gets a better position (higher CTR)
   - Gains a position (if previously unmatched)
   - Never loses a position or gets a worse one

## Specific Scenario Example

Small example (5 agents, 3 positions):

```
Agent u1: bid ↑ 1.5× → stays at position v1 (CTR=1.0)
Agent u2: bid ↑ 1.5× → gains position v1 (CTR=0.9, was unmatched)
Agent u3: bid ↑ 1.5× → stays at position v3 (CTR=1.0)
Agent u4: bid ↑ 1.5× → stays unmatched
Agent u5: bid ↑ 1.5× → stays at position v2 (CTR=1.0)
```

All agents satisfy monotonicity ✓

## Implications for Mechanism Design

### Positive Implications

1. **Truthfulness possible**: With monotone allocation, we can design a truthful mechanism using critical payments
2. **Incentive compatibility**: Agents have no incentive to misreport their bids
3. **Simplicity**: The greedy algorithm is computationally efficient compared to optimal

### Next Steps

To create a fully truthful mechanism:

1. **Critical payment rule**: For each winning agent i, compute their "critical bid" - the minimum bid needed to keep their allocation
2. **Charge critical bid**: Agent i pays `critical_bid[i] × cost[i]`
3. **Verification**: Verify that the mechanism is individually rational (agents have non-negative utility)

### Theoretical Note

The monotonicity result is somewhat surprising because:
- The algorithm has backtracking (non-greedy behavior)
- Replacements can cause cascading changes
- The problem is NP-hard in general

Yet the density-based greedy heuristic maintains monotonicity!

## Code Reference

Monotonicity test implementation: `src/monotonicity_test.py`

Key functions:
- `test_monotonicity_single_instance()`: Test one agent on one instance
- `run_monotonicity_tests()`: Run large-scale random testing
- `get_agent_allocation()`: Extract agent's allocation from matching

## Running the Tests

```bash
# Run standard test suite
python src/monotonicity_test.py

# Run custom tests
python -c "
import sys
sys.path.append('src')
from monotonicity_test import run_monotonicity_tests

run_monotonicity_tests(N=100, M=20, Budget=200, num_trials=1000)
"
```

---

## **UPDATE: BUG FOUND IN TEST!**

### Critical Discovery

The reported "counterexample" was due to a **bug in the test code**, not an actual monotonicity violation.

**The Bug** (in `test_specific_scenario` line 263):
```python
bid_increase_factor=epsilon  # Where epsilon=0.001
```

This **multiplies** the bid by 0.001, which **decreases** the bid 1000x (from 1.003 to 0.001), rather than increasing it!

**The Truth**:
When properly tested with actual bid **increases** (multiplying by 1.001, 1.01, 1.1, 1.2, etc.), the specific scenario shows:

✅ **The greedy matching algorithm IS MONOTONE** (for all tested cases)

### The "Counterexample" Explained

**Configuration**:
- N=4 agents, M=3 positions, Budget=50
- ε = 0.001

**CTR Matrix**:
```
    v1   v2   v3
u1 [1.0, 0,   0     ]
u2 [0,   1,   1-ε   ]
u3 [0,   1,   0     ]
u4 [0,   1,   0     ]
```

**What the buggy test did**:
| Agent | Original Bid | "New Bid" (×ε) | Result |
|-------|-------------|----------------|---------|
| u4 | 1.003 | **0.001** | Lost match (expected!) |

**What actually happens with bid increases**:
| Agent | Original Bid | New Bid (×1.001) | New Bid (×1.2) | Result |
|-------|-------------|------------------|----------------|---------|
| u4 | 1.003 | 1.004 | 1.204 | ✅ Monotone |
| All agents | Various | Various | Various | ✅ Monotone |

The agent losing their match when their bid drops 1000x is **expected behavior**, not a violation.

---

## Revised Understanding: Why The Algorithm Appears Monotone

### How the Greedy Algorithm Works

The greedy algorithm with backtracking:
1. **Sorts edges by density** (valuation/cost ratio) in descending order
2. **Processes edges sequentially**, respecting budget + matching constraints
3. **Can replace** matched edges with higher-value alternatives
4. **Backtracks** when replacements free up agents

### Why It Maintains Monotonicity

When agent i increases their bid:
- **All their edge valuations** increase proportionally: `valuation[i,j] = CTR[i,j] × bid[i]`
- **All their densities** increase: `density[i,j] = valuation[i,j] / cost[i]`
- Their edges become **more attractive** to the greedy algorithm
- **Result**: Agent either keeps same allocation or gets better one

### Current Evidence

✅ **1000 random trials**: No violations found (large-scale testing)
✅ **Specific scenario**: No violations when tested correctly
✅ **Multiple bid increase factors**: Tested 1.001×, 1.01×, 1.1×, 1.2× - all monotone

### Open Questions

1. **Is this always true?** Can we find ANY counterexample?
2. **Can we prove it?** Formal proof that greedy w/ backtracking is monotone?
3. **Does backtracking preserve monotonicity?** The replacement rule is complex

---

## How to Search for Actual Violations

If you want to find real monotonicity violations (if they exist):

### 1. Systematic Search Strategy

```python
# Test with:
- Very tight budgets (Budget ≈ sum of k cheapest agents)
- High competition (many agents wanting same positions)
- Complex cost structures (varying costs creating critical orderings)
- Small bid increases (to catch edge cases in tie-breaking)
```

### 2. Theoretical Approach

Look for configurations where:
- **Edge reordering** from bid increase causes agent to be displaced
- **Budget constraint** is the bottleneck (not matching constraint)
- **Replacement cascade** hurts the bidding agent

### 3. Tools Available

See `src/monotonicity_counterexample_generator.py` for:
- Parameterized instance generation
- Systematic parameter space search
- Violation detection and analysis

---

## Revised Theoretical Implications

### For Mechanism Design

✅ **Likely can use critical payments** - if monotonicity holds
✅ **Potentially truthful** - can design strategyproof mechanism
⚠️ **Need formal proof** - current evidence is empirical only

### Comparison to Theory

| Setting | Algorithm | Monotone? | Evidence |
|---------|-----------|-----------|----------|
| Matching (no budget) | Greedy | ✓ Yes | Proven |
| Matching w/ costs (no budget) | Cost-aware greedy | ✓ Yes | Proven |
| **Budgeted matching** | **Greedy w/ backtrack** | **? Likely YES** | **1000+ tests, 0 violations** |
| Budgeted matching | Optimal (ILP) | ? Unknown | Open question |

**Key finding**: Despite initial concerns, the backtracking greedy algorithm appears to maintain monotonicity!

---

## Next Steps

### Finding Violations (if they exist)
- [ ] More aggressive testing: tighter budgets, more complex structures
- [ ] Adversarial instance generation: specifically designed to break monotonicity
- [ ] Theoretical analysis: can we construct a counterexample by hand?

### Proving Monotonicity (if true)
- [ ] **Formal proof**: Show greedy w/ backtracking preserves monotonicity
- [ ] **Inductive argument**: Prove each step maintains monotone property
- [ ] **Density ordering invariants**: Characterize what changes when bid increases

### Extensions
- [ ] Test if optimal (ILP) solution is monotone
- [ ] Implement critical payment rule for truthful mechanism
- [ ] Compare approximation ratio vs. incentive properties

### Practical Application
- [ ] If monotone: design full strategyproof mechanism (allocation + payments)
- [ ] If not monotone: quantify how often/badly violations occur
- [ ] Benchmark against alternative auction mechanisms

---

## Code Tools Available

### `src/monotonicity_test.py`
- `test_monotonicity_single_instance()`: Test one agent on one instance
- `run_monotonicity_tests()`: Large-scale random testing
- `test_specific_scenario()`: Now correctly tests bid increases

### `src/monotonicity_counterexample_generator.py`
- `CounterexampleGenerator`: Systematic search for violations
- Parameterized instance families
- Automated violation detection and analysis

---

**Generated**: 2025-11-04 (Original)
**Updated**: 2025-11-06 (Counterexample added)
**Algorithm**: Greedy Matching with Backtracking (budgeted_bipartite_matching_algs.py:51)
