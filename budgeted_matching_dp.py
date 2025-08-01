from collections import defaultdict
from math import inf
from typing import Dict, List, Tuple

Edge = Tuple[int, int, int, int]          # (u, v, value, cost)

def max_value_budget_dp(n_left: int,
                        n_right: int,
                        edges: List[Edge],
                        B: int
                       ) -> Tuple[int, Dict[int, int]]:
    """
    Maximum-value matching in a bipartite graph subject to a budget.

    Parameters
    ----------
    n_left, n_right : int
        Number of vertices on the left / right side, labelled 0 … n-1.
    edges : list[(u, v, value, cost)]
        u ∈ [0,n_left), v ∈ [0,n_right).  Multiple edges between the same
        pair are allowed; the best one is kept automatically.
    B : int
        Budget (non-negative).

    Returns
    -------
    best_value : int
        Maximum total value achievable within budget B.
    matching    : dict[int,int]
        A dict {u: v} describing one optimal matching.
        Vertices not in the dict are left unmatched.
    """
    # ------------------------------------------------------------------
    # 1.  Build best edge (value,cost) for every (u,v) that exists.
    # ------------------------------------------------------------------
    value = [[-inf]*n_right for _ in range(n_left)]
    cost  = [[ inf]*n_right for _ in range(n_left)]
    for u, v, val, c in edges:
        if val > value[u][v] or (val == value[u][v] and c < cost[u][v]):
            value[u][v] = val
            cost[u][v]  = c

    # ------------------------------------------------------------------
    # 2.  If the *right* side is larger than the left, swap sides so
    #     that we always enumerate subsets of the *smaller* side.
    #     (Purely for speed/memory; result is swapped back at the end.)
    # ------------------------------------------------------------------
    swapped = False
    if n_right > n_left:
        swapped = True
        n_left, n_right = n_right, n_left       # swap sizes
        # transpose the matrices
        value = [list(row) for row in zip(*value)]
        cost  = [list(row) for row in zip(*cost)]

    m = n_right                 # size of subset dimension   (≤ 20 recommended)
    FULL = 1 << m

    # ------------------------------------------------------------------
    # 3.  DP tables: dp[S] is a dict {budget_used: best_value}
    #     Only two layers (current and next) are kept to save RAM.
    # ------------------------------------------------------------------
    dp: List[Dict[int, int]] = [defaultdict(lambda: -inf) for _ in range(FULL)]
    dp[0][0] = 0                      # nothing chosen yet

    parent = [{} for _ in range(n_left)]   # to reconstruct matching

    for u in range(n_left):
        next_dp = [dict(cell) for cell in dp]   # start with "skip u"
        parent_u = {}                           # (S, c) ➜ (prev_S, prev_c, v_matched_or_None)

        for S in range(FULL):
            if not dp[S]:                       # no feasible budgets for this subset
                continue
            free_mask = (~S) & (FULL - 1)
            while free_mask:
                v = (free_mask & -free_mask).bit_length() - 1   # lowest free v
                free_mask &= free_mask - 1

                val_uv = value[u][v]
                if val_uv == -inf:               # edge nonexistent
                    continue
                c_uv = cost[u][v]

                for c_used, best_val in dp[S].items():
                    c_new = c_used + c_uv
                    if c_new > B:
                        continue
                    S_new = S | (1 << v)
                    new_val = best_val + val_uv
                    if new_val > next_dp[S_new].get(c_new, -inf):
                        next_dp[S_new][c_new] = new_val
                        parent_u[(S_new, c_new)] = (S, c_used, v)

        dp = next_dp
        parent[u] = parent_u

    # ------------------------------------------------------------------
    # 4.  Extract best answer over all subsets & budgets
    # ------------------------------------------------------------------
    best_val = -inf
    best_state = None
    for S in range(FULL):
        for c_used, val in dp[S].items():
            if val > best_val:
                best_val = val
                best_state = (n_left - 1, S, c_used)   # last layer index

    # ------------------------------------------------------------------
    # 5.  Reconstruct matching by following 'parent' pointers backwards
    # ------------------------------------------------------------------
    matching = {}
    layer, S, c_used = best_state
    for u in range(layer, -1, -1):
        prev = parent[u].get((S, c_used))
        if prev is None:          # u was skipped
            continue
        prev_S, prev_c, v = prev
        if v is not None:
            if swapped:
                matching[v] = u   # swap sides back
            else:
                matching[u] = v
        S, c_used = prev_S, prev_c

    return best_val if best_val > -inf else 0, matching


# Example

if __name__ == "__main__":
    # Example: 3×3 complete bipartite graph
    nL, nR = 3, 3
    import random, itertools
    random.seed(0)
    edges = []
    for u, v in itertools.product(range(nL), range(nR)):
        val  = random.randint( 5, 30)
        cost = random.randint(10, 25)
        edges.append((u, v, val, cost))

    B = 40
    best_val, match = max_value_budget_dp(nL, nR, edges, B)

    print("Best total value :", best_val)
    print("Matching within budget:", match)