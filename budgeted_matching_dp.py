from itertools import chain, combinations
from sortedcontainers import SortedList, SortedSet, SortedDict


# Code is taken from https://github.com/yesyesufcurs/AMBFP/blob/main/BudgetedMatchingDynamic.py

# w[i,j,k] = (maximumWeight using first i edges with budget j using vertices in k)  
# w : {0,...,n} x {0,...,B} x P(V) -> Z+
# w[0,j,k] = 0
# w[i,j,{}] = 0
# w[i,j,k] = max(w[i-1,j,k], w[i-1,j-c(i),k\{v : v in e(i)}] + w(i)) if c(i) <= j and {v : v in e(i)} \in k
# w[i,j,k] = w[i-1,j,k] otherwise

def isSubset(a, b):
    """
    Checks if a subset b
    """
    return a.intersection(b)==a

def verticesIn(edge):
    return SortedSet([edge[0], edge[1]])

def getEdgeCost(G, edge):
    return G[edge[0]][edge[1]]['cost']

def getEdgeWeight(G, edge):
    return G[edge[0]][edge[1]]['weight']

def powerset(iterable, maxSize):
    "Source: https://docs.python.org/3/library/itertools.html#itertools-recipes"
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(maxSize+1))

def powersetAll(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def optimal_budgeted_matching_dp(G, B):
    """
    Solve Budgeted Matching Problem using dynamic programming
    """
    w = [[dict() for _ in range(B + 1)] for _ in range(len(G.edges) + 1)]
    edges = list(G.edges)
    nodes = list(G.nodes)
    for i in range(len(G.edges) + 1):
        for j in range(B + 1):
            for el in powersetAll(nodes):
                k = SortedSet(el)
                if i == 0: w[i][j][str(k)] = 0
                elif k == SortedSet(set()): w[i][j][str(k)] = 0
                elif getEdgeCost(G,edges[i - 1]) <= j and isSubset(verticesIn(edges[i-1]), k): 
                    w[i][j][str(k)] = max(w[i-1][j][str(k)], w[i-1][j-getEdgeCost(G,edges[i-1])][str(k.difference(verticesIn(edges[i-1])))] + getEdgeWeight(G, edges[i-1]))
                else:
                    w[i][j][str(k)] = w[i-1][j][str(k)]
    return w[len(G.edges)][B][str(SortedSet(nodes))]