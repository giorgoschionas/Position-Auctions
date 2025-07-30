import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Constants of the setting
num_txs = 10
num_positions = 5
knapsack_size = 1000
max_transaction_size = 5

class Transaction:
    def __init__(self, id, size, vals):
        self.id =id
        self.size = size
        self.vals = vals
        self.densities = [val/ size for val in vals]

def create_transactions(num_transactions, num_positions):
    transactions = []
    for i in range(num_transactions):
        size = random.uniform(0.1, max_transaction_size)  # Random size between 0.1 and max_transaction_size
        vals = [random.uniform(0, 10) for _ in range(num_positions)]
        transactions.append(Transaction(id=i, size=size, vals=vals))
    return transactions

# Create a new list of tuples (density, transaction_index, position_index) - not very Pythonic
densities_with_id_pos = []

for idx, transaction in enumerate(create_transactions(num_txs, num_positions)):
    for pos, density in enumerate(transaction.densities):
        densities_with_id_pos.append((density, idx, pos))

# Sort the list by density (descending order)
densities_with_id_pos.sort(key=lambda x: x[0], reverse=True)

# To show the result
for density, owner_idx, pos in densities_with_id_pos:
    print(f"Density: {density}, belongs to transaction index: {owner_idx}, position index: {pos}")


# Our greedy matching algorithm - NEEDS to be checked!
def greedy_budgeted_matching(transactions, densities_with_id_pos, knapsack_size):
    used_budget = 0
    greedy_matching = set()
    used_transactions = set()

    for density, transaction_idx, position_idx in densities_with_id_pos:
        if used_budget + transactions[transaction_idx].size <= knapsack_size and transaction_idx not in used_transactions:
            greedy_matching.add((transaction_idx, position_idx))
            used_budget += transactions[transaction_idx].size
            used_transactions.add(transaction_idx)

    print("value of budgeted greedy matching: ", sum(transactions[idx].vals[pos] for idx, pos in greedy_matching))
    return greedy_matching









