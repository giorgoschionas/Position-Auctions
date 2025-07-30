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















