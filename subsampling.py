import random

def subsample_nodes(nodes, sample_size):
    return random.sample(nodes, sample_size)

def subsample_edges(edges, sample_size):
    return random.sample(edges, sample_size)