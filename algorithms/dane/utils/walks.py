import networkx as nx
import numpy as np
from tqdm import tqdm
import random

def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

def simulate_walks(G, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    walks = []
    nodes = list(G.nodes())

    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter+1) + '/' + str(num_walks))
        random.shuffle(nodes)
        for node in nodes:
            walks.append(deepwalk_walk(G=G, walk_length=walk_length, start_node=node))
    return walks


def get_walks(A, config):
    graph = nx.from_numpy_matrix(A.todense())

    num_walks = config['num_walks']
    walk_length = config['walk_length']
    window_size = config['window_size']

    walks = simulate_walks(graph, num_walks, walk_length)

    num_nodes = len(graph.nodes())
    adj_matrix = np.ndarray((num_nodes, num_nodes), np.float32)

    for line in tqdm(walks):
        for pos, src in enumerate(line):
            start = max(0, pos - window_size)
            for pos2, dst in enumerate(line[start:(pos + window_size + 1)], start):
                if pos2 != pos:
                    adj_matrix[src, dst] = 1.0
                    adj_matrix[dst, src] = 1.0

    return adj_matrix
