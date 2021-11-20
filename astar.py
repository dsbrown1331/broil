import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode


def astar_search(states, actions, rsa, start, end):
    paths = []
    cities = [i[0] for i in states]
    for j in range(rsa.shape[1]):
        G = nx.Graph()
        for city in cities:
            G.add_node(city)

        for i, action in enumerate(actions):
            s, e = cities.index(action[0]), cities.index(action[1])
            G.add_edge(action[0], action[1], weight = -1 / rsa[i * len(states) + s][j])

        paths.append(tuple(nx.astar_path(G, start[0], end[0], weight='weight')))
    vertex_list = mode(paths)
    edges = [(vertex_list[i], vertex_list[i + 1]) for i in range(0, len(vertex_list) -1)]
    policy = np.zeros(rsa.shape[0])
    for edge in edges:
        policy[actions.index(edge) * len(states) + cities.index(edge[0])] = 1
    return policy
