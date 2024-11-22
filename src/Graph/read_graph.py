import json
import networkx as nx

def read_graph(file_path):
    with open(file_path, 'r') as f:
        G_New = json.load(f)
    G = nx.Graph()
    for node in G_New:
        for neighbor, weight in G_New[node].items():
            G.add_edge(int(node), int(neighbor), weight=float(weight))
    return G