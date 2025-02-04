import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import os
import json
import pdb
import torch
from sklearn.manifold import SpectralEmbedding
from networkx.readwrite import json_graph
import pickle
import time

# knn graph vs. connected graph
# assert distance in ["l2", "spectral"], "distance must be either 'l2' or 'spectral'"
# assert mode in ["connectivity", "distance"], "mode must be either 'connectivity' or 'distance'", # connectivity == unweighted, distance == weighted
# assert isinstance(n_components, int) if distance == "spectral" else n_components is None, "n_components must be an integer if distance is 'spectral' or None if distance is 'l2'"

def construct_graph(passage_embeddings, file_path, k=100, graph_type="knn", distance="l2", n_components=None, max_edges=None, max_percentage=None):
    """
    Constructs a graph based on k-nearest neighbors and returns a NetworkX graph.
    """
    assert graph_type in ["knn", "connected"], "graph must be either 'knn' or 'connected'"
    assert distance in ["l2", "spectral"], "distance must be either 'l2' or 'spectral'"
    # assert mode in ["connectivity", "distance"], "mode must be either 'connectivity' or 'distance'" # connectivity == unweighted, distance == weighted
    assert isinstance(n_components, int) if distance == "spectral" else n_components is None, "n_components must be an integer if distance is 'spectral' or None if distance is 'l2'"
    
    if graph_type == "knn":
        return construct_knn_graph(passage_embeddings, file_path, k, distance, "distance", n_components)
    elif graph_type == "connected":
        return construct_connected_graph(passage_embeddings, file_path, k, max_edges, max_percentage, distance, "distance", n_components)
    else:
        raise ValueError(f"Invalid graph type: {graph_type}")


def construct_knn_graph(passages_embeddings, k, file_path, distance="l2", mode="connectivity", n_components=None):
    """
    Constructs a weighted graph based on k-nearest neighbors and returns a NetworkX graph.
    """
    embeddings = create_spectral_embedding(passages_embeddings, n_components, k, file_path) if distance == "spectral" else passages_embeddings
    print("Finished creating embeddings")
    start = time.time()
    adjacency_matrix = kneighbors_graph(embeddings, n_neighbors=k, mode=mode, include_self=False)
    if distance == "l2":
        adjacency_matrix_include_self = kneighbors_graph(embeddings, n_neighbors=k, include_self=True)
        adjacency_matrix_save_path = file_path.replace(".pkl", "_adjacency_matrix.pkl")
        save_adjacency_matrix(adjacency_matrix_include_self, adjacency_matrix_save_path)
    graph = nx.from_scipy_sparse_array(adjacency_matrix)
    end = time.time()
    print("Finished creating graph, it takes", end-start, "seconds")

    save_graph(graph, file_path)
    return graph


# Manifold Ranking
def construct_connected_graph(passages_embeddings, file_path, k=100, max_edges=None, max_percentage=None, distance="l2", mode="connectivity", n_components=None):
    """
    Constructs a connected graph with optional limits on the number of edges.
    """
    # prevent size being too large
    embeddings = create_spectral_embedding(passages_embeddings, n_components, k, file_path) if distance == "spectral" else passages_embeddings
    adjacency_matrix = kneighbors_graph(
        embeddings, n_neighbors=k, mode='distance', include_self=False
    )
    graph = nx.from_scipy_sparse_array(adjacency_matrix)


    # Step 1: Sort edges by weight
    edges = [(u, v, graph[u][v]['weight']) for u, v in graph.edges()]
    edges.sort(key=lambda x: x[2])  
    num_nodes = len(passages_embeddings)

    # Step 2: Initialize graph and set limits
    num_nodes = len(passages_embeddings)
    connected_graph = nx.Graph()
    connected_graph.add_nodes_from(range(num_nodes))
    total_possible_edges = num_nodes * (num_nodes - 1) // 2
    max_edges_limit = max_edges or total_possible_edges
    max_edges_limit = min(max_edges_limit, int(max_percentage * total_possible_edges) if max_percentage else max_edges_limit)

    # Step 3: Add edges until limits are reached
    for u, v, weight in edges:
        weight = weight if mode == 'distance' else 1
        connected_graph.add_edge(u, v, weight=weight)  

        if nx.is_connected(connected_graph):
            break

        if connected_graph.number_of_edges() >= max_edges_limit:
            break

    if file_path:
        save_graph(connected_graph, file_path)

    return connected_graph


def save_graph(G, file_path):
    # # save the graph as a json file
    # G_New = json_graph.node_link_data(G)
    # with open(file_path, 'w') as f:
    #     json.dump(dict(G_New), f, indent=4)
    # return G_New

    # save the graph as a pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(G, f)
    print("Saving graph to", file_path)
    return

def save_adjacency_matrix(adjacency_matrix, file_path):
    print("Saving adjacency matrix to", file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(adjacency_matrix, f)
    return

def read_graph(file_path):
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    print("Loading graph from", file_path)
    return G

def read_adjacency_matrix(file_path):
    with open(file_path, 'rb') as f:
        adjacency_matrix = pickle.load(f)
    print("Loading adjacency matrix from", file_path)
    return adjacency_matrix

def create_spectral_embedding(embeddings, n_components, k, file_path):
    """
    Creates spectral embeddings from input embeddings by constructing a k-nearest neighbor graph,
    """

    # create a spectral embedding
    file_path = file_path.replace(".pkl", "_adjacency_matrix.pkl").replace(f"_spectral_n_components={n_components}", "_l2")
    if os.path.exists(file_path):
        print("Creating spectral embeddings using cached adjacency matrix ...")
        affinity_matrix = read_adjacency_matrix(file_path)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        spectral_embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed', random_state=311)
        start = time.time()
        embeddings = spectral_embedding.fit_transform(affinity_matrix)
        end = time.time()
        print("Finished fit_transform, it takes", end-start, "seconds")
    else:
        print("Creating spectral embeddings from scratch ...")
        start = time.time()
        affinity_matrix = kneighbors_graph(embeddings, n_neighbors=k, include_self=True)
        end = time.time()
        print("Finished creating affinity matrix, it takes", end-start, "seconds")
        save_adjacency_matrix(affinity_matrix, file_path)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        spectral_embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed', random_state=311)
        start = time.time()
        embeddings = spectral_embedding.fit_transform(affinity_matrix)
        end = time.time()
        print("Finished fit_transform, it takes", end-start, "seconds")
    return embeddings

