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
import pickle
import time
from scipy.sparse import csgraph
from scipy import sparse
from copy import deepcopy


# knn graph vs. connected graph
# assert distance in ["l2", "spectral"], "distance must be either 'l2' or 'spectral'"
# assert mode in ["connectivity", "distance"], "mode must be either 'connectivity' or 'distance'", # connectivity == unweighted, distance == weighted
# assert isinstance(n_components, int) if distance == "spectral" else n_components is None, "n_components must be an integer if distance is 'spectral' or None if distance is 'l2'"

def construct_graph(query_embeddings, passage_embeddings, file_path, eigenvectors_path, k=100, distance="l2", n_components=None, use_spectral_distance=False, query_projection=False):
    """
    Constructs a graph based on k-nearest neighbors and returns a NetworkX graph.
    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param file_path: path to save the graph
    :param k: number of nearest neighbors
    :param distance: distance metric
    :param n_components: number of components for spectral embedding
    :param use_spectral_distance: whether to use spectral decomposition
    :param query_projection: whether to project the query embeddings to the spectral embedding space
    :return: graph, query_embeddings, passage_embeddings, original_query_embeddings, original_passage_embeddings
    """    
    return construct_knn_graph(query_embeddings, passage_embeddings, file_path, eigenvectors_path, k, distance, "distance", n_components, use_spectral_distance, query_projection)

def construct_knn_graph(query_embeddings, passages_embeddings, k, file_path, eigenvectors_path, distance="l2", mode="connectivity", n_components=None, use_spectral_distance=False, query_projection=False):
    """
    Constructs a weighted graph based on k-nearest neighbors and returns a NetworkX graph.
    """
    if use_spectral_distance:
       eigenvectors = create_spectral_embedding(query_embeddings, passages_embeddings, n_components, k, file_path, query_projection, distance)
       # save eigenvectors using pickle
       with open(eigenvectors_path, 'wb') as f:
           pickle.dump(eigenvectors, f)
       passages_embeddings = eigenvectors[:, 1:n_components+1] 
    start = time.time()
    if type(passages_embeddings) == torch.Tensor:
        passages_embeddings = passages_embeddings.cpu().numpy()
    adjacency_matrix = kneighbors_graph(passages_embeddings, n_neighbors=k, mode=mode, include_self=False, metric=distance)
    if not use_spectral_distance:

        adjacency_matrix_include_self = kneighbors_graph(passages_embeddings, n_neighbors=k, include_self=True, metric=distance)

        adjacency_matrix_save_path = file_path.replace(".pkl", "_adjacency_matrix.pkl")

        save_adjacency_matrix(adjacency_matrix_include_self, adjacency_matrix_save_path)
    graph = nx.from_scipy_sparse_array(adjacency_matrix)
    end = time.time()
    print("Finished creating graph, it takes", end-start, "seconds")

    save_graph(graph, file_path)
    return graph, query_embeddings, passages_embeddings

def save_graph(G, file_path):
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

def create_spectral_embedding(query_embeddings, passage_embeddings, n_components, k, file_path, query_projection, distance):
    """
    Creates spectral embeddings from input embeddings by constructing a k-nearest neighbor graph,
    """
    file_path = file_path.replace(".pkl", "_adjacency_matrix.pkl").replace(f"_spectral_n_components={n_components}", f"_{distance}")

    # deep copy original embeddings
    original_query_embeddings = deepcopy(query_embeddings)
    original_passage_embeddings = deepcopy(passage_embeddings)

    if os.path.exists(file_path):
        print("Creating spectral embeddings using cached adjacency matrix ...")
        affinity_matrix = read_adjacency_matrix(file_path)

    else:
        print("Creating spectral embeddings from scratch ...")
        start = time.time()
        affinity_matrix = kneighbors_graph(
            passage_embeddings, 
            n_neighbors=k,
            mode='distance',
            include_self=True,
            metric=distance
        )
        end = time.time()
        print("Finished creating affinity matrix, it takes", end-start, "seconds")
        save_adjacency_matrix(affinity_matrix, file_path)
    
    # Make matrix symmetric
    affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
    
    # Convert affinity matrix to dense PyTorch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.from_numpy(affinity_matrix.toarray() if sparse.issparse(affinity_matrix) else affinity_matrix).float().to(device)
    
    # Compute degree matrix
    D = torch.diag(A.sum(dim=1))
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    D_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(D) + 1e-8))  # Add small epsilon for numerical stability
    L = torch.eye(A.shape[0], device=device) - D_sqrt_inv @ A @ D_sqrt_inv
    # Compute eigenvectors and eigenvalues using PyTorch
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Sort eigenvectors by eigenvalues in ascending order
    idx = torch.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors

