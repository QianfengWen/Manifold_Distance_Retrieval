import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import json
import pdb
import torch
from networkx.readwrite import json_graph
import pickle

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
    embeddings = create_spectral_embedding(passages_embeddings, k, n_components) if distance == "spectral" else passages_embeddings
    print("Finished creating embeddings")
    adjacency_matrix = kneighbors_graph(embeddings, n_neighbors=k, mode=mode, include_self=False)
    print("Finished creating adjacency matrix")
    graph = nx.from_scipy_sparse_array(adjacency_matrix)
    print("Finished creating graph")

    save_graph(graph, file_path)
    return graph


# Manifold Ranking
def construct_connected_graph(passages_embeddings, file_path, k=100, max_edges=None, max_percentage=None, distance="l2", mode="connectivity", n_components=None):
    """
    Constructs a connected graph with optional limits on the number of edges.
    """
    # prevent size being too large
    embeddings = create_spectral_embedding(passages_embeddings, k, n_components) if distance == "spectral" else passages_embeddings
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
    return


def read_graph(file_path):
    # # read the graph using networkx
    # G = json_graph.node_link_graph(json.load(open(file_path, 'r')))
    # return G

    # read the graph as a adjacency matrix
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G


# helper for spectral embedding
def construct_similarity_graph(embeddings, k):
    """
    Constructs a k-nearest neighbor similarity graph from input embeddings.
    """
    adjacency_matrix = kneighbors_graph(
        embeddings,
        n_neighbors=k,
        mode='connectivity',
        include_self=False,
        metric='euclidean'
    )
    return adjacency_matrix.toarray()

# def compute_laplacian(adjacency_matrix, normalized):
#     """
#     Computes the Laplacian matrix from an adjacency matrix.
#     """
#     degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
#     print("Finished creating degree matrix")
#     if normalized:
#         with np.errstate(divide='ignore'):
#             d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
#             print("Finished creating d_inv_sqrt")
#             d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0 
#             print("Finished creating d_inv_sqrt")
#         laplacian = np.identity(adjacency_matrix.shape[0]) - d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
#         print("Finished creating laplacian matrix")
#     else:
#         laplacian = degree_matrix - adjacency_matrix
#         print("Finished creating laplacian matrix")
#     return laplacian


def compute_laplacian(adjacency_matrix, normalized=True):
    """
    Computes the Laplacian matrix from an adjacency matrix using CUDA for GPU acceleration.
    
    Args:
        adjacency_matrix (torch.Tensor): The adjacency matrix (should be on GPU).
        normalized (bool): Whether to compute the normalized Laplacian.
    
    Returns:
        torch.Tensor: The Laplacian matrix (on GPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for computing laplacian")

    # Process in chunks to reduce memory usage
    chunk_size = 1000  # Adjust based on available GPU memory
    n = adjacency_matrix.shape[0]
    num_chunks = (n + chunk_size - 1) // chunk_size

    # Initialize result arrays on CPU
    degree_vector = np.zeros(n)
    
    # Calculate degrees in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n)
        
        chunk = torch.tensor(adjacency_matrix[start_idx:end_idx], device=device)
        degree_vector[start_idx:end_idx] = torch.sum(chunk, dim=1).cpu().numpy()
        del chunk  # Free GPU memory

    if normalized:
        # Compute D^(-1/2) on CPU
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vector))
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_inv_sqrt = torch.tensor(d_inv_sqrt, device=device)
        print("Finished creating d_inv_sqrt")
        
        # Initialize laplacian on CPU
        laplacian = np.eye(n)
        
        # Compute normalized Laplacian in chunks
        for i in tqdm(range(num_chunks), desc="Computing normalized Laplacian"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            
            chunk = torch.tensor(adjacency_matrix[start_idx:end_idx], device=device)
            # Fix: Correct matrix multiplication order and slicing
            result_chunk = torch.mm(
                torch.mm(d_inv_sqrt[start_idx:end_idx, start_idx:end_idx], chunk),
                d_inv_sqrt
            ).cpu().numpy()
            laplacian[start_idx:end_idx] -= result_chunk
            del chunk, result_chunk  # Free GPU memory
            
        print("Finished creating laplacian matrix")
        
    else:
        # Compute unnormalized Laplacian on CPU
        laplacian = np.diag(degree_vector) - adjacency_matrix
        print("Finished creating laplacian matrix")
    
    return laplacian

def compute_spectral_embedding(laplacian, n_components):
    """
    Computes spectral embeddings from the Laplacian matrix.
    """
    # Convert to numpy for eigenvalue computation
    # We only need the first n_components+1 eigenvalues/vectors
    eigenvalues, eigenvectors = eigh(laplacian, eigvals=(1, n_components))
    
    # Normalize the eigenvectors
    embeddings = normalize(eigenvectors, norm='l2', axis=1)
    
    return embeddings

def create_spectral_embedding(embeddings, k, n_components, normalized=True):
    """
    Creates spectral embeddings from input embeddings by constructing a k-nearest neighbor graph,
    """

    adjacency_matrix = construct_similarity_graph(embeddings, k)
    print("Finished creating similarity graph")
    laplacian = compute_laplacian(adjacency_matrix, normalized)
    print("Finished creating laplacian matrix")
    spectral_embeddings = compute_spectral_embedding(laplacian, n_components)
    print("Finished creating spectral embeddings")

    return spectral_embeddings


# def nearest_neighbors(passages_embeddings, k):
#     doc_nearest_neighbors_indices = []
#     doc_nearest_neighbors_distances = []

#     # convert passages_embeddings to torch tensor
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device} for constructing graph")
#     passages_embeddings = torch.tensor(passages_embeddings, device=device)

#     for idx, d in tqdm(enumerate(passages_embeddings), desc="Searching nearest neighbors"):

#         l2_distance_matrix = torch.cdist(d.reshape(1, -1), passages_embeddings, p=2)
#         distances, indices = torch.topk(l2_distance_matrix, k + 1, dim=1, largest=False, sorted=True)

#         # skip the search embedding itself
#         self_index = torch.where(indices == idx)[1][0]

#         # skip self_index
#         indices = torch.cat((indices[0][0:self_index], indices[0][self_index+1:])).cpu().numpy().flatten()
#         distances = torch.cat((distances[0][0:self_index], distances[0][self_index+1:])).cpu().numpy().flatten()

#         doc_nearest_neighbors_indices.append(indices)
#         doc_nearest_neighbors_distances.append(distances)
    
#     return doc_nearest_neighbors_indices, doc_nearest_neighbors_distances



# def construct_graph_reciprocal(passages_embeddings, k, file_path):
#     # search for the nearest neighbors
#     doc_nearest_neighbors_indices, doc_nearest_neighbors_distances = nearest_neighbors(passages_embeddings, k)

#     # create a graph
#     G = nx.Graph()
#     for i in range(len(passages_embeddings)):
#         for j in range(k):
#             if i in doc_nearest_neighbors_indices[doc_nearest_neighbors_indices[i][j]]:
#                 G.add_edge(i, doc_nearest_neighbors_indices[i][j], weight=doc_nearest_neighbors_distances[i][j]) # add an weighted edge that connects i-th query (i) and its j-th nearest neighbor passage, the weight is the distance between them
#     # save the graph
#     save_graph(G, file_path)
#     return G