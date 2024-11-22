from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

def retrieve_k_manifold_baseline(G: nx.Graph, query_embeddings: np.ndarray, passage_embeddings: np.ndarray, k_neighbors: int=3, weight=1, top_k: int=100) -> np.ndarray:
    """
    Retrieve the top-k passages for each query using the manifold distance retrieval method.

    We assume that the embeddings are already normalized.

    :param G: NetworkX graph representing the manifold
    :param embedding_space: faiss index containing the embeddings of the passages
    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param k_neighbors: Number of neighbors to consider for each query
    :param top_k: Number of passages to retrieve for each query
    :return: numpy array of shape (num_queries, top_k) containing the indices of the retrieved passages
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for manifold distance retrieval")

    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
    all_distances, all_indices = torch.topk(l2_distance_matrix, k_neighbors, dim=1, largest=False, sorted=True)


    indices_set = []
    for i in tqdm(range(len(query_embeddings)), desc="Processing queries"):
        G_copy = deepcopy(G)
        query_idx = len(G_copy.nodes)
        G_copy.add_node(query_idx) # add query node
        
        distances = all_distances[i].cpu().numpy().flatten()
        indices = all_indices[i].cpu().numpy().flatten()

        for i in range(k_neighbors):
            G_copy.add_edge(len(G_copy.nodes) - 1, indices[i], weight=distances[i])

        # find the shortest path
        shortest_path = nx.single_source_dijkstra_path_length(G_copy, query_idx, weight=weight)

        # pop the query node
        shortest_path.pop(query_idx)
        
        sorted_shortest_path = {k_: v_ for k_, v_ in sorted(shortest_path.items(), key=lambda item: item[1])}

        indices = list(sorted_shortest_path.keys())[:top_k] 
        indices_set.append(indices)
    
    return np.array(indices_set)


# def retrieve_k_manifold_qe(G: nx.Graph, query_embeddings: np.ndarray, passage_embeddings: np.ndarray, psuedo_query_embeddings: np.ndarray, weight=1, k_neighbors: int=3, top_k: int=100) -> np.ndarray:
#     """
#     Retrieve the top-k passages for each query using the manifold distance retrieval method with query expansion.

#     We assume that the embeddings are already normalized.

#     :param G: NetworkX graph representing the manifold
#     :param embedding_space: faiss index containing the embeddings of the passages
#     :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
#     :param psuedo_query_embeddings: numpy array of shape (num_queries, num_psuedo_queries, embedding_dim)
#     :param k_neighbors: Number of neighbors to consider for each psuedo query
#     :param top_k: Number of passages to retrieve for each query
#     :return: numpy array of shape (num_queries, top_k) containing the indices of the retrieved passages
#     """
#     assert query_embeddings.shape[1] == psuedo_query_embeddings.shape[1], "Embedding dimensions of queries and psuedo queries do not match"
#     assert query_embeddings.shape[0] == psuedo_query_embeddings.shape[0], "Number of queries and psuedo queries do not match"

#     query_embeddings = torch.tensor(query_embeddings, device=device)
#     passage_embeddings = torch.tensor(passage_embeddings, device=device)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
#     all_distances, all_indices = torch.topk(l2_distance_matrix, k_neighbors, dim=1, largest=False, sorted=True)

#     indices_set = []
#     for query_embedding, psuedo_query_embedding_set in zip(query_embeddings, psuedo_query_embeddings):
#         G_copy = deepcopy(G)
#         query_idx = len(G_copy.nodes)
#         G_copy.add_node(query_idx) # add query node
        
#         ###### add all psuedo query #######
#         for psuedo_query_embedding in psuedo_query_embedding_set:
#             G_copy.add_node(len(G_copy.nodes)) # add psuedo query node
#             l2_distance = np.linalg.norm(query_embedding - psuedo_query_embedding) # connect query and psuedo query
#             G_copy.add_edge(query_idx, len(G_copy.nodes) - 1, weight=l2_distance)

#             # create edges between the psuedo query and its k nearest neighbors
#             squared_distances, indices = embedding_space.search(psuedo_query_embedding.reshape(1, -1), k_neighbors)
#             distances = np.sqrt(squared_distances)
#             indices = indices.flatten()
#             distances = distances.flatten()

#             for i in range(k_neighbors):
#                 G_copy.add_edge(len(G_copy.nodes) - 1, indices[i], weight=distances[i])

#         # find the shortest path
#         shortest_path = nx.single_source_dijkstra_path_length(G_copy, query_idx, weight=weight)
#         sorted_shortest_path = {k_: v_ for k_, v_ in sorted(shortest_path.items(), key=lambda item: item[1])}
            
#         indices = list(sorted_shortest_path.keys())[:top_k] 
#         indices_set.append(indices)
    
#     return np.array(indices_set)