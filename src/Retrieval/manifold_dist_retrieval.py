from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from heapq import heappush, heappop

def dijkstra_shortest_path(G: nx.Graph, query_idx: int, top_k: int=100) -> dict:
    """
    Use Dijkstra's algorithm to find the shortest path from the query node to the top-k nodes.
    """
    distances = {node: float('infinity') for node in G.nodes()}
    distances[query_idx] = 0
    
    # Priority queue entries are tuples of (distance, node)
    pq = [(0, query_idx)]
    visited = {}
    
    while pq and len(visited) < top_k:
        current_dist, current = heappop(pq)
        
        # Skip if we've already found a better path
        if current in visited:
            continue
            
        visited[current] = current_dist
        
        # Explore neighbors
        for neighbor in G[current]:
            if neighbor in visited:
                continue
                
            # Get edge weight, default to 1 if not specified
            weight = G[current][neighbor].get('weight', 1)
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(pq, (distance, neighbor))
    
    return visited

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

        for j in range(k_neighbors):
            G_copy.add_edge(len(G_copy.nodes) - 1, indices[j], weight=distances[j])

        # find the shortest path
        # shortest_path = nx.single_source_dijkstra_path_length(G=G_copy, source=query_idx, cutoff=1, weight=weight)
        shortest_path = dijkstra_shortest_path(G=G_copy, query_idx=query_idx, top_k=top_k)

        # pop the query node
        shortest_path.pop(query_idx)
        
        sorted_shortest_path = {k_: v_ for k_, v_ in sorted(shortest_path.items(), key=lambda item: item[1])}

        indices = list(sorted_shortest_path.keys())[:top_k] 
        indices_set.append(indices)
    
    return np.array(indices_set)


def retrieve_k_manifold_baseline_reciprocal(G: nx.Graph, query_embeddings: np.ndarray, passage_embeddings: np.ndarray, k_neighbors: int=3, weight=1, top_k: int=100, N: int=10) -> np.ndarray:
    """
    Retrieve the top-k passages for each query using the manifold distance retrieval method.

    We assume that the embeddings are already normalized.

    :param G: NetworkX graph representing the manifold
    :param embedding_space: faiss index containing the embeddings of the passages
    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param k_neighbors: Number of neighbors to consider for each query
    :param top_k: Number of passages to retrieve for each query
    :param N: Number of neighbors to consider for each passage
    :return: numpy array of shape (num_queries, top_k) containing the indices of the retrieved passages
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for manifold distance retrieval")

    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)

    # get the top N neighbors
    all_distances, all_indices = torch.topk(l2_distance_matrix, N, dim=1, largest=False, sorted=True)
    
    indices_set = []
    for i in tqdm(range(len(query_embeddings)), desc="Processing queries"):
        G_copy = deepcopy(G)
        query_idx = len(G_copy.nodes)
        G_copy.add_node(query_idx) # add query node
        
        distances = all_distances[i].cpu().numpy().flatten()
        indices = all_indices[i].cpu().numpy().flatten()


        # only keep neighbors that its top N neighbors contain the query
        test_passage_embeddings = passage_embeddings[indices]
        test_passage_l2_distance_matrix = torch.cdist(test_passage_embeddings, passage_embeddings, p=2)
        test_passage_distances, test_passage_indices = torch.topk(test_passage_l2_distance_matrix, N + 1, dim=1, largest=False, sorted=True)

        reciprocal_indices = []
        for j in range(N):
            # if distance is less than the distance between the query and its top N neighbors, then it is a reciprocal neighbor
            if test_passage_distances[j, N] < distances[j]: 
                reciprocal_indices.append(indices[j])

        for k in range(len(reciprocal_indices)):
            G_copy.add_edge(len(G_copy.nodes) - 1, reciprocal_indices[k], weight=distances[k])

        # find the shortest path
        shortest_path = nx.single_source_dijkstra_path_length(G_copy, query_idx, cutoff=top_k, weight=weight)

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