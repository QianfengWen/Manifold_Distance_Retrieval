from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import deque
from heapq import heappush, heappop
import heapq

# def dijkstra_shortest_path(G: nx.Graph, query_idx: int, top_k: int=100) -> dict:
#     """
#     Use Dijkstra's algorithm to find the shortest path from the query node to the top-k nodes.
#     """
#     distances = {node: float('infinity') for node in G.nodes()}
#     distances[query_idx] = 0
    
#     # Priority queue entries are tuples of (distance, node)
#     pq = [(0, query_idx)]
#     visited = {}
    
#     while pq and len(visited) < top_k * 10:
#         current_dist, current = heappop(pq)
        
#         # Skip if we've already found a better path
#         if current in visited:
#             continue
            
#         visited[current] = current_dist
        
#         # Explore neighbors
#         for neighbor in G[current]:
#             if neighbor in visited:
#                 continue
                
#             # Get edge weight 
#             weight = G[current][neighbor]["weight"]
#             distance = current_dist + weight
            
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heappush(pq, (distance, neighbor))
    
#     return visited

def dijkstra_shortest_path(G: nx.Graph, query_idx: int, top_k: int = 100) -> dict:
    """
    Finds the shortest paths from a source node to other nodes in a weighted graph using Dijkstra's algorithm.

    Parameters:
    - G (nx.Graph): The input graph, where edge weights are stored in the `weight` attribute.
    - query_idx (int): The starting node for the shortest path search.
    - top_k (int): The maximum number of nodes for which shortest paths will be computed.

    Returns:
    - dict: A dictionary with nodes as keys and their shortest path distances from the source node as values.
    """
    # Priority queue for Dijkstra's algorithm
    priority_queue = [(0, query_idx)]  # (distance, node)
    shortest_paths = {query_idx: 0}  # Store the shortest distances
    visited = set()  # Track visited nodes

    while priority_queue and len(shortest_paths) < top_k:
        # Pop the node with the smallest distance
        current_dist, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        # Iterate over neighbors
        for neighbor, edge_attrs in G[current_node].items():
            weight = edge_attrs.get('weight', 1)  # Default weight is 1 if not provided
            new_dist = current_dist + weight

            if neighbor not in shortest_paths or new_dist < shortest_paths[neighbor]:
                shortest_paths[neighbor] = new_dist
                heapq.heappush(priority_queue, (new_dist, neighbor))
    return shortest_paths

def bfs_shortest_path(G: nx.Graph, query_idx: int, top_k: int=100) -> dict:
    # Use deque for O(1) pop operations
    queue = deque([query_idx])
    visited = {query_idx: 0}  # Combine visited set and shortest_path_length dict
    
    while queue and len(visited) < top_k:
        current = queue.popleft()  # O(1) operation instead of O(n)
        current_dist = visited[current]
        
        # Get neighbors and their distances in one go
        for neighbor in G[current]:  # More efficient neighbor iteration
            if neighbor not in visited:
                visited[neighbor] = current_dist + 1
                queue.append(neighbor)
                
                if len(visited) >= top_k:
                    break
    return visited

def retrieve_k_manifold_baseline(
    G: nx.Graph, 
    query_embeddings: np.ndarray, 
    passage_embeddings: np.ndarray, 
    k_neighbors: int = 3,  
    top_k: int = 100,
    use_edge_weight: bool = False
) -> np.ndarray:
    """
    Retrieve the top-k passages for each query using the manifold distance retrieval method.

    We assume that the embeddings are already normalized.

    :param G: NetworkX graph representing the manifold
    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param k_neighbors: Number of neighbors to consider for each query
    :param top_k: Number of passages to retrieve for each query
    :return: numpy array of shape (num_queries, top_k) containing the indices of the retrieved passages
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for manifold distance retrieval")

    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    # Compute initial distances to find candidate neighbors
    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
    all_distances, all_indices = torch.topk(l2_distance_matrix, k_neighbors, dim=1, largest=False, sorted=True)

    indices_set = []
    # We'll assume that the query node index does not collide with existing nodes.
    # One way to ensure uniqueness is to start from a large number or the max node index + 1.
    # For safety, find the max existing node index:
    max_node_idx = max(G.nodes) if len(G.nodes) > 0 else 0
    current_query_node = max_node_idx + 1

    for i in tqdm(range(len(query_embeddings)), desc="Processing queries"):
        query_idx = current_query_node
        current_query_node += 1
        
        # Add a query node for this iteration
        G.add_node(query_idx)
        
        distances = all_distances[i].cpu().numpy().flatten()
        indices = all_indices[i].cpu().numpy().flatten()

        # Add edges from query node to its k_neighbors
        for j in range(k_neighbors):
            G.add_edge(query_idx, indices[j], weight=distances[j])

        # Compute shortest path
        if use_edge_weight:
            shortest_path = dijkstra_shortest_path(G=G, query_idx=query_idx, top_k=top_k * 2)
        else:
            shortest_path = bfs_shortest_path(G=G, query_idx=query_idx, top_k=top_k)
        # shortest_path = bfs_shortest_path(G=G, query_idx=query_idx, top_k=top_k)
        # Remove the query node and associated edges after computation
        # Removing the node will also remove any edges associated with it.
        G.remove_node(query_idx)

        # Remove query_idx from the path results
        if query_idx in shortest_path:
            shortest_path.pop(query_idx)

        # Sort results by distance
        sorted_shortest_path = {
            k_: v_ for k_, v_ in sorted(shortest_path.items(), key=lambda item: item[1])
        }

        retrieved_indices = list(sorted_shortest_path.keys())[:top_k]
        indices_set.append(retrieved_indices)
    
    max_len = max(len(seq) for seq in indices_set)
    padded_indices_set = [seq + [-1] * (max_len - len(seq)) for seq in indices_set]
    return np.array(padded_indices_set)

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