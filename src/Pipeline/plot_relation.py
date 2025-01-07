# load embeddings
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pdb
from heapq import heappush, heappop
import seaborn as sns

def load_embeddings(query_embeddings_path, passage_embeddings_path):
    with open(query_embeddings_path, "rb") as f:
        query_embeddings = pickle.load(f)
        assert isinstance(query_embeddings, np.ndarray), "query_embeddings should be a numpy array"

    with open(passage_embeddings_path, "rb") as f:
        passage_embeddings = pickle.load(f)
        assert isinstance(passage_embeddings, np.ndarray), "passage_embeddings should be a numpy array"

    return query_embeddings, passage_embeddings

def nearest_neighbors(passages_embeddings, k):
    doc_nearest_neighbors_indices = []
    doc_nearest_neighbors_distances = []

    # convert passages_embeddings to torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for constructing graph")
    passages_embeddings = torch.tensor(passages_embeddings, device=device)

    for idx, d in tqdm(enumerate(passages_embeddings), desc="Searching nearest neighbors"):

        l2_distance_matrix = torch.cdist(d.reshape(1, -1), passages_embeddings, p=2)
        distances, indices = torch.topk(l2_distance_matrix, k + 1, dim=1, largest=False, sorted=True)

        # skip the search embedding itself
        self_index = torch.where(indices == idx)[1][0]

        # skip self_index
        indices = torch.cat((indices[0][0:self_index], indices[0][self_index+1:])).cpu().numpy().flatten()
        distances = torch.cat((distances[0][0:self_index], distances[0][self_index+1:])).cpu().numpy().flatten()

        doc_nearest_neighbors_indices.append(indices)
        doc_nearest_neighbors_distances.append(distances)
    
    return doc_nearest_neighbors_indices, doc_nearest_neighbors_distances

def construct_graph(passages_embeddings, k):
    # search for the nearest neighbors
    doc_nearest_neighbors_indices, doc_nearest_neighbors_distances = nearest_neighbors(passages_embeddings, k)

    # create a graph
    G = nx.Graph()
    for i in range(len(passages_embeddings)):
        for j in range(k):
            G.add_edge(i, doc_nearest_neighbors_indices[i][j], weight=doc_nearest_neighbors_distances[i][j]) # add an weighted edge that connects i-th query (i) and its j-th nearest neighbor passage, the weight is the distance between them
    return G

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


if __name__ == "__main__":
    dataset = "scidoc"
    k = 10
    query_embeddings_path = f"data/{dataset}/tas-b-query_embeddings.pkl"
    passage_embeddings_path = f"data/{dataset}/tas-b-doc_embeddings.pkl"
    

    query_embeddings, passage_embeddings = load_embeddings(query_embeddings_path, passage_embeddings_path)

    G = construct_graph(passage_embeddings, k)

    l2_distances = []
    manifold_distances = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for constructing graph")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
    all_distances, all_indices = torch.topk(l2_distance_matrix, k, dim=1, largest=False, sorted=True)

    for i in tqdm(range(len(query_embeddings)), desc="Constructing graph"):
        query_idx = len(G.nodes)
        G.add_node(query_idx) # add query node
        
        distances = all_distances[i].cpu().numpy().flatten()
        indices = all_indices[i].cpu().numpy().flatten()

        for j in range(k):
            G.add_edge(len(G.nodes) - 1, indices[j], weight=distances[j])

        # find the shortest path between the query and the passage
        try:
            manifold = dijkstra_shortest_path(G, query_idx, len(G.nodes))
            # Store both L2 and manifold distances for plotting
            for node, man_dist in manifold.items():
                if node != query_idx:  # Skip the query node itself
                    l2_dist = float(l2_distance_matrix[i][node].cpu().numpy())
                    l2_distances.append(l2_dist)
                    manifold_distances.append(man_dist)
        except nx.NetworkXNoPath:
            continue
        # delete the query embedding from the graph
        G.remove_node(query_idx)

    # cache the data
    with open(f"data/{dataset}/l2_distances.pkl", "wb") as f:
        pickle.dump(l2_distances, f)
    with open(f"data/{dataset}/manifold_distances.pkl", "wb") as f:
        pickle.dump(manifold_distances, f)

    # load the data
    with open(f"data/{dataset}/l2_distances.pkl", "rb") as f:
        l2_distances = pickle.load(f)
    with open(f"data/{dataset}/manifold_distances.pkl", "rb") as f:
        manifold_distances = pickle.load(f) 

    # Apply Seaborn style for a modern aesthetic
    sns.set(style='whitegrid', font_scale=1.2, palette='muted')

    plt.rcParams['font.family'] = 'Times New Roman'

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    # Add the 45-degree line
    min_val = min(min(l2_distances), min(manifold_distances))
    max_val = max(max(l2_distances), max(manifold_distances))
    plt.plot([min_val, max_val], [min_val, max_val], color='darkred', linestyle='--', linewidth=2, label='Identity line', zorder=1)

    plt.scatter(l2_distances, manifold_distances, alpha=0.7, s=15, edgecolors='k', color='royalblue', label='Data Points', zorder=2)

    # Titles and labels with professional formatting
    plt.title(f'Relationship between L2 Distance and Manifold Distance - {dataset}', fontsize=16, fontweight='bold')
    plt.xlabel('L2 Distance', fontsize=14)
    plt.ylabel('Manifold Distance', fontsize=14)

    # Add a grid with minor ticks
    plt.grid(True, which='both', linestyle=':', linewidth=0.6)

    # Legend customization
    plt.legend(fontsize=12, loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

