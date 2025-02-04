# load embeddings
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
# should do "pip install SciencePlots"
import scienceplots
from heapq import heappush, heappop
import seaborn as sns
import json
from sklearn.neighbors import NearestNeighbors
from src.Dataset.msmarco import MSMARCO
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.antique import Antique
from src.Dataset.scidocs import Scidocs

def load_embeddings(query_embeddings_path, passage_embeddings_path):
    with open(query_embeddings_path, "rb") as f:
        query_embeddings = pickle.load(f)
        assert isinstance(query_embeddings, np.ndarray), "query_embeddings should be a numpy array"

    with open(passage_embeddings_path, "rb") as f:
        passage_embeddings = pickle.load(f)
        assert isinstance(passage_embeddings, np.ndarray), "passage_embeddings should be a numpy array"

    return query_embeddings, passage_embeddings

def nearest_neighbors(passages_embeddings, k):
    
    # Initialize nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k+1, metric='l2', n_jobs=-1)
    nn.fit(passages_embeddings)
    
    # Find k+1 nearest neighbors (including self)
    distances, indices = nn.kneighbors(passages_embeddings)
    
    doc_nearest_neighbors_indices = []
    doc_nearest_neighbors_distances = []
    
    # Process results to exclude self-connections
    for idx in tqdm(range(len(passages_embeddings)), desc="Processing nearest neighbors"):
        # Remove self-reference (always at index 0 since it's the closest)
        neighbor_indices = indices[idx][1:]  # Skip first element (self)
        neighbor_distances = distances[idx][1:]  # Skip first element (self)
        
        doc_nearest_neighbors_indices.append(neighbor_indices)
        doc_nearest_neighbors_distances.append(neighbor_distances)
    
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
    Ensures that manifold distances are never smaller than direct L2 distances.
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
            
        # Store the distance only if it's the first time we're visiting this node
        if current not in visited:
            visited[current] = current_dist
        
        # Explore neighbors
        for neighbor in G[current]:
            if neighbor in visited:
                continue
                
            weight = G[current][neighbor]['weight']
            # Calculate the direct L2 distance between query and neighbor
            direct_l2_dist = None
            if query_idx in G and neighbor in G[query_idx]:
                direct_l2_dist = G[query_idx][neighbor]['weight']
            
            # Calculate the manifold distance through current path
            manifold_dist = current_dist + weight
            
            # If we have a direct L2 distance, ensure manifold distance isn't smaller
            if direct_l2_dist is not None:
                manifold_dist = max(manifold_dist, direct_l2_dist)
            
            if manifold_dist < distances[neighbor]:
                distances[neighbor] = manifold_dist
                heappush(pq, (manifold_dist, neighbor))
    
    return visited


def get_distances(passage_embeddings, k):
    G = construct_graph(passage_embeddings, k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for constructing graph")
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    # Randomly choose a start passage
    start_idx = np.random.randint(0, len(passage_embeddings))
    start_embedding = passage_embeddings[start_idx].unsqueeze(0)

    # Calculate L2 distances from start to all passages
    l2_distance_matrix = torch.cdist(start_embedding, passage_embeddings, p=2)
    l2_distances = l2_distance_matrix[0].cpu().numpy().tolist()

    # Calculate manifold distances from start to all passages
    manifold_distances_dict = dijkstra_shortest_path(G, start_idx, len(passage_embeddings))
    
    # Initialize manifold distances array with infinity for any unreached nodes
    manifold_distances = [float('infinity')] * len(passage_embeddings)
    for node_idx, distance in manifold_distances_dict.items():
        manifold_distances[node_idx] = distance

    return l2_distances, manifold_distances, start_idx 

if __name__ == "__main__":
    k = 300
    available_datasets = {"scidocs": Scidocs, "msmarco": MSMARCO, "antique": Antique, "NFCorpus": NFCorpus}
    dataset_name = "NFCorpus"

    dataset = available_datasets[dataset_name]()

    question_ids, question_texts, passage_ids, passage_texts, relevance_map = dataset.load_data()

    # only use the passages that are relevant to the queries
    with open(f"data/{dataset_name}/tas-b-doc_embeddings.pkl", "rb") as f:
        passage_embeddings = pickle.load(f)

    # only use the passages that are relevant to the queries
    relevant_passage_idx = set()

    for query_id in relevance_map:
        for doc_id in relevance_map[query_id]:
            relevant_passage_idx.add(doc_id)

    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(passage_ids)}
    relevant_passage_idx = [doc_id_to_idx[doc_id] for doc_id in relevant_passage_idx]
    passage_embeddings = np.array(passage_embeddings)
    passage_embeddings = passage_embeddings[relevant_passage_idx]


    l2_distances, manifold_distances, start_idx = get_distances(passage_embeddings, k)

    # remove the start passage from the l2 distances and manifold distances
    l2_distances = np.delete(l2_distances, start_idx)
    manifold_distances = np.delete(manifold_distances, start_idx)

    # calculate the percentage of the l2 distance == manifold distance, add it to legend
    percentage = np.sum(np.isclose(l2_distances, manifold_distances)) / len(l2_distances)
    print(f"Percentage of l2 distance == manifold distance: {percentage:.2f}")

    with plt.style.context(['science', 'scatter', 'no-latex']):     
        fig, ax = plt.subplots()
        
        # Plot 45 degree line
        min_val = min(min(l2_distances), min(manifold_distances))
        max_val = max(max(l2_distances), max(manifold_distances))

        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=1, zorder=2, color='#999999')
        
        
        ax.scatter(l2_distances, manifold_distances, 
                    s=0.1,  # tiny point size
                    alpha=0.4,  # low transparency to distinguish from irrelevant points
                    color='#496C88',
                    label='Relevant',
                    zorder=4)

        ax.tick_params(axis='both', which='major', labelsize=6)
        
        # Improve legend, and add the percentage to the legend
        ax.legend(fontsize=6, markerscale=5, handletextpad=0.1, loc='lower right', title=f"Percentage: {percentage:.2f}")
        
        # Improve labels
        ax.set_xlabel('Euclidean Distance', fontsize=8)
        ax.set_ylabel('Manifold Distance', fontsize=8)
        # ax.set_title(f'Comparison of Euclidean and Manifold Distances on {dataset}', fontsize=8, fontweight='bold')
        
        # Set limits and adjust ticks
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        
        ax.autoscale(tight=True)
        fig.savefig(f'{dataset_name}_distances.pdf')
        fig.savefig(f'{dataset_name}_distances.jpg', dpi=300)
        print(f"Saved {dataset_name}_distances.pdf and {dataset_name}_distances.jpg")