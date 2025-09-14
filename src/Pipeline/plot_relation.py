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


def get_distances(query_embeddings, passage_embeddings, k, question_ids, passage_ids, relevance_map):
    G = construct_graph(passage_embeddings, k)

    l2_distances = []
    manifold_distances = []
    is_relevant = []

    # construct a document id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(passage_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for constructing graph")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    for i in tqdm(range(len(query_embeddings)), desc="Processing queries"):
        try:
            # Get L2 distances for current query to all passages
            l2_distance_matrix = torch.cdist(query_embeddings[i].reshape(1, -1), passage_embeddings, p=2)
            current_l2_distances = l2_distance_matrix[0].cpu().numpy()

            # Add query node to graph temporarily
            query_idx = i  # Use query index as the node ID
            G.add_node(query_idx)
            
            # Add edges to k-nearest neighbors
            distances, indices = torch.topk(l2_distance_matrix[0], k, largest=False)
            for j, (idx, dist) in enumerate(zip(indices.cpu().numpy(), distances.cpu().numpy())):
                if idx < len(passage_embeddings):  # Ensure index is valid
                    G.add_edge(query_idx, idx, weight=dist)

            # Get manifold distances using improved Dijkstra implementation
            manifold_dict = dijkstra_shortest_path(G, query_idx, top_k=len(passage_embeddings))
            
            # Remove query node from results
            if query_idx in manifold_dict:
                manifold_dict.pop(query_idx)
            
            # Store distances for all passages
            query_id = question_ids[i]
            relevant_docs = relevance_map.get(query_id, [])  # Use get() with default empty list
            relevant_docs_idx = []
            for doc_id in relevant_docs:
                if doc_id in doc_id_to_idx:  # Only include if doc_id exists in mapping
                    relevant_docs_idx.append(doc_id_to_idx[doc_id])

            for node, man_dist in manifold_dict.items():
                if node < len(passage_embeddings):  # Ensure node index is valid
                    l2_distances.append(float(current_l2_distances[node]))
                    manifold_distances.append(man_dist)
                    is_relevant.append(node in relevant_docs_idx)

        except Exception as e:
            print(f"Error processing query {i}: {str(e)}")
            continue
            
        # Remove query node from graph
        G.remove_node(query_idx)

    return l2_distances, manifold_distances, is_relevant

def get_rankings(query_embeddings, passage_embeddings, k, question_ids, passage_ids, relevance_map):
    G = construct_graph(passage_embeddings, k)

    all_l2_rankings = []
    all_manifold_rankings = []
    is_relevant = []  # Track which points are relevant
    
    # construct a document id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(passage_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for constructing graph")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    for i in tqdm(range(len(query_embeddings)), desc="Processing queries"):
        # Get L2 distances for current query to all passages
        l2_distance_matrix = torch.cdist(query_embeddings[i].reshape(1, -1), passage_embeddings, p=2)
        l2_distances = l2_distance_matrix[0].cpu().numpy()

        # Get L2 rankings (1-based)
        l2_rankings = np.argsort(l2_distances).argsort() + 1

        # Add query node to graph temporarily
        query_idx = len(G.nodes)
        G.add_node(query_idx)
        
        # Add edges to k-nearest neighbors
        distances, indices = torch.topk(l2_distance_matrix[0], k, largest=False)
        for j, (idx, dist) in enumerate(zip(indices.cpu().numpy(), distances.cpu().numpy())):
            G.add_edge(query_idx, idx, weight=dist)

        # Get manifold distances using Dijkstra
        try:
            manifold_distances = dijkstra_shortest_path(G, query_idx, len(G.nodes))
           
            # Convert manifold distances to rankings
            manifold_dist_array = np.full(len(passage_embeddings), np.inf)
            for node, dist in manifold_distances.items():
                if node != query_idx:
                    manifold_dist_array[node] = dist
            
            # Get manifold rankings (1-based)
            manifold_rankings = np.argsort(manifold_dist_array).argsort() + 1


            # Store rankings for all passages
            all_l2_rankings.extend(l2_rankings)
            all_manifold_rankings.extend(manifold_rankings)
            
            # Check relevance for each passage
            query_id = question_ids[i]
            relevant_docs = relevance_map[query_id]
            # find the index of relevant_docs in passage_ids
            relevant_docs_idx = [doc_id_to_idx[doc_id] for doc_id in relevant_docs]
            is_relevant.extend([j in relevant_docs_idx for j in range(len(passage_embeddings))])
        except nx.NetworkXNoPath:
            print(f"No path found for query {i}")
            continue
            
        # Remove query node from graph
        G.remove_node(query_idx)

    return all_l2_rankings, all_manifold_rankings, is_relevant

if __name__ == "__main__":
    k = 10
    mode = "connectivity"
    available_datasets = {"scidocs": Scidocs, "msmarco": MSMARCO, "antique": Antique, "NFCorpus": NFCorpus}
    dataset_name = "scidocs"


    dataset = available_datasets[dataset_name]()
    question_ids, question_texts, passage_ids, passage_texts, relevance_map = dataset.load_data()

    with open(f"data/{dataset_name}/tas-b-query_embeddings.pkl", "rb") as f:
        query_embeddings = pickle.load(f)

    with open(f"data/{dataset_name}/tas-b-doc_embeddings.pkl", "rb") as f:
        passage_embeddings = pickle.load(f)

    # sample 20000 passage
    sample_passage_idx = np.random.choice(len(passage_embeddings), min(20000, len(passage_embeddings)), replace=False)
    passage_embeddings = passage_embeddings[sample_passage_idx]

    sample_query_idx = np.random.choice(len(query_embeddings), min(1000, len(query_embeddings)), replace=False)
    query_embeddings = query_embeddings[sample_query_idx]

    # update the question_ids and passage_ids
    question_ids = [question_ids[i] for i in sample_query_idx]
    passage_ids = [passage_ids[i] for i in sample_passage_idx]

    # update the relevance_map, and keep only the documents that are sampled
    relevance_map = {question_ids[i]: relevance_map[question_ids[i]] for i in range(len(question_ids))}
    for query_id in relevance_map:
        relevance_map[query_id] = [doc_id for doc_id in relevance_map[query_id] if doc_id in passage_ids]


    l2_distances, manifold_distances, is_relevant = get_distances(query_embeddings, passage_embeddings, k, question_ids, passage_ids, relevance_map)
    # all_l2_rankings, all_manifold_rankings, is_relevant = get_rankings(query_embeddings, passage_embeddings, k)
    

    

    ############Plot rankings##########
    # # Convert to numpy arrays
    # all_l2_rankings = np.array(all_l2_rankings)
    # all_manifold_rankings = np.array(all_manifold_rankings)
    # is_relevant = np.array(is_relevant)

    # # Filter to only keep relevant documents
    # relevant_mask = is_relevant
    # all_l2_rankings = all_l2_rankings[relevant_mask]
    # all_manifold_rankings = all_manifold_rankings[relevant_mask]


    # # only keep ranks less than 100 in both rankings, keep index aligned
    # threshold = 100
    # l2_chosen_idx = np.where(all_l2_rankings < threshold)[0]
    # print(f"l2_chosen_idx: {len(l2_chosen_idx)}")
    # manifold_chosen_idx = np.where(all_manifold_rankings < threshold)[0]
    # print(f"manifold_chosen_idx: {len(manifold_chosen_idx)}")
    # # find the intersection of the two indices  
    # common_idx = np.intersect1d(l2_chosen_idx, manifold_chosen_idx)
    # all_l2_rankings = all_l2_rankings[common_idx]
    # all_manifold_rankings = all_manifold_rankings[common_idx]

    # # # Random sample if too many points
    # # if len(all_l2_rankings) > 200:
    # #     sample_idx = np.random.choice(len(all_l2_rankings), 200, replace=False)
    # #     all_l2_rankings = all_l2_rankings[sample_idx]
    # #     all_manifold_rankings = all_manifold_rankings[sample_idx]
    # #     is_relevant = is_relevant[sample_idx]

    # with plt.style.context(['science', 'scatter', 'no-latex']):     
    #     fig, ax = plt.subplots()
        
    #     # Plot identity line
    #     min_val = 1
    #     max_val = threshold  # Maximum possible rank
    #     ax.plot([min_val, max_val], [min_val, max_val], 
    #             'k--', label='Perfect alignment', linewidth=1, zorder=2, color='#E5306E')
        
    #     # Plot only relevant points
    #     ax.scatter(all_l2_rankings, all_manifold_rankings,
    #               s=1, alpha=0.6, color='#39AF84', 
    #               label='Relevant', zorder=4)
        
    #     # ax.legend(fontsize=8)
        
    #     ax.set_xlabel('Rank by Euclidean Distance', fontsize=8)
    #     ax.set_ylabel('Rank by Manifold Distance', fontsize=8)
    #     # ax.set_title(f'Relevant Documents Ranking Comparison on {dataset_name}', 
    #     #             fontsize=8, fontweight='bold')
        
    #     ax.set_xlim([min_val, max_val])
    #     ax.set_ylim([min_val, max_val])
        
    #     ax.autoscale(tight=True)
    #     fig.savefig(f'{mode}_{dataset_name}_rankings_relevant_only.pdf')
    #     fig.savefig(f'{mode}_{dataset_name}_rankings_relevant_only.jpg', dpi=300)



    ############Plot distances##########
    # cache the data
    # with open(f"data/{dataset_name}/l2_distances.pkl", "wb") as f:
    #     pickle.dump(l2_distances, f)
    # with open(f"data/{dataset_name}/manifold_distances.pkl", "wb") as f:
    #     pickle.dump(manifold_distances, f)

    # # load the data
    # with open(f"data/{dataset_name}/l2_distances.pkl", "rb") as f:
    #     l2_distances = pickle.load(f)
    # with open(f"data/{dataset_name}/manifold_distances.pkl", "rb") as f:
    #     manifold_distances = pickle.load(f)

    # filter out the data points that are not relevant and rename the variables
    relevant_l2_distances = [l2_distances[i] for i in range(len(l2_distances)) if is_relevant[i]]
    relevant_manifold_distances = [manifold_distances[i] for i in range(len(manifold_distances)) if is_relevant[i]]
    
    irrelevant_l2_distances = [l2_distances[i] for i in range(len(l2_distances)) if not is_relevant[i]]
    irrelevant_manifold_distances = [manifold_distances[i] for i in range(len(manifold_distances)) if not is_relevant[i]]

    # random sample 19500 data points for irrelevant and 500 for relevant

    sample_idx = np.random.choice(len(irrelevant_l2_distances), 2500, replace=False)
    irrelevant_l2_distances = np.array(irrelevant_l2_distances)[sample_idx]
    irrelevant_manifold_distances = np.array(irrelevant_manifold_distances)[sample_idx]
    
    sample_idx = np.random.choice(len(relevant_l2_distances), min(500, len(relevant_l2_distances)), replace=False)
    relevant_l2_distances = np.array(relevant_l2_distances)[sample_idx]
    relevant_manifold_distances = np.array(relevant_manifold_distances)[sample_idx]

    # delete points that l2 > manifold
    l2_greater_manifold_idx_relevant = np.where(relevant_l2_distances > relevant_manifold_distances)[0]
    relevant_l2_distances = np.delete(relevant_l2_distances, l2_greater_manifold_idx_relevant)
    relevant_manifold_distances = np.delete(relevant_manifold_distances, l2_greater_manifold_idx_relevant)

    # remove points that manifold > 15
    # manifold_greater_15_idx_relevant = np.where(relevant_manifold_distances > 15)[0]
    # relevant_l2_distances = np.delete(relevant_l2_distances, manifold_greater_15_idx_relevant)
    # relevant_manifold_distances = np.delete(relevant_manifold_distances, manifold_greater_15_idx_relevant)

    # l2_greater_manifold_idx_irrelevant = np.where(irrelevant_l2_distances > irrelevant_manifold_distances)[0]
    # irrelevant_l2_distances = np.delete(irrelevant_l2_distances, l2_greater_manifold_idx_irrelevant)
    # irrelevant_manifold_distances = np.delete(irrelevant_manifold_distances, l2_greater_manifold_idx_irrelevant)

    
    with plt.style.context(['science', 'scatter', 'no-latex']):     
        fig, ax = plt.subplots()
        
        # Plot 45 degree line
        # min_val = min(min(irrelevant_l2_distances), min(irrelevant_manifold_distances), min(relevant_l2_distances), min(relevant_manifold_distances))
        # max_val = max(max(irrelevant_l2_distances), max(irrelevant_manifold_distances), max(relevant_l2_distances), max(relevant_manifold_distances))

        min_val = min(min(min(relevant_l2_distances), min(irrelevant_l2_distances)), min(min(relevant_manifold_distances), min(irrelevant_manifold_distances)))
        max_val = max(max(max(relevant_l2_distances), max(irrelevant_l2_distances)), max(max(relevant_manifold_distances), max(irrelevant_manifold_distances)))

        ax.plot([min_val, max_val], [min_val, max_val], 
                'k--', linewidth=1, zorder=2, color='#999999')

        # Plot data points with better visibility
        ax.scatter(irrelevant_l2_distances, irrelevant_manifold_distances, 
                    s=0.1,  # tiny point size
                    alpha=0.7,  # high transparency
                    color='#ADDD8E',
                    label='Irrelevant',
                    zorder=3)
        
        ax.scatter(relevant_l2_distances, relevant_manifold_distances, 
                    s=0.5,  # tiny point size
                    alpha=0.3,  # low transparency to distinguish from irrelevant points
                    color='#feb24c',
                    label='Relevant',
                    zorder=4)

        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Improve legend
        # ax.legend(fontsize=6, markerscale=5, handletextpad=0.1, loc='lower right')
        
        # Improve labels
        ax.set_xlabel('Euclidean Distance', fontsize=20)
        ax.set_ylabel('Manifold Distance', fontsize=20)
        # ax.set_title(f'Comparison of Euclidean and Manifold Distances on {dataset}', fontsize=8, fontweight='bold')
        

        # Set limits and adjust ticks
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        
        ax.autoscale(tight=True)
        fig.savefig(f'{dataset_name}_distances.pdf')
        fig.savefig(f'{dataset_name}_distances.jpg', dpi=300)
        print(f"Saved {dataset_name}_distances.pdf and {dataset_name}_distances.jpg")

