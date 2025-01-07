import networkx as nx
import json
import pickle
import torch
import matplotlib.pyplot as plt
from heapq import heappush, heappop

dataset = "nfcorpus"
k_neighbors = 6

def read_graph(file_path):
    with open(file_path, 'r') as f:
        G_New = json.load(f)
    G = nx.Graph()
    for node in G_New:
        for neighbor, weight in G_New[node].items():
            G.add_edge(int(node), int(neighbor), weight=float(weight))
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
    file_path = f"data/{dataset}/graph_root_{k_neighbors}.json"
    G = read_graph(file_path)

    # read the embeddings
    with open(f"data/{dataset}/tas-b-query_embeddings.pkl", "rb") as f:
        query_embeddings = pickle.load(f)

    with open(f"data/{dataset}/tas-b-doc_embeddings.pkl", "rb") as f:
        passage_embeddings = pickle.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for manifold distance retrieval")

    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
    all_distances, all_indices = torch.topk(l2_distance_matrix, k_neighbors, dim=1, largest=False, sorted=True)

    G.add_node('Q') # add query node
        
    distances = all_distances[0].cpu().numpy().flatten()
    indices = all_indices[0].cpu().numpy().flatten()

    for neighbor in range(k_neighbors):
        G.add_edge('Q', indices[neighbor], weight=distances[neighbor])

   

    # only keep 50 nodes near Q
    G = nx.subgraph(G, dijkstra_shortest_path(G=G, query_idx='Q', top_k=50))
    # for weight only keep 3 digits after the decimal point
    edge_labels=dict([((u,v,),round(d['weight'], 3)) for u,v,d in G.edges(data=True)])

    pos = nx.spring_layout(G)
   
    # Draw edges first
    nx.draw_networkx_edges(G, pos, width=0.2)
    
    # Draw regular nodes
    regular_nodes = [node for node in G.nodes() if node != 'Q']
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_color='lightblue', 
                            node_size=10)
    
    # Draw Q node as a star
    nx.draw_networkx_nodes(G, pos, nodelist=['Q'], node_color='red', 
                            node_size=300, node_shape='*')
    
    # Add labels
    nx.draw_networkx_labels(G, pos, 
                            {node: node if node == 'Q' else '' for node in G.nodes()},
                            font_size=8, font_weight='bold')
    
    # Add edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
    
    plt.axis('off')
    plt.show()

    
