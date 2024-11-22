import networkx as nx
from tqdm import tqdm
import numpy as np
import torch
import json

def construct_graph(passages_embeddings, k, file_path):
    doc_nearest_neighbors_indices = []
    doc_nearest_neighbors_embeddings = []
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
        indices = torch.cat((indices[0][0:self_index], indices[0][self_index+1:]))
        distances = torch.cat((distances[0][0:self_index], distances[0][self_index+1:]))

        doc_nearest_neighbors_indices.append(indices.flatten())
        doc_nearest_neighbors_distances.append(distances.flatten())
        doc_nearest_neighbors_embeddings.append(passages_embeddings[indices.flatten()])


    # create a graph
    G = nx.Graph()
    for i in range(len(passages_embeddings)):
        for j in range(k):
            G.add_edge(i, doc_nearest_neighbors_indices[i][j], weight=doc_nearest_neighbors_distances[i][j]) # add an weighted edge that connects i-th query (i) and its j-th nearest neighbor passage, the weight is the distance between them
    # save the graph
    save_graph(G, file_path)
    return G


def save_graph(G, file_path):
    G_New = {}
    for edge in G.edges(data=True):
        weight = str(edge[2]['weight'])
        node1 = str(edge[0])
        node2 = str(edge[1])  
        try:
            G_New[node1].update({node2: weight})
        except KeyError:
            G_New[node1] = {node2: weight}
        try:
            G_New[node2].update({node1: weight})
        except KeyError:
            G_New[node2] = {node1: weight}

    with open(file_path, 'w') as f:
        json.dump(G_New, f, indent=4)

    return
