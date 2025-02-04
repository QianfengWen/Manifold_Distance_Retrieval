import pickle
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from scipy import sparse
import time
from sklearn.neighbors import kneighbors_graph
import pdb

def create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path):
    embedder = SentenceTransformer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder.to(device)

    query_embeddings = embedder.encode(query_texts, convert_to_tensor=False, show_progress_bar=True)
    passages_embeddings = embedder.encode(passage_texts, convert_to_tensor=False, show_progress_bar=True)

    save_embeddings(query_embeddings, passages_embeddings, query_embeddings_path, passage_embeddings_path)
   
    return query_embeddings, passages_embeddings


def save_embeddings(query_embeddings, passage_embeddings=None, query_embeddings_path=None, passage_embeddings_path=None):
    os.makedirs(os.path.dirname(query_embeddings_path), exist_ok=True)
    if passage_embeddings_path is not None:
        os.makedirs(os.path.dirname(passage_embeddings_path), exist_ok=True)
    with open(query_embeddings_path, "wb") as f:
        pickle.dump(query_embeddings, f)

    if passage_embeddings is not None:
        with open(passage_embeddings_path, "wb") as f:
            pickle.dump(passage_embeddings, f)
    
    return


def load_embeddings(query_embeddings_path, passage_embeddings_path):
    with open(query_embeddings_path, "rb") as f:
        query_embeddings = pickle.load(f)
        assert isinstance(query_embeddings, np.ndarray), "query_embeddings should be a numpy array"

    with open(passage_embeddings_path, "rb") as f:
        passage_embeddings = pickle.load(f)
        assert isinstance(passage_embeddings, np.ndarray), "passage_embeddings should be a numpy array"
    
    # passage_embeddings = passage_embeddings[:100]
    return query_embeddings, passage_embeddings

def save_adjacency_matrix(adjacency_matrix, file_path):
    print("Saving adjacency matrix to", file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(adjacency_matrix, f)
    return

def read_adjacency_matrix(file_path):
    with open(file_path, 'rb') as f:
        adjacency_matrix = pickle.load(f)
    print("Loading adjacency matrix from", file_path)
    return adjacency_matrix

def create_spectral_embedding(embeddings, n_components, k, file_path, all_spectral_embeddings_path):
    """
    Creates spectral embeddings from input embeddings by constructing a k-nearest neighbor graph,
    """
    file_path = file_path.replace(".pkl", "_adjacency_matrix.pkl").replace(f"_spectral_n_components={n_components}", "_l2")
    if os.path.exists(file_path):
        print("Creating spectral embeddings using cached adjacency matrix ...")
        affinity_matrix = read_adjacency_matrix(file_path)
    else:
        print("Creating spectral embeddings from scratch ...")
        start = time.time()
        affinity_matrix = kneighbors_graph(
            embeddings, 
            n_neighbors=k,
            mode='distance',
            include_self=True
        )
        end = time.time()
        print("Finished creating affinity matrix, it takes", end-start, "seconds")
        save_adjacency_matrix(affinity_matrix, file_path)
    
    # Make matrix symmetric
    affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

    print("Computing normalized Laplacian matrix...")
    
    # Convert affinity matrix to dense PyTorch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.from_numpy(affinity_matrix.toarray() if sparse.issparse(affinity_matrix) else affinity_matrix).float().to(device)
    
    # Compute degree matrix
    D = torch.diag(A.sum(dim=1))
    
    # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    D_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(D) + 1e-8))  # Add small epsilon for numerical stability
    L = torch.eye(A.shape[0], device=device) - D_sqrt_inv @ A @ D_sqrt_inv
    
    print(f"Computing eigenvectors on {device}...")
    print("L shape:", L.shape)
    # Compute eigenvectors and eigenvalues using PyTorch
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Sort eigenvectors by eigenvalues in ascending order
    idx = torch.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # Select the specified number of components
    # Skip the first eigenvector (constant vector) as per spectral embedding convention
    embedding = eigenvectors[:, 1:n_components+1]
    
    # Normalize the embedding using torch 
    embedding = torch.nn.functional.normalize(embedding, dim=0)

    embedding = embedding.cpu().numpy()

    save_embeddings(query_embeddings=embedding, query_embeddings_path=all_spectral_embeddings_path)
    
    print("Finished computing spectral embedding")
    return embedding

def slice_embedding(embedding, n_components):
    """
    Slices the embedding to n_components
    embedding: numpy array of shape (num_samples, embedding_dim) or (batch, num_samples, embedding_dim)
    n_components: int, number of components to slice
    """
    return embedding[:, :n_components]
