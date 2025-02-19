import torch
import numpy as np

def retrieve_k(query_embeddings: np.ndarray, distance_type: str, passage_embeddings: np.ndarray, k=100) -> np.ndarray:
    """
    Retrieve top-k passages for each query using L2 distance and return the indices of the top-k passages.

    We assume that the embeddings are already normalized.

    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param k: Number of passages to retrieve for each query
    :return: numpy array of shape (num_queries, k) containing the indices of the top-k passages for each query
    """
    print(f"Retrieving top-{k} passages using {distance_type} distance")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Embedding dimensions of queries and passages do not match")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {distance_type} distance retrieval")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    if distance_type == "euclidean":
        l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
        _, indices = torch.topk(l2_distance_matrix, k, dim=1, largest=False, sorted=True)
        
    return indices.detach().cpu().numpy()