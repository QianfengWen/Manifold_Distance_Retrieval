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
    assert distance_type in ["l2", "cosine"], "Distance type must be either 'l2' or 'cosine'"
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Embedding dimensions of queries and passages do not match")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {distance_type} distance retrieval")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    if distance_type == "l2":
        l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
        _, indices = torch.topk(l2_distance_matrix, k, dim=1, largest=False, sorted=True)
    
    elif distance_type == "cosine":
        # cosine distance is the same as 1 - cosine similarity
        cosine_similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) / (torch.norm(query_embeddings, dim=1) * torch.norm(passage_embeddings, dim=1))
        cosine_distance_matrix = 1 - cosine_similarity_matrix
        _, indices = torch.topk(cosine_distance_matrix, k, dim=1, largest=False, sorted=True)
    
    return indices.detach().cpu().numpy()

def retrieve_k_l2_reciprocal(query_embeddings: np.ndarray, passage_embeddings: np.ndarray, k=100, N=10) -> np.ndarray:
    """
    Retrieve top-k passages for each query using L2 distance and return the indices of the top-k passages.

    We assume that the embeddings are already normalized.

    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param k: Number of passages to retrieve for each query
    :param N: Number of neighbors to consider for each passage
    :return: numpy array of shape (num_queries, k) containing the indices of the top-k passages for each query
    """
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("Embedding dimensions of queries and passages do not match")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for l2 distance retrieval")
    query_embeddings = torch.tensor(query_embeddings, device=device)
    passage_embeddings = torch.tensor(passage_embeddings, device=device)

    l2_distance_matrix = torch.cdist(query_embeddings, passage_embeddings, p=2)
    _, indices = torch.topk(l2_distance_matrix, k, dim=1, largest=False, sorted=True)

    
    
    return indices.detach().cpu().numpy()

