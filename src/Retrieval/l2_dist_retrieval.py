import torch
import numpy as np

def retrieve_k_l2(query_embeddings: np.ndarray, passage_embeddings: np.ndarray, k=100) -> np.ndarray:
    """
    Retrieve top-k passages for each query using L2 distance and return the indices of the top-k passages.

    We assume that the embeddings are already normalized.

    :param query_embeddings: numpy array of shape (num_queries, embedding_dim)
    :param passage_embeddings: numpy array of shape (num_passages, embedding_dim)
    :param k: Number of passages to retrieve for each query
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

