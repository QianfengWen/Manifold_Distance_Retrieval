import pickle
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer


def create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path):
    embedder = SentenceTransformer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder.to(device)

    query_embeddings = embedder.encode(query_texts, convert_to_tensor=False, show_progress_bar=True)
    passages_embeddings = embedder.encode(passage_texts, convert_to_tensor=False, show_progress_bar=True)

    save_embeddings(query_embeddings, passages_embeddings, query_embeddings_path, passage_embeddings_path)
   
    return query_embeddings, passages_embeddings


def save_embeddings(query_embeddings, passage_embeddings, query_embeddings_path, passage_embeddings_path):
    os.makedirs(os.path.dirname(query_embeddings_path), exist_ok=True)
    os.makedirs(os.path.dirname(passage_embeddings_path), exist_ok=True)
    with open(query_embeddings_path, "wb") as f:
        pickle.dump(query_embeddings, f)

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