import numpy as np
from collections import defaultdict
import json, os, pickle
from src.Retrieval.l2_dist_retrieval import retrieve_k
from src.Retrieval.manifold_dist_retrieval import retrieve_k_manifold_baseline
from src.Evaluation.evaluation import evaluate
from src.Graph.graph import construct_graph, read_graph
from src.Embedding.embedding import create_embeddings, load_embeddings
import torch
from scipy import sparse
import time
import pdb
import copy

class Pipeline:
    def __init__(self, experiment_name, **kwargs):
        experiment_path = os.path.join("results", experiment_name)
        self.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)    

        try:
            self.dataloader = kwargs["dataloader"]

            self.model_name = kwargs.get("model_name", None)
            self.query_embeddings_path = kwargs["query_embeddings_path"]
            self.passage_embeddings_path = kwargs["passage_embeddings_path"]

            self.experiment_type = kwargs["experiment_type"]

            # graph
            self.create_new_graph = kwargs["create_new_graph"]
            self.graph_path = kwargs["graph_path"]
            self.distance = kwargs["distance"] # l2, cos distance 
            self.mode = kwargs["mode"] # connectivity (hop) vs. distance
            self.use_spectral_decomposition = kwargs.get("use_spectral_decomposition", False)
            self.query_projection = kwargs.get("query_projection", False)
            self.eigenvectors_path = kwargs.get("eigenvectors_path", None)
            self.n_components = kwargs.get("n_components", None) # for spectral only
            self.max_edges = kwargs.get("max_edges", None) # for connected only
            self.max_percentage = kwargs.get("max_percentage", None) # for connected only
            self.k_neighbours = kwargs["k_neighbours"]

            self.evaluation_functions = kwargs["evaluation_functions"]
            self.k_list = kwargs["k_list"]
        
        except KeyError as e:
            print(f"Missing argument: {e}")
            raise
    
    def run_pipeline(self, with_cache=False):
        question_ids, question_texts, passage_ids, passage_texts, relevance_map = self.load_data(self.dataloader)
        
        if not with_cache:        
            query_embeddings, passage_embeddings = self.handle_embeddings(self.model_name, self.query_embeddings_path, self.passage_embeddings_path, question_texts, passage_texts)  

            if self.experiment_type == "manifold":
                print("********************* Handling Graph *********************")
                G, query_embeddings, passage_embeddings = self.handle_graph(query_embeddings, passage_embeddings, self.k_neighbours, self.graph_path, self.eigenvectors_path, self.distance, self.n_components, self.use_spectral_decomposition, self.query_projection)
                print("********************* Running Evaluation *********************")
                self.run_evaluation(self.experiment_type, self.k_list, self.evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G, self.k_neighbours, self.distance)

            # baseline
            else:
                print("********************* Running Baseline Evaluation *********************")
                self.run_evaluation("baseline", self.k_list, self.evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, None, None, self.distance)
        else:
            print("********************* Running Evaluation with Cache *********************")
            self.run_evaluation_with_cache(self.k_list, self.evaluation_functions, question_ids, passage_ids, relevance_map)


    def load_data(self, dataloader):
        return dataloader.load_data()


    def handle_embeddings(self, model_name, query_embeddings_path, passage_embeddings_path, query_texts, passage_texts):
        if model_name:
            return create_embeddings(model_name, query_texts, passage_texts, query_embeddings_path, passage_embeddings_path)
        else:
            return load_embeddings(query_embeddings_path, passage_embeddings_path)
        
    def out_of_sample_barycentric(self, queries_emb, passages_emb, passages_spectral):
        """
        Heuristic approach: For each query, compute its weights W(q, p_i)
        and then do a weighted average of the passage spectral embeddings.
        
        queries_emb: (M, d)
        passages_emb: (N, d)
        passages_spectral: (N, k)
        
        Returns: queries_spectral: (M, k)
        """
        # change to torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        queries_emb = torch.from_numpy(queries_emb).float().to(device)
        passages_emb = torch.from_numpy(passages_emb).float().to(device)
        if type(passages_spectral) == np.ndarray:
            passages_spectral = torch.from_numpy(passages_spectral).float().to(device)
        else:
            passages_spectral = passages_spectral.to(device)    




        M = queries_emb.shape[0]
        N = passages_emb.shape[0]
        k = passages_spectral.shape[1]


        # Precompute norms of passages
        norms_passages = torch.sqrt((passages_emb**2).sum(dim=1, keepdim=True))
        queries_spectral = torch.zeros((M, k), dtype=torch.float32)


        for m in range(M):
            q = queries_emb[m]   # shape (d,)
            q_norm = torch.sqrt((q**2).sum())


            # Compute weight W(q, p_i) = cos(q, p_i)
            # shape (N,)
            dot_products = q @ passages_emb.T
            denom = q_norm * norms_passages.squeeze()
            w = dot_products / denom  # shape (N,)

            # Weighted average in spectral space
            w_sum = w.sum()
            if abs(w_sum) < 1e-9:
                # Avoid division by zero if all similarities are extremely small
                # or if query is zero vector.
                # Fallback to all-zeros or random?
                queries_spectral[m] = 0.0
            else:
                weighted_sum = (w.reshape(-1, 1) * passages_spectral).sum(dim=0)
                queries_spectral[m] = weighted_sum / w_sum
                

        return queries_spectral
    

    def handle_graph(self, query_embeddings, passage_embeddings, k_neighbours, graph_path, eigenvectors_path, distance, n_components, use_spectral_decomposition, query_projection):
        original_query_embeddings = copy.deepcopy(query_embeddings)
        original_passage_embeddings = copy.deepcopy(passage_embeddings)
        if self.create_new_graph and not os.path.exists(graph_path):
            print("Constructing Graph ...")

            G, query_embeddings, passage_embeddings = construct_graph(query_embeddings, passage_embeddings, k_neighbours, graph_path, eigenvectors_path, distance, n_components, use_spectral_decomposition, query_projection)
            if use_spectral_decomposition:
                if query_projection:
                    query_embeddings = self.out_of_sample_barycentric(original_query_embeddings, original_passage_embeddings, passage_embeddings)
                    return G, query_embeddings, passage_embeddings
                else:
                    return G, original_query_embeddings, original_passage_embeddings
            else:
                return G, original_query_embeddings, original_passage_embeddings
        else:
            G = read_graph(graph_path)
            if use_spectral_decomposition:
                if query_projection:
                    eigenvectors = pickle.load(open(eigenvectors_path, "br"))
                    spectral_passage_embeddings = eigenvectors[:, 1:n_components+1]
                    query_embeddings = self.out_of_sample_barycentric(original_query_embeddings, original_passage_embeddings, spectral_passage_embeddings)
                    return G, query_embeddings, spectral_passage_embeddings
                else:
                    return G, original_query_embeddings, original_passage_embeddings

            else:
                return G, original_query_embeddings, original_passage_embeddings
        
    def run_evaluation(self, experiment_type, k_list, evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G=None, k_neighbours=None, distance=None):
        if experiment_type == "baseline":
            retrieve_results = retrieve_k(query_embeddings, distance, passage_embeddings, max(k_list))
        
        elif experiment_type == "manifold":
            if self.mode == "connectivity":
                retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, max(k_list), False, distance)
            elif self.mode == "distance":
                retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, max(k_list), True, distance)

        retrieval_path = os.path.join(self.experiment_path, f"{experiment_type}_retrieval_results.pkl")
        with open(retrieval_path, "bw") as f:
            pickle.dump(retrieve_results, f)

        results = defaultdict(dict) 
        for evaluation_function in evaluation_functions:
            for k in k_list:
                results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, retrieve_results, relevance_map, evaluation_function, k)
        
        result_path = os.path.join(self.experiment_path, f"{experiment_type}_evaluation_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)


    def run_evaluation_with_cache(self, k_list, evaluation_functions, question_ids, passage_ids, relevance_map):
        try:
            result_path = os.path.join(self.experiment_path, "retrieval_results.pkl")
        except FileNotFoundError as e:
            print(f"Cache files not found: {e}")
            raise
        
        retrieve_results = pickle.load(open(result_path, "br"))

        results = defaultdict(dict) 
        for evaluation_function in evaluation_functions:
            for k in k_list:
                results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, retrieve_results, relevance_map, evaluation_function, k)
        
        result_path = os.path.join(self.experiment_path, "evaluation_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)

