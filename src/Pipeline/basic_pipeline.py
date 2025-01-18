import numpy as np
from collections import defaultdict
import json, os, pickle
from src.Retrieval.l2_dist_retrieval import retrieve_k
from src.Retrieval.manifold_dist_retrieval import retrieve_k_manifold_baseline, retrieve_k_manifold_baseline_reciprocal
from src.Evaluation.evaluation import evaluate
from src.Graph.graph import construct_graph, read_graph
from src.Embedding.embedding import create_embeddings, load_embeddings

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
            self.graph_type = kwargs["graph_type"] # connected vs. knn
            self.distance = kwargs["distance"] # l2, cos distance or spectral
            self.mode = kwargs["mode"] # connectivity (hop) vs. distance
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
        # print(f"********************* Running Experiments: {self.experiment_path} *********************")

        # print("********************* Loading Data *********************")
        question_ids, question_texts, passage_ids, passage_texts, relevance_map = self.load_data(self.dataloader)
        
        
        if not with_cache:        
            # print("********************* Handling Embeddings *********************")
            query_embeddings, passage_embeddings = self.handle_embeddings(self.model_name, self.query_embeddings_path, self.passage_embeddings_path, question_texts, passage_texts)
            
            print("********************* Handling Graph *********************")
            G = self.handle_graph(passage_embeddings, self.k_neighbours, self.graph_path, self.graph_type, self.distance, self.n_components, self.max_edges, self.max_percentage)
            
            print("********************* Running Evaluation *********************")
            self.run_evaluation(self.experiment_type, self.k_list, self.evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G, self.k_neighbours, self.distance)
        
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
    

    def handle_graph(self, passage_embeddings, k_neighbours, graph_path, graph_type, distance, n_components, max_edges, max_percentage):
        if self.create_new_graph and not os.path.exists(graph_path):
            print("Constructing Graph ...")
            G = construct_graph(passage_embeddings, k_neighbours, graph_path, graph_type, distance, n_components, max_edges, max_percentage)

        else:
            G = read_graph(graph_path)
        
        return G
        

    def run_evaluation(self, experiment_type, k_list, evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G, k_neighbours, distance):
        if experiment_type == "baseline":
            retrieve_results = retrieve_k(query_embeddings, distance, passage_embeddings, max(k_list))
        
        elif experiment_type == "manifold":
            if self.mode == "connectivity":
                retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, max(k_list), False)
            elif self.mode == "distance":
                retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, max(k_list), True)

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

