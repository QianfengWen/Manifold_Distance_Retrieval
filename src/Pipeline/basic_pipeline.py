import numpy as np
from collections import defaultdict
import json, os, pickle
from src.Retrieval.l2_dist_retrieval import retrieve_k_l2
from src.Retrieval.manifold_dist_retrieval import retrieve_k_manifold_baseline, retrieve_k_manifold_baseline_reciprocal
from src.Evaluation.evaluation import evaluate
from src.Graph.graph import construct_graph, construct_graph_reciprocal, read_graph
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

            self.create_new_graph = kwargs["create_new_graph"]
            self.graph_path = kwargs["graph_path"]
            self.reciprocal = kwargs.get("reciprocal", False)
            self.k_neighbours = kwargs["k_neighbours"]
            self.weight = kwargs.get("weight", 1)

            self.evaluation_functions = kwargs["evaluation_functions"]
            self.k_list = kwargs["k_list"]
        
        except KeyError as e:
            print(f"Missing argument: {e}")
            raise
    
    def run_pipeline(self, with_cache=False):
        print(f"********************* Running Experiments: {self.experiment_path} *********************")

        print("********************* Loading Data *********************")
        question_ids, question_texts, passage_ids, passage_texts, relevance_map = self.load_data(self.dataloader)
        
        
        if not with_cache:        
            print("********************* Handling Embeddings *********************")
            query_embeddings, passage_embeddings = self.handle_embeddings(self.model_name, self.query_embeddings_path, self.passage_embeddings_path, question_texts, passage_texts)
            
            print("********************* Handling Graph *********************")
            G = self.handle_graph(self.create_new_graph, passage_embeddings, self.k_neighbours, self.graph_path, self.reciprocal)
            
            print("********************* Running Evaluation *********************")
            self.run_evaluation(self.k_list, self.evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G, self.k_neighbours, self.weight)
        
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
    

    def handle_graph(self, create_new, passage_embeddings, k_neighbours, graph_path, reciprocal):
        if create_new:
            if not reciprocal:
                G = construct_graph(passage_embeddings, k_neighbours, graph_path)
            else:
                G = construct_graph_reciprocal(passage_embeddings, k_neighbours, graph_path)
                
        else:
            G = read_graph(graph_path)
        
        return G
        

    def run_evaluation(self, k_list, evaluation_functions, question_ids, passage_ids, query_embeddings, passage_embeddings, relevance_map, G, k_neighbours, weight):
        baseline_retrieve_results = retrieve_k_l2(query_embeddings, passage_embeddings, max(k_list))
        manifold_retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, weight, max(k_list))

        baseline_path = os.path.join(self.experiment_path, "baseline_retrieval_results.pkl")
        with open(baseline_path, "bw") as f:
            pickle.dump(baseline_retrieve_results, f)

        manifold_path = os.path.join(self.experiment_path, "manifold_retrieval_results.pkl")
        with open(manifold_path, "bw") as f:
            pickle.dump(manifold_retrieve_results, f)
        

        baseline_results = defaultdict(dict) 
        manifold_results = defaultdict(dict)
        for evaluation_function in evaluation_functions:
            for k in k_list:
                baseline_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, baseline_retrieve_results, relevance_map, evaluation_function, k)

        for evaluation_function in evaluation_functions:
            for k in k_list:
                manifold_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, manifold_retrieve_results, relevance_map, evaluation_function, k)
        
        baseline_path = os.path.join(self.experiment_path, "baseline_evaluation_results.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=4)
        
        manifold_path = os.path.join(self.experiment_path, "manifold_evaluation_results.json")
        with open(manifold_path, "w") as f:
            json.dump(manifold_results, f, indent=4)


    def run_evaluation_with_cache(self, k_list, evaluation_functions, question_ids, passage_ids, relevance_map):
        try:
            baseline_path = os.path.join(self.experiment_path, "baseline_retrieval_results.pkl")
            manifold_path = os.path.join(self.experiment_path, "manifold_retrieval_results.pkl")
        except FileNotFoundError as e:
            print(f"Cache files not found: {e}")
            raise
        
        baseline_retrieve_results = pickle.load(open(baseline_path, "br"))
        manifold_retrieve_results = pickle.load(open(manifold_path, "br"))

        baseline_results = defaultdict(dict) 
        manifold_results = defaultdict(dict)
        for evaluation_function in evaluation_functions:
            for k in k_list:
                baseline_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, baseline_retrieve_results, relevance_map, evaluation_function, k)

        for evaluation_function in evaluation_functions:
            for k in k_list:
                manifold_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, manifold_retrieve_results, relevance_map, evaluation_function, k)
        
        baseline_path = os.path.join(self.experiment_path, "baseline_evaluation_results.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=4)
        
        manifold_path = os.path.join(self.experiment_path, "manifold_evaluation_results.json")
        with open(manifold_path, "w") as f:
            json.dump(manifold_results, f, indent=4)
