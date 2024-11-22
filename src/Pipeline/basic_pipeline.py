import numpy as np
from collections import defaultdict
import json, os, pickle

from src.Retrieval.l2_dist_retrieval import retrieve_k_l2
from src.Retrieval.manifold_dist_retrieval import retrieve_k_manifold_baseline
from src.Evaluation.evaluation import evaluate, recall_k, precision_k, mean_average_precision_k
from src.Dataset.scidocs import Scidocs
from src.Graph.read_graph import read_graph
from src.Graph.construct_graph import construct_graph

experiment_name = "scidocs, k=10, weight=1"
os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

############ Load data ############
dataloader = Scidocs()
question_ids, _, passage_ids, _, relevance_map = dataloader.load_data()


############ Load embeddings ############
query_embeddings_path = "data/scidoc/tas-b-query_embeddings.pkl"
passage_embeddings_path = "data/scidoc/tas-b-doc_embeddings.pkl"


with open(query_embeddings_path, "rb") as f:
    query_embeddings = pickle.load(f)
    assert isinstance(query_embeddings, np.ndarray), "query_embeddings should be a numpy array"

with open(passage_embeddings_path, "rb") as f:
    passage_embeddings = pickle.load(f)
    assert isinstance(passage_embeddings, np.ndarray), "passage_embeddings should be a numpy array"

############ Manifold Setup ############
k_neighbours = 10
weight = 1

graph_path = "data/scidoc/graph_root_10.json"
G = read_graph(graph_path)
# G = construct_graph(passage_embeddings, k_neighbours, graph_path)


############ Evaluation Setup ############
k_list = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
evaluation_functions = [recall_k, precision_k, mean_average_precision_k]


############ Begin pipeline ############

baseline_retrieve_results = retrieve_k_l2(query_embeddings, passage_embeddings, max(k_list))
manifold_retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, weight, max(k_list))

# save the retrieve results
baseline_path = os.path.join("results", experiment_name, "baseline_retrieval_results.pkl")
with open(baseline_path, "bw") as f:
    pickle.dump(baseline_retrieve_results, f)

manifold_path = os.path.join("results", experiment_name, "manifold_retrieval_results.pkl")
with open(manifold_path, "bw") as f:
    pickle.dump(manifold_retrieve_results, f)

print(manifold_retrieve_results.shape)


baseline_results = defaultdict(dict)
manifold_results = defaultdict(dict)

for evaluation_function in evaluation_functions:
    for k in k_list:
        baseline_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, baseline_retrieve_results, relevance_map, evaluation_function, k)

for evaluation_function in evaluation_functions:
    for k in k_list:
        manifold_results[evaluation_function.__name__][k] = evaluate(question_ids, passage_ids, manifold_retrieve_results, relevance_map, evaluation_function, k)

# Save results
baseline_path = os.path.join("results", experiment_name, "baseline_evaluation_results.json")
with open(baseline_path, "w") as f:
    json.dump(baseline_results, f, indent=4)

manifold_path = os.path.join("results", experiment_name, "manifold_evaluation_results.json")
with open(manifold_path, "w") as f:
    json.dump(manifold_results, f, indent=4)

