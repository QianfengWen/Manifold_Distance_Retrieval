import numpy as np
from collections import defaultdict
import networkx as nx
import json, os

from src.Retrieval.l2_dist_retrieval import retrieve_k_l2
from src.Retrieval.manifold_dist_retrieval import retrieve_k_manifold_baseline
from src.Evaluation.evaluation import evaluate, recall_k, precision_k, mean_average_precision_k

experiment_name = ""

############ Load data ############
question_ids = []
passage_ids = []
relevance_map = {}


############ Load embeddings ############
query_embeddings = np.zeros((len(question_ids), 768))
passage_embeddings = np.zeros((len(passage_ids), 768))


############ Manifold Setup ############
G = nx.Graph()
k_neighbours = 10
weight = 1

############ Evaluation Setup ############
k_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
evaluation_functions = [recall_k, precision_k, mean_average_precision_k]


############ Begin pipeline ############
# baseline
baseline_results = defaultdict(dict)
baseline_retrieve_results = retrieve_k_l2(query_embeddings, passage_embeddings, max(k_list))

for evaluation_function in evaluation_functions:
    for k in k_list:
        baseline_results[evaluation_function][k] = evaluate(question_ids, passage_ids, baseline_retrieve_results, relevance_map, evaluation_function, k)

# manifold
manifold_results = defaultdict(dict)
manifold_retrieve_results = retrieve_k_manifold_baseline(G, query_embeddings, passage_embeddings, k_neighbours, weight, max(k_list))

for evaluation_function in evaluation_functions:
    for k in k_list:
        manifold_results[evaluation_function][k] = evaluate(question_ids, passage_ids, manifold_retrieve_results, relevance_map, evaluation_function, k)

# Save results
baseline_path = os.path.join("results", experiment_name, "baseline_results.json")
with open(baseline_path, "w") as f:
    json.dump(baseline_results, f, indent=4)

manifold_path = os.path.join("results", experiment_name, "manifold_results.json")
with open(manifold_path, "w") as f:
    json.dump(manifold_results, f, indent=4)

