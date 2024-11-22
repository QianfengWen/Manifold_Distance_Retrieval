from src.Dataset.scidocs import Scidocs
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline

experiment_name = "scidocs, k=10, weight=1"

dataloader = Scidocs()

query_embeddings_path = "data/scidoc/tas-b-query_embeddings.pkl"
passage_embeddings_path = "data/scidoc/tas-b-doc_embeddings.pkl"


create_new_graph = False
graph_path = "data/scidoc/graph_root_10.json"
weight = 1
k_neighbours = 10


evaluation_functions = [recall_k, precision_k, mean_average_precision_k]
k_list = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

if __name__ == "__main__":
    pipeline_kwargs = {
        "dataloader": Scidocs(),
        "query_embeddings_path": "data/scidoc/tas-b-query_embeddings.pkl",
        "passage_embeddings_path": "data/scidoc/tas-b-doc_embeddings.pkl",
        "create_new_graph": False,
        "graph_path": "data/scidoc/graph_root_10.json",
        "weight": 1,
        "k_neighbours": 10,
        "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
        "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }
    pipeline = Pipeline(experiment_name, **pipeline_kwargs)
    pipeline.run_pipeline()


