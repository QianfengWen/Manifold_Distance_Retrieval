from src.Dataset.scidocs import Scidocs
from src.Dataset.fiqa import Fiqa
from src.Dataset.trec_covid import TrecCovid
from src.Dataset.msmarco import Msmarco
from src.Dataset.nfcorpus import NFCorpus
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline

if __name__ == "__main__":
    experiment_name = "Scidocs, manifold, graph_type=knn, distance=l2, mode=connectivity, n_components=500"
    pipeline_kwargs = {
        "dataloader": Scidocs(),
        # "model_name": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "query_embeddings_path": "data/scidocs/tas-b-query_embeddings.pkl",
        "passage_embeddings_path": "data/scidocs/tas-b-doc_embeddings.pkl",
        "experiment_type": "manifold",
        
        "create_new_graph": True,
        "graph_path": "data/scidocs/graph_root_6.json",
        "graph_type": "knn",
        "distance": "l2",
        "mode": "connectivity",
        "k_neighbours": 6,
        
        "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
        "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    
    # pipeline_kwargs = {
    #     "dataloader": Fiqa()    ,
    #     "query_embeddings_path": "data/fiqa/tas-b-query_embeddings.pkl",
    #     "passage_embeddings_path": "data/fiqa/tas-b-doc_embeddings.pkl",
    #     "create_new_graph": False,
    #     "graph_path": "data/fiqa/graph_root_10.json",
    #     "weight": 1,
    #     "k_neighbours": 10,
    #     "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
    #     "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     # 'model_name': 'sentence-transformers/msmarco-distilbert-base-tas-b'
    # }
    # experiment_name = "trec_covid, k=30, weight=1"
    # pipeline_kwargs = {
    #     "dataloader": TrecCovid(),
    #     "query_embeddings_path": "data/trec-covid/tas-b-query_embeddings.pkl",
    #     "passage_embeddings_path": "data/trec-covid/tas-b-doc_embeddings.pkl",
    #     "create_new_graph": False,
    #     "graph_path": "data/trec-covid/graph_root_30.json",
    #     "weight": 1,
    #     "k_neighbours": 30,
    #     "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
    #     "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000],
    # }
    # experiment_name = " msmarcro, k=10, weight=1,reciprocal"
    # pipeline_kwargs = {
    #     "dataloader": Msmarco(),
    #     "query_embeddings_path": "data/msmarco/tas-b-query_embeddings.pkl",
    #     "passage_embeddings_path": "data/msmarco/tas-b-doc_embeddings.pkl",
    #     "create_new_graph": True,
    #     "graph_path": "data/msmarco/graph_root_10.json",
    #     "weight": 1,
    #     "k_neighbours": 10,
    #     "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
    #     "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     "model_name": "sentence-transformers/msmarco-distilbert-base-tas-b"
    # }
    pipeline = Pipeline(experiment_name, **pipeline_kwargs)
    pipeline.run_pipeline()
    # pipeline.run_evaluation_with_cache()

