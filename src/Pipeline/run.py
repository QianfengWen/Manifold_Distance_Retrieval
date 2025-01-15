from src.Dataset.scidocs import Scidocs
from src.Dataset.fiqa import Fiqa
from src.Dataset.trec_covid import TrecCovid
from src.Dataset.msmarco import Msmarco
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.antique import Antique
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline

available_datasets = {"scidocs": Scidocs, "msmarco": Msmarco, "antique": Antique, "nfcorpus": NFCorpus}
dataset_name = "scidocs"

if __name__ == "__main__":
    for k in range(10, 11):
        print(f"Running experiment for k={k}")
        pipeline_kwargs = {
            "dataloader": available_datasets[dataset_name](),
            # "model_name": "sentence-transformers/msmarco-distilbert-base-tas-b",
            "query_embeddings_path": f"data/{dataset_name}/tas-b-query_embeddings.pkl",
            "passage_embeddings_path": f"data/{dataset_name}/tas-b-doc_embeddings.pkl",

            "experiment_type": "manifold",
            "create_new_graph": False,
            "k_neighbours": k,
            "graph_type": "knn",
            "distance": "l2",
            "mode": "connectivity",
            
            "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
            "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        if pipeline_kwargs["distance"] == "spectral":
            pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['graph_type']}_{pipeline_kwargs['distance']}_n_components={pipeline_kwargs['n_components']}.pkl"
            experiment_name = f"{dataset_name}/k={pipeline_kwargs['k_neighbours']}___graph_type={pipeline_kwargs['graph_type']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}__n_components={pipeline_kwargs['n_components']}"
        else:    
            pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['graph_type']}_{pipeline_kwargs['distance']}.pkl"
            experiment_name = f"{dataset_name}/k={pipeline_kwargs['k_neighbours']}___graph_type={pipeline_kwargs['graph_type']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}"

        pipeline = Pipeline(experiment_name, **pipeline_kwargs)
        pipeline.run_pipeline()
    # pipeline.run_evaluation_with_cache()

