from src.Dataset.scidocs import Scidocs
from src.Dataset.fiqa import Fiqa
from src.Dataset.trec_covid import TrecCovid
from src.Dataset.msmarco import MSMARCO
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.antique import Antique
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline
import time

available_datasets = {"scidocs": Scidocs, "msmarco": MSMARCO, "antique": Antique, "NFCorpus": NFCorpus}
dataset_name = "NFCorpus"

if __name__ == "__main__":
    # design choices
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    embedding_model_list = ["tas-b", 
                            # "all-MiniLM-L6-v2"
                            ]
    distance_type_list = ["euclidean", "cosine"]
    use_spectral_decomposition = True
    query_projection = False
    n_components_list = [700, 500, 300, 100]
    mode_list = ["connectivity", "distance"]
    for k in k_list:
        for distance_type in distance_type_list:
            for mode in mode_list:  
                if use_spectral_decomposition:
                    for n_components in n_components_list:
                        print(f"\n\nRunning experiment for k = {k}, distance_type = {distance_type}, distance mode = {mode}, n_components = {n_components}")
                        pipeline_kwargs = {
                            "dataloader": available_datasets[dataset_name](),
                            # "model_name": embedding_model_list[0],
                            "query_embeddings_path": f"data/{dataset_name}/{embedding_model_list[0]}-query_embeddings.pkl",
                            "passage_embeddings_path": f"data/{dataset_name}/{embedding_model_list[0]}-doc_embeddings.pkl",

                            "experiment_type": "manifold",
                            # "experiment_type": "baseline",
                            "create_new_graph": True,
                            "use_spectral_decomposition": True,
                            "query_projection": query_projection,
                            "k_neighbours": k,
                            "distance": distance_type,
                            "mode": mode,
                            "n_components": n_components,

                            "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
                            "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        }
                        pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['distance']}_n_components={pipeline_kwargs['n_components']}.pkl"
                        experiment_name = f"{dataset_name}/k={pipeline_kwargs['k_neighbours']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}__n_components={pipeline_kwargs['n_components']}"
                        # experiment_name = f"baseline/{dataset_name}_spectral_n_components={pipeline_kwargs['n_components']}"
                        pipeline = Pipeline(experiment_name, **pipeline_kwargs)
                        start = time.time()
                        pipeline.run_pipeline()
                        end = time.time()
                        print("Finished running the experiment, it takes", end-start, "seconds")
                else:
                    print(f"Running experiment for k = {k}, distance_type = {distance_type}, distance mode = {mode}")
                    pipeline_kwargs = {
                        "dataloader": available_datasets[dataset_name](),
                        "model_name": embedding_model_list[0],
                        "query_embeddings_path": f"data/{dataset_name}/{embedding_model_list[0]}-query_embeddings.pkl",
                        "passage_embeddings_path": f"data/{dataset_name}/{embedding_model_list[0]}-doc_embeddings.pkl",

                        # "experiment_type": "manifold",
                        "experiment_type": "baseline",
                        "create_new_graph": True,
                        "use_spectral_decomposition": False,
                        "query_projection": False,
                        "k_neighbours": k,
                        "distance": distance_type,
                        "mode": mode,

                        "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
                        "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    }  
                    pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['distance']}.pkl"
                    experiment_name = f"{dataset_name}/k={pipeline_kwargs['k_neighbours']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}"
                    # experiment_name = f"baseline/{dataset_name}"
                    pipeline = Pipeline(experiment_name, **pipeline_kwargs)
                    start = time.time()
                    pipeline.run_pipeline()
                    end = time.time()
                    print("Finished running the experiment, it takes", end-start, "seconds")

