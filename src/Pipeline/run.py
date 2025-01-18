from src.Dataset.scidocs import Scidocs
from src.Dataset.fiqa import Fiqa
from src.Dataset.trec_covid import TrecCovid
from src.Dataset.msmarco import MSMARCO
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.antique import Antique
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline
import time

available_datasets = {"scidocs": Scidocs, "msmarco": MSMARCO, "antique": Antique, "nfcorpus": NFCorpus}
dataset_name = "nfcorpus"

if __name__ == "__main__":
    # design choices
    k_list = [6]
    # graph_type_list = ["knn", "connected"]
    graph_type_list = ["knn"]
    distance_type_list = ["spectral"]
    n_components_list = [100, 300, 500, 700]
    mode_list = ["connectivity", "distance"]
    for k in k_list:
        for graph_type in graph_type_list:
            for distance_type in distance_type_list:
                for mode in mode_list:  
                    if distance_type == "spectral":
                        for n_components in n_components_list:
                            print(f"\n\nRunning experiment for k = {k}, graph_type = {graph_type}, distance_type = {distance_type}, mode = {mode}, n_components = {n_components}")
                            pipeline_kwargs = {
                                "dataloader": available_datasets[dataset_name](),
                                # "model_name": "sentence-transformers/msmarco-distilbert-base-tas-b",
                                "query_embeddings_path": f"data/{dataset_name}/tas-b-query_embeddings.pkl",
                                "passage_embeddings_path": f"data/{dataset_name}/tas-b-doc_embeddings.pkl",

                                "experiment_type": "manifold",
                                "create_new_graph": True,
                                "k_neighbours": k,
                                "graph_type": graph_type,
                                "distance": distance_type,
                                "mode": mode,
                                "n_components": n_components,

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
                            start = time.time()
                            pipeline.run_pipeline()
                            end = time.time()
                            print("Finished running the experiment, it takes", end-start, "seconds")
                    else:
                        print(f"Running experiment for k = {k}, graph_type = {graph_type}, distance_type = {distance_type}, mode = {mode}")
                        pipeline_kwargs = {
                            "dataloader": available_datasets[dataset_name](),
                            # "model_name": "sentence-transformers/msmarco-distilbert-base-tas-b",
                            "query_embeddings_path": f"data/{dataset_name}/tas-b-query_embeddings.pkl",
                            "passage_embeddings_path": f"data/{dataset_name}/tas-b-doc_embeddings.pkl",

                            "experiment_type": "manifold",
                            "create_new_graph": True,
                            "k_neighbours": k,
                            "graph_type": graph_type,
                            "distance": distance_type,
                            "mode": mode,

                            "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
                            "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        }  
                        pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['graph_type']}_{pipeline_kwargs['distance']}.pkl"
                        experiment_name = f"{dataset_name}/k={pipeline_kwargs['k_neighbours']}___graph_type={pipeline_kwargs['graph_type']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}"

                        pipeline = Pipeline(experiment_name, **pipeline_kwargs)
                        start = time.time()
                        pipeline.run_pipeline()
                        end = time.time()
                        print("Finished running the experiment, it takes", end-start, "seconds")

