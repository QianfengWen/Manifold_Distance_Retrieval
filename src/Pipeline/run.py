from src.Dataset.scidocs import Scidocs
from src.Dataset.msmarco import MSMARCO
from src.Dataset.nfcorpus import NFCorpus
from src.Dataset.antique import Antique
from src.Evaluation.evaluation import recall_k, precision_k, mean_average_precision_k
from src.Pipeline.basic_pipeline import Pipeline
import time
import argparse

available_datasets = {
    "scidocs": Scidocs, 
    "msmarco": MSMARCO, 
    "antique": Antique, 
    "NFCorpus": NFCorpus
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")

    # Define arguments
    parser.add_argument("--k_list", type=int, nargs="+", default=[1], help="List of k values for KNN.")
    parser.add_argument("--embedding_model_list", type=str, nargs="+", 
                        default=["msmarco-distilbert-base-tas-b"], help="List of embedding models to use.")
    parser.add_argument("--use_spectral_distance", type=int, default=0)
    parser.add_argument("--n_components_list", type=int, nargs="+", 
                        default=[700], help="List of components for dimensionality reduction.")
    parser.add_argument("--mode_list", type=str, nargs="+", 
                        default=["connectivity"], help="List of modes (e.g., 'connectivity', 'distance').")
    parser.add_argument("--experiment_type", type=str, default="manifold", help="Experiment type (e.g., 'manifold', 'baseline').")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for embedding_model in args.embedding_model_list:
        for dataset_name in available_datasets.keys():
            for k in args.k_list:
                eigenvectors_path = f"data/{dataset_name}/eigenvectors_k={k}_euclidean.pkl"
                for mode in args.mode_list:  
                    if args.use_spectral_distance:
                        for n_components in args.n_components_list:
                            print(f"\n\nRunning manifold experiment for for {dataset_name} and {embedding_model} with k = {k}, distance_type = euclidean, distance mode = {mode}, n_components = {n_components}")
                            pipeline_kwargs = {
                                "dataloader": available_datasets[dataset_name](),
                                "query_embeddings_path": f"data/{dataset_name}/{embedding_model}-query_embeddings.pkl",
                                "passage_embeddings_path": f"data/{dataset_name}/{embedding_model}-doc_embeddings.pkl",

                                "experiment_type": args.experiment_type,
                                "create_new_graph": True,
                                "use_spectral_distance": True,
                                "query_projection": False,
                                "k_neighbours": k,
                                "distance": "euclidean",
                                "mode": mode,
                                "n_components": n_components,
                                "eigenvectors_path": eigenvectors_path,

                                "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
                                "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            }
                            pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['distance']}_n_components={pipeline_kwargs['n_components']}_{embedding_model}.pkl"
                            experiment_name = f"{dataset_name}/{embedding_model}/k={pipeline_kwargs['k_neighbours']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}__n_components={pipeline_kwargs['n_components']}"
                            pipeline = Pipeline(experiment_name, **pipeline_kwargs)
                            start = time.time()
                            pipeline.run_pipeline()
                            end = time.time()
                            print("Finished running the experiment, it takes", end-start, "seconds")
                    else:
                        if args.experiment_type == "baseline":
                            print(f"\n\nRunning baseline experiment for {dataset_name} and {embedding_model} with distance type = euclidean")
                        elif args.experiment_type == "manifold":
                            print(f"\n\nRunning manifold experiment for {dataset_name} and {embedding_model} with k = {k}, distance_type = euclidean, distance mode = {mode}")
                        pipeline_kwargs = {
                            "dataloader": available_datasets[dataset_name](),
                            "query_embeddings_path": f"data/{dataset_name}/{embedding_model}-query_embeddings.pkl",
                            "passage_embeddings_path": f"data/{dataset_name}/{embedding_model}-doc_embeddings.pkl",
                            "experiment_type": args.experiment_type,
                            "create_new_graph": True,
                            "use_spectral_distance": False,
                            "query_projection": False,
                            "k_neighbours": k,
                            "distance": "euclidean",
                            "mode": mode,
                            "eigenvectors_path": None,

                            "evaluation_functions": [recall_k, precision_k, mean_average_precision_k],
                            "k_list": [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        }  
                        pipeline_kwargs["graph_path"] = f"data/{dataset_name}/graph_k={pipeline_kwargs['k_neighbours']}_{pipeline_kwargs['distance']}_{embedding_model}.pkl"
                        if args.experiment_type == "manifold":
                            experiment_name = f"{dataset_name}/{embedding_model}/k={pipeline_kwargs['k_neighbours']}___mode={pipeline_kwargs['mode']}___distance_type={pipeline_kwargs['distance']}"
                        elif args.experiment_type == "baseline":
                            experiment_name = f"{dataset_name}/{embedding_model}/baseline___distance_type={pipeline_kwargs['distance']}"
                        pipeline = Pipeline(experiment_name, **pipeline_kwargs)
                        start = time.time()
                        pipeline.run_pipeline()
                        end = time.time()
                        print("Finished running the experiment, it takes", end-start, "seconds")

