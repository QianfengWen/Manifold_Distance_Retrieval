# Manifold-aware Distance Metrics for Dense Passage Retrieval

This repository provides the code and data accompanying our paper titled **"Manifold-aware Distance Metrics for Dense Passage Retrieval"**.

## Setup
To get started, install the required Python packages:
```bash
pip install -r requirements.txt
```

## Configuration Options
You can customize the evaluation using the following flags:
- **`k_list`**: List of k values for KNN. We use a list of k values from 1 to 15 for main evaluation.
- **`embedding_model_list`**: List of embedding models to use. We use msmarco-distilbert-base-tas-b and msmarco-distilbert-dot-v5 for main evaluation.
- **`use_spectral_distance`**: Use spectral distance for graph construction. Default is 0 (False).
- **`n_components_list`**: List of components for dimensionality reduction of spectral decomposition. We use a list of dim = [700, 500, 300, 100].
- **`mode_list`**: List of distance cost to use. We use connectivity (uniform cost) and distance (distance cost) for main evaluation.
- **`experiment_type`**: What type of retrieval metric to use. Either "baseline" (Euclidean) or "manifold".

## Running the Project
1️⃣ Running the baseline experiments using Euclidean Distance as distance metric
```bash
bash run_baseline.sh
```

2️⃣ Running the knn graph experiments with Euclidean Distance as distance function
```bash
bash run_knn.sh
```

3️⃣ Running the knn graph experiments with Spectral Distance as distance function
```bash
bash run_knn_spectral.sh
```
