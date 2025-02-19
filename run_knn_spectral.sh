#!/bin/bash
K_LIST="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
EMBEDDING_MODEL_LIST="msmarco-distilbert-base-tas-b msmarco-distilbert-dot-v5"
MODE_LIST="connectivity distance"
N_COMPONENTS_LIST="700 500 300 100"

python -m src.Pipeline.run \
    --k_list $K_LIST \
    --embedding_model_list $EMBEDDING_MODEL_LIST \
    --use_spectral_distance 1 \
    --n_components_list $N_COMPONENTS_LIST \
    --mode_list $MODE_LIST \
    --experiment_type "manifold"