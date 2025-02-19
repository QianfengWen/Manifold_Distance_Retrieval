#!/bin/bash
EMBEDDING_MODEL_LIST="msmarco-distilbert-base-tas-b msmarco-distilbert-dot-v5"

# Run the Python script with the specified arguments
python -m src.Pipeline.run \
    --embedding_model_list $EMBEDDING_MODEL_LIST \
    --experiment_type "baseline"