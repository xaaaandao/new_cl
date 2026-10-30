#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8
do
    echo "=========================================="
    echo "Treinando com batch_size = $batch..."
    echo "=========================================="
    python main.py --train --batch_sizes $batch --use_pretrained --dataset_name "herbarium2019+min=50" --epochs 4000
done
