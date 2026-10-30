#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8
do
    echo "=========================================="
    echo "Treinando com batch_size = $batch..."
    echo "=========================================="
    python main.py --train --batch_sizes $batch --use_pretrained --dataset_name "102flowers" --epochs 4000
    python main.py --train --batch_sizes $batch --use_pretrained --dataset_name "102segmentations" --epochs 4000
done
