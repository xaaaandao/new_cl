#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8
do
    for loss in 0.05 0.6 0.15 0.45 0.3
    do
for batch in 8
        echo "=========================================="
        echo "Treinando com batch_size = $batch..."
        echo "=========================================="
        python main.py --train --batch_sizes $batch --loss_weight $loss --use_pretrained --dataset_name "102flowers" --epochs 4000
        python main.py --train --batch_sizes $batch --loss_weight $loss --use_pretrained --dataset_name "102segmentations" --epochs 4000
    done
done