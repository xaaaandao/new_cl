#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8 16 32 64 128 256 512
do
    echo "=========================================="
    echo "Avaliando com batch_size = $batch..."
    echo "=========================================="
    python main.py --eval --batch_sizes $batch --f1 weighted --use_pretrained
done