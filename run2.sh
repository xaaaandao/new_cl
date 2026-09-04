#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8 16 32 64 128 256 512
do
    for loss in 0.2ace
    do
        echo "=========================================="
        echo "Avaliando com batch_size = $batch..."
        echo "=========================================="
        python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 weighted --use_pretrained
	    
    done
done