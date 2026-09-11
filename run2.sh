#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8 16 32 64
do
    for loss in 0.3
    do
        echo "=========================================="
        echo "Avaliando com batch_size = $batch..."
        echo "=========================================="
        python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 weighted --use_pretrained --dataset_name "herbarium2019+min=50"
        # python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 weighted
        python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 macro --use_pretrained --dataset_name "herbarium2019+min=50"
        # python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 macro
	    
    done
done

echo "===================================================="
echo "Todos os treinamentos e avaliações foram concluídas!"
echo "===================================================="

echo "===================================================="
echo "Criando cópia dos resultados!"
echo "===================================================="
python saved_models/copy.py

echo "===================================================="
echo "Comprimindo resultados!"
echo "===================================================="
cd saved_models
for d in */; do tar -czvf "${d%/}.tar.gz" "$d"; done
