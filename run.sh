#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 256
do
    for loss in 0.3
    do
        echo "=========================================="
        echo "Treinando com batch_size = $batch..."
        echo "=========================================="
        # python main.py --train --batch_sizes $batch --loss_weight $loss
        python main.py --train --batch_sizes $batch --loss_weight $loss --use_pretrained

        echo "=========================================="
        echo "Avaliando com batch_size = $batch..."
        echo "=========================================="
        python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 weighted --use_pretrained
        # python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 weighted
        python main.py --eval --batch_sizes $batch --loss_weight $loss --f1 macro --use_pretrained
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
