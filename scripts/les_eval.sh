#!/bin/bash

# Define os valores de batch que você quer testar
for batch in 8
do
    echo "=========================================="
    echo "Avaliando com batch_size = $batch..."
    echo "=========================================="
    python main.py --eval --batch_sizes $batch --f1 weighted --use_pretrained --max_iter 10000 --dataset_name "1-piece2" --eval_epochs 8
    python main.py --eval --batch_sizes $batch --f1 weighted --use_pretrained --max_iter 10000 --dataset_name "1-piece2" --eval_epochs 8 --feature_type projection
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
