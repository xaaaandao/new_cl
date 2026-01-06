import os
import shutil
import random
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def setup_directory_structure(base_path: Path) -> Path:
    original_path = base_path / 'original'
    train_path = base_path / 'treino'
    test_path = base_path / 'teste'

    if not original_path.exists():
        logger.info(f"Estruturando dados...")
        original_path.mkdir(parents=True, exist_ok=True)

        items = [x for x in base_path.iterdir() if x.is_dir()]
        
        moved_count = 0
        for item in items:
            if item.name in ['original', 'treino', 'teste', 'saved_models']:
                continue
            
            shutil.move(str(item), str(original_path / item.name))
            moved_count += 1
        
        logger.info(f"Estruturação concluída. {moved_count} pastas de classe movidas para {original_path}")
    else:
        logger.info("Dados já estruturados.")

    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    return original_path

def split_data(base_path: str, train_ratio: float = 0.8, seed: int = 42):
    root = Path(base_path)
    
    if not root.exists():
        logger.error(f"Diretório não encontrado: {root}")
        return

    original_dir = setup_directory_structure(root)
    train_dir = root / 'treino'
    test_dir = root / 'teste'

    random.seed(seed)

    classes = [d for d in original_dir.iterdir() if d.is_dir()]
    
    if not classes:
        logger.warning("Nenhuma classe encontrada dentro de '/original'. Verifique o diretório.")
        return

    logger.info(f"Iniciando split {train_ratio*100}% / {(1-train_ratio)*100}% para {len(classes)} classes...")

    total_train = 0
    total_test = 0

    for class_dir in classes:
        class_name = class_dir.name
        
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        dest_train = train_dir / class_name
        dest_test = test_dir / class_name
        
        dest_train.mkdir(parents=True, exist_ok=True)
        dest_test.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            if not (dest_train / img.name).exists():
                shutil.copy2(img, dest_train / img.name)
        
        for img in test_imgs:
            if not (dest_test / img.name).exists():
                shutil.copy2(img, dest_test / img.name)

        total_train += len(train_imgs)
        total_test += len(test_imgs)
        
        logger.info(f"Classe '{class_name}': {len(train_imgs)} treino, {len(test_imgs)} teste.")

    print("-" * 50)
    logger.info("Processo finalizado com sucesso.")
    logger.info(f"Total Imagens Treino: {total_train}")
    logger.info(f"Total Imagens Teste:  {total_test}")
    logger.info(f"Proporção Final: {total_train / (total_train+total_test):.2f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separa dataset em treino e teste mantendo backup original.")
    
    parser.add_argument('--path', type=str, default='pr_dataset', help='Caminho raiz do dataset')
    parser.add_argument('--ratio', type=float, default=0.8, help='Proporção para treino (0.0 a 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatória para reprodutibilidade')

    args = parser.parse_args()
    
    split_data(args.path, args.ratio, args.seed)