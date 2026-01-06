import os
import shutil
import pandas as pd
import logging
import argparse
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def sanitize_folder_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace(" ", "_")
    name = re.sub(r'[^\w\-]', '', name)
    return name.lower()

def count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS)

def standardize_structure(base_path: str, info_csv_path: str, execute: bool = False):
    root_dir = Path(base_path)
    csv_path = Path(info_csv_path)

    if not root_dir.exists():
        logger.error(f"Diretório do dataset não encontrado: {root_dir}")
        return

    if not csv_path.exists():
        logger.error(f"Arquivo CSV não encontrado: {csv_path}")
        return

    logger.info("Carregando arquivo de mapeamento...")
    try:
        df = pd.read_csv(csv_path, sep=';')
        if 'levels' not in df.columns or 'f' not in df.columns:
            logger.error(f"Colunas esperadas ('levels', 'f') não encontradas. Colunas atuais: {df.columns}")
            return
    except Exception as e:
        logger.error(f"Erro ao ler CSV: {e}")
        return

    class_stats = []
    
    logger.info(f"Iniciando processamento de {len(df)} classes...")
    if not execute:
        logger.warning(">>> Contagens baseadas nas pastas atuais. <<<")

    for index, row in df.iterrows():
        try:
            folder_id = str(row['f']).strip()
            original_folder_name = f"f{folder_id}"
            
            raw_class_name = row['levels']
            new_class_name = sanitize_folder_name(raw_class_name)

            src_path = root_dir / original_folder_name
            dst_path = root_dir / new_class_name

            path_to_count = src_path
            
            renamed = False
            if execute:
                if src_path.exists() and not dst_path.exists():
                    src_path.rename(dst_path)
                    logger.info(f"[Renomeado]: {original_folder_name} -> {new_class_name}")
                    path_to_count = dst_path
                    renamed = True
                elif dst_path.exists():
                    logger.warning(f"[Pulo]: Destino já existe: {dst_path}")
                    path_to_count = dst_path
                else:
                    logger.warning(f"[Erro]: Origem não encontrada: {src_path}")
                    path_to_count = None
            else:
                if src_path.exists():
                    logger.info(f"[Simulação]: {original_folder_name} -> {new_class_name}")
                else:
                    logger.warning(f"[Simulação - Falha]: Pasta original {src_path} não existe.")
                    path_to_count = None

            count = 0
            if path_to_count and path_to_count.exists():
                count = count_images_in_dir(path_to_count)
            
            class_stats.append({
                'ID Original': original_folder_name,
                'Classe': new_class_name,
                'Imagens': count
            })

        except Exception as e:
            logger.error(f"Erro na linha {index}: {e}")

    print("\n" + "="*50)
    print(f"{'RELATÓRIO DO DATASET':^50}")
    print("="*50)
    
    stats_df = pd.DataFrame(class_stats)
    stats_df = stats_df.sort_values(by='Imagens', ascending=False)
    
    pd.set_option('display.max_rows', None) 
    print(stats_df.to_string(index=False))
    
    total_images = stats_df['Imagens'].sum()
    total_classes = len(stats_df)
    
    print("-" * 50)
    print(f"TOTAL DE CLASSES: {total_classes}")
    print(f"TOTAL DE IMAGENS: {total_images}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Padroniza dataset e gera estatísticas.")
    
    parser.add_argument('--dataset_dir', type=str, default='pr_dataset', help='Pasta contendo as classes...')
    parser.add_argument('--csv_path', type=str, default='pr_dataset/info_levels.csv', help='Caminho do CSV')
    parser.add_argument('--execute', action='store_true', help='Aplica as mudanças.')

    args = parser.parse_args()

    standardize_structure(args.dataset_dir, args.csv_path, args.execute)