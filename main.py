import os
import argparse
from config import Config
from data import DataManager
from trainer import SupConTrainer

def main():
    cfg = Config()
    
    data_folder = os.path.join(cfg.data.data_path, cfg.data.dataset_name, 'treino')
    
    print(f"Carregando dados de: {data_folder}")
    dm = DataManager(cfg.data)
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Dataset não encontrado em {data_folder}.")

    train_loader = dm.get_loader(data_folder, is_contrastive=True)

    trainer = SupConTrainer(cfg, train_loader)
    trainer.run()

if __name__ == '__main__':
    main()