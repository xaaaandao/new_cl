import os
import argparse
import logging
import sys
from config import Config
from data import DataManager
from trainer import SupConTrainer
from evaluator import LinearEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento SupCon e Avaliação Linear")
    parser.add_argument('--train', action='store_true', help='Executa o treinamento')
    parser.add_argument('--eval', action='store_true', help='Executa a avaliação')
    args = parser.parse_args()

    # Se nenhum argumento for passado, mostra o help
    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(1)

    # 1. Carregar Configurações
    cfg = Config()
    experiment_dir = cfg.get_checkpoint_dir()
    cfg.train.checkpoint_dir = experiment_dir

    logger.info(f"Diretório do Experimento: {experiment_dir}")

    if args.train:
        if os.path.exists(experiment_dir):
            logger.error("=" * 60)
            logger.error(f"[ERRO CRÍTICO] Modelo já existe!")
            logger.error(f"Pasta encontrada: {experiment_dir}")
            logger.error("Para retreinar, apague a pasta manualmente ou altere os parâmetros (ex: epochs).")
            logger.error("=" * 60)
            sys.exit(1)
        else:
            os.makedirs(experiment_dir)
            os.makedirs(os.path.join(experiment_dir, "checkpoints"))
            os.makedirs(os.path.join(experiment_dir, "logs"))
            os.makedirs(os.path.join(experiment_dir, "results"))
            logger.info("Diretorio criado com sucesso.")
    
    if args.eval:
        if not os.path.exists(experiment_dir):
            logger.error(f"[ERRO] Modelo não encontrado para avaliação: {experiment_dir}")
            sys.exit(1)
            
    # Caminhos baseados na estrutura de pastas definida anteriormente
    train_dir_path = os.path.join(cfg.data.data_path, cfg.data.dataset_name, 'treino')
    test_dir_path = os.path.join(cfg.data.data_path, cfg.data.dataset_name, 'teste')

    if not os.path.exists(train_dir_path):
        pass
    
    print(f"Inicializando DataManager e calculando estatísticas em: {train_dir_path}")
    dm = DataManager(cfg.data, train_dir=train_dir_path)

    if args.train:
        logger.info(">>> INICIANDO MODO DE TREINAMENTO (SUPCON) <<<")
        
        if not os.path.exists(train_dir_path):
            raise FileNotFoundError(f"Dataset de treino não encontrado em {train_dir_path}. Rode o split_dataset.py primeiro.")

        # Carrega com TwoCropTransform e Augmentations
        train_loader = dm.get_loader(train_dir_path, is_contrastive=True, mode='train')

        trainer = SupConTrainer(cfg, train_loader)
        trainer.run()

    # --- FLUXO DE AVALIAÇÃO ---
    if args.eval:
        logger.info(">>> INICIANDO MODO DE AVALIAÇÃO  <<<")
        
        if not os.path.exists(train_dir_path) or not os.path.exists(test_dir_path):
            raise FileNotFoundError(f"Pastas de treino ou teste não encontradas.")

        # Carrega dados "crus" (sem crop duplo, apenas resize + normalize)
        # Precisamos das features de treino para treinar o SVM
        logger.info("Carregando loader de Treino para extração de features...")
        eval_train_loader = dm.get_loader(train_dir_path, is_contrastive=False, mode='test')
        
        # Precisamos das features de teste para validar
        logger.info("Carregando loader de Teste...")
        eval_test_loader = dm.get_loader(test_dir_path, is_contrastive=False, mode='test')

        evaluator = LinearEvaluator(cfg, eval_train_loader, eval_test_loader)
        evaluator.run()

if __name__ == '__main__':
    main()