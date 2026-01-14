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
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32], help='Lista de batch sizes para executar sequencialmente (preferencialmente múltiplos de 8). Ex: 8 16 32 64 128')
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(1)
    
    cfg = Config()
    
    if args.batch_sizes is None:
        default_batch = cfg.data.batch_size
        logger.info(f"Flag --batch_sizes não informada. Usando valor padrão do Config: {default_batch}")
        args.batch_sizes = [default_batch]
        
    for b in args.batch_sizes:
        if b <= 0:
            logger.error(f"[ERRO DE CONFIGURAÇÃO] O batch size {b} inválido.")
            logger.error("Todos os batch sizes devem ser inteiros positivos.")
            sys.exit(1)
            
    logger.info(f"Pipeline iniciado para os seguintes batch sizes: {args.batch_sizes}")

    for current_batch_size in args.batch_sizes:
        
        cfg.data.batch_size = current_batch_size
        
        experiment_dir = cfg.get_checkpoint_dir()
        cfg.train.checkpoint_dir = experiment_dir

        logger.info(f"Diretório do Experimento: {experiment_dir}")
    
        cfg.data.batch_size = current_batch_size
        
        logger.info("=" * 60)
        logger.info(f">>> INICIANDO EXECUÇÃO PARA BATCH SIZE: {current_batch_size} <<<")
        logger.info("=" * 60)
        
        if args.train:
            if os.path.exists(experiment_dir):
                logger.error("=" * 60)
                logger.error(f"[ERRO] Modelo já existe!")
                logger.error(f"Pasta encontrada: {experiment_dir}")
                logger.error("Para retreinar, apague a pasta manualmente ou altere os parâmetros.")
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
                
        train_dir_path = os.path.join(cfg.data.data_path, cfg.data.dataset_name, 'treino')
        test_dir_path = os.path.join(cfg.data.data_path, cfg.data.dataset_name, 'teste')

        if not os.path.exists(train_dir_path):
            pass
        
        print(f"Inicializando DataManager e calculando estatísticas em: {train_dir_path}")
        dm = DataManager(cfg.data, train_dir=train_dir_path, train_config=cfg.train)

        if args.train:
            logger.info(">>> INICIANDO TREINAMENTO <<<")
            
            if not os.path.exists(train_dir_path):
                raise FileNotFoundError(f"Dataset de treino não encontrado em {train_dir_path}. Rode o split_dataset.py primeiro.")

            train_loader = dm.get_loader(train_dir_path, is_contrastive=True, mode='train')

            trainer = SupConTrainer(cfg, train_loader)
            trainer.run()

        if args.eval:
            logger.info(">>> INICIANDO AVALIAÇÃO  <<<")
            
            if not os.path.exists(train_dir_path) or not os.path.exists(test_dir_path):
                raise FileNotFoundError(f"Pastas de treino ou teste não encontradas.")

            logger.info("Carregando loader de Treino para extração de features...")
            eval_train_loader = dm.get_loader(train_dir_path, is_contrastive=False, mode='test')
            
            logger.info("Carregando loader de Teste...")
            eval_test_loader = dm.get_loader(test_dir_path, is_contrastive=False, mode='test')

            evaluator = LinearEvaluator(cfg, eval_train_loader, eval_test_loader)
            evaluator.run()

if __name__ == '__main__':
    main()