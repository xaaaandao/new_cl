import os
import torch
import logging
import re
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path

from config import Config
from network import MultiHeadSupConResNet

logger = logging.getLogger(__name__)


class LinearEvaluator:
    def __init__(self, config: Config, train_loader, test_loader, average, max_iter, eval_epochs):
        self.cfg = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = self.cfg.train.device
        self.max_iter = max_iter
        self.average = average
        self.eval_epochs = eval_epochs

        self.results_path = os.path.join(self.cfg.train.checkpoint_dir, 'results')
        os.makedirs(self.results_path, exist_ok=True)
        self.csv_file = f'results+{average}.csv' if self.max_iter == -1 else f'results+{average}+maxiter={self.max_iter}.csv'
        self.csv_file = os.path.join(self.results_path, self.csv_file)

    def load_backbone(self, checkpoint_path: str, use_pretrained) -> MultiHeadSupConResNet:
        model = MultiHeadSupConResNet(
            name=self.cfg.model.name,
            feat_dim_genus=self.cfg.model.feat_dim_genus,
            feat_dim_species=self.cfg.model.feat_dim_species,
            use_pretrained=use_pretrained
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint['model']
        new_state_dict = {}

        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        return model

    def extract_features(self, model: MultiHeadSupConResNet, loader):
        """
        Extrai features do encoder compartilhado (antes das projection heads),
        junto com os rótulos de gênero e de espécie de cada amostra.
        """
        features_list = []
        genus_labels_list = []
        species_labels_list = []

        with torch.no_grad():
            for images, genus_labels, species_labels in tqdm(loader, desc="Extraindo Features"):

                if isinstance(images, list):
                    images = images[0]

                images = images.to(self.device)
                feats = model.encoder(images)

                features_list.append(feats.cpu().numpy())
                genus_labels_list.append(genus_labels.numpy())
                species_labels_list.append(species_labels.numpy())

        X = np.concatenate(features_list, axis=0)
        y_genus = np.concatenate(genus_labels_list, axis=0)
        y_species = np.concatenate(species_labels_list, axis=0)
        return X, y_genus, y_species

    def train_svm(self, X_train, y_train):
        logger.info("Iniciando GridSearch do SVM...")

        param_grid = {
            'svc__C': self.cfg.eval.svm_c,
            'svc__kernel': self.cfg.eval.svm_kernel
        }

        pipe = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42, verbose=True, max_iter=self.max_iter))

        clf = GridSearchCV(
            pipe,
            param_grid,
            cv=5,  # 3-Fold Cross Validation
            n_jobs=self.cfg.eval.n_jobs,
            scoring=f'f1_{self.average}',
            verbose=True
        )

        clf.fit(X_train, y_train)
        logger.info(f"Melhor estimador encontrado: {clf.best_params_}")
        return clf.best_estimator_, clf.best_estimator_.named_steps['svc'].n_iter_.tolist()

    def compute_metrics(self, clf, X_test, y_test, epoch_num, level: str):
        """Calcula F1, Top-3 e Top-5 para um dado nível ('genus' ou 'species')."""

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        f1 = f1_score(y_test, y_pred, average=self.average)

        n_classes = len(np.unique(y_test))
        acc3 = top_k_accuracy_score(y_test, y_prob, k=3) if n_classes > 3 else 1.0
        acc5 = top_k_accuracy_score(y_test, y_prob, k=5) if n_classes > 5 else 1.0

        return {
            'epoch': epoch_num,
            'level': level,
            'f1_score': f1,
            'top3_acc': acc3,
            'top5_acc': acc5
        }

    def save_results(self, metrics: dict):
        df_new = pd.DataFrame([metrics])

        if not os.path.exists(self.csv_file):
            df_new.to_csv(self.csv_file, index=False)
        else:
            df_new.to_csv(self.csv_file, mode='a', header=False, index=False)

        logger.info(f"Resultados salvos: {metrics}")

    def save_n_iters(self, epoch_num, level, n_iters):
        data = {
            "epoch": epoch_num,
            "level": level,
            "n_iters": n_iters
        }
        df = pd.DataFrame([data])

        eval_dir = os.path.join(self.cfg.train.checkpoint_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        csv_n_iters_file = os.path.join(eval_dir, f"n_iters.csv")
        if not os.path.exists(csv_n_iters_file):
            df.to_csv(csv_n_iters_file, index=False)
        else:
            df.to_csv(csv_n_iters_file, mode='a', header=False, index=False)


    def sort_csv_by_epoch(self):
        if not os.path.exists(self.csv_file):
            return

        try:
            logger.info("Reordenando arquivo final de resultados...")
            df = pd.read_csv(self.csv_file)

            def epoch_sorter(val):
                if str(val).isdigit():
                    return int(val)

                return 999999

            df['sort_key'] = df['epoch'].apply(epoch_sorter)
            df = df.sort_values('sort_key')
            df = df.drop(columns=['sort_key'])

            df.to_csv(self.csv_file, index=False)
            logger.info("Arquivo results.csv reordenado com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao reordenar CSV: {e}")
            
    def run(self, use_pretrained):
        base_dir = Path(self.cfg.train.checkpoint_dir).resolve()
        checkpoints_dir = base_dir / "checkpoints"

        logger.info(f"Buscando checkpoints em: {checkpoints_dir}")

        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"O diretorio: {checkpoints_dir} nao existe.")

        checkpoints = list(checkpoints_dir.rglob("*.pth"))
        checkpoints = [str(p) for p in checkpoints if p.is_file()]
        if self.eval_epochs > 0:
            checkpoints = self.filter_checkpoints(checkpoints)
        checkpoints = sorted(checkpoints)

        if not checkpoints:
            logger.warning(f"Nenhum checkpoint encontrado em {self.cfg.train.checkpoint_dir}")
            logger.info(f"Conteudo encontrado na pasta: {[f.name for f in checkpoints_dir.iterdir()]}")
            return

        logger.info(f"Encontrados {len(checkpoints)} checkpoints para avaliar.")

        for ckpt_path in checkpoints:
            try:
                match = re.search(r'epoch_(\d+)', ckpt_path)
                epoch_num = int(match.group(1)) if match else "last"

                logger.info(f"--- Avaliando Checkpoint: {os.path.basename(ckpt_path)} (Epoch {epoch_num}) ---")

                model = self.load_backbone(ckpt_path, use_pretrained)

                logger.info("Extraindo features de TREINO...")
                X_train, y_train_genus, y_train_species = self.extract_features(model, self.train_loader)

                logger.info("Extraindo features de TESTE...")
                X_test, y_test_genus, y_test_species = self.extract_features(model, self.test_loader)

                # Avalia o mesmo espaço de features (encoder compartilhado) tanto
                # para a tarefa de classificar GÊNERO quanto para classificar ESPÉCIE.
                for level, y_train, y_test in [
                    ('genus', y_train_genus, y_test_genus),
                    ('species', y_train_species, y_test_species),
                ]:
                    logger.info(f"--- Nível: {level} ---")
                    clf, n_iters = self.train_svm(X_train, y_train)
                    metrics = self.compute_metrics(clf, X_test, y_test, epoch_num, level)
                    self.save_results(metrics)
                    self.save_n_iters(epoch_num, level, n_iters)

            except Exception as e:
                logger.error(f"Falha ao avaliar checkpoint {ckpt_path}: {e}")
                continue

        self.sort_csv_by_epoch()

    def filter_checkpoints(self, checkpoints):
        max_checkpoint = max([get_epoch(c) for c in checkpoints])
        aux = max_checkpoint // self.eval_epochs
        return [c for c in checkpoints if get_epoch(c) % aux == 0]


def get_epoch(ckpt_path):
    match = re.search(r'epoch_(\d+)', ckpt_path)
    return int(match.group(1)) if match else "last"