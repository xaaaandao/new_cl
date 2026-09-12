import os
import sys
import time
import math
import logging
from typing import Any

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # backend sem interface gráfica (seguro para treino em servidor)
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

from config import Config
from losses import SupConLoss
from network import MultiHeadSupConResNet
from utils import AverageMeter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SupConTrainer:
    def __init__(self, config: Config, train_loader, use_pretrained):
        self.cfg = config
        self.loader = train_loader
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.train.checkpoint_dir, 'logs'))

        self.model = MultiHeadSupConResNet(
            name=self.cfg.model.name,
            feat_dim_genus=self.cfg.model.feat_dim_genus,
            feat_dim_species=self.cfg.model.feat_dim_species,
            use_pretrained=use_pretrained
        )
        # Mesma SupConLoss é reutilizada para as duas cabeças (é stateless em relação aos rótulos)
        self.criterion = SupConLoss(temperature=self.cfg.model.temp).to(self.cfg.train.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.cfg.train.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            momentum=self.cfg.train.momentum,
            weight_decay=self.cfg.train.weight_decay
        )

    def adjust_learning_rate(self, epoch):
        lr = self.cfg.train.learning_rate

        if self.cfg.train.cosine_annealing:
            eta_min = lr * (self.cfg.train.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / self.cfg.train.epochs)) / 2
        else:
            steps = sum([epoch > e for e in self.cfg.train.lr_decay_epochs])
            if steps > 0:
                lr = lr * (self.cfg.train.lr_decay_rate ** steps)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def warmup_learning_rate(self, epoch, batch_id, total_batches):
        if self.cfg.train.warmup and epoch <= self.cfg.train.warmup_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / (self.cfg.train.warmup_epochs * total_batches)
            lr = self.cfg.train.learning_rate * p

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, epoch, collect_tsne: bool = False):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        losses_genus = AverageMeter()
        losses_species = AverageMeter()
        end = time.time()

        # Acumulador de embeddings para o t-SNE (só é usado em epochs de checkpoint)
        tsne_data = None
        tsne_data2 = None
        if collect_tsne:
            tsne_data = {
                'feats': [],
                'genus_labels': [],
                'species_labels': []
            }
            tsne_data2 = {
                'genus_feats': [], 'species_feats': [],
                'genus_labels': [], 'species_labels': []
            }

        for idx, (images, genus_labels, species_labels) in enumerate(self.loader):
            # Lógica de Warmup
            self.warmup_learning_rate(epoch, idx, len(self.loader))

            # Preparação dos dados
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.to(self.cfg.train.device, non_blocking=True)
                genus_labels = genus_labels.to(self.cfg.train.device, non_blocking=True)
                species_labels = species_labels.to(self.cfg.train.device, non_blocking=True)

            bsz = genus_labels.shape[0]

            # Forward -> dicionário com as duas projeções normalizadas
            feat, out = self.model(collect_tsne, images)

            if collect_tsne:
                # feat = split_views(bsz, feat)
                tsne_data['feats'].append(feat[:bsz].detach().cpu().numpy())
                tsne_data['genus_labels'].append(genus_labels.detach().cpu().numpy())
                tsne_data['species_labels'].append(species_labels.detach().cpu().numpy())

            features_genus = split_views(bsz, out['genus'])
            features_species = split_views(bsz, out['species'])

            # --- Coleta dos embeddings para o t-SNE ---
            # Feito ANTES da função de perda: usa os mesmos embeddings que
            # alimentam a loss, pegando só a 1ª view (índices [0:bsz]) para não
            # duplicar pontos referentes à mesma imagem original.
            if collect_tsne:
                tsne_data2['genus_feats'].append(out['genus'][:bsz].detach().cpu().numpy())
                tsne_data2['species_feats'].append(out['species'][:bsz].detach().cpu().numpy())
                tsne_data2['genus_labels'].append(genus_labels.detach().cpu().numpy())
                tsne_data2['species_labels'].append(species_labels.detach().cpu().numpy())

            # Loss em cada nível de granularidade
            loss_genus = self.criterion(features_genus, genus_labels)
            loss_species = self.criterion(features_species, species_labels)

            loss = (self.cfg.model.loss_weight_genus * loss_genus
                    + self.cfg.model.loss_weight_species * loss_species)

            losses.update(loss.item(), bsz)
            losses_genus.update(loss_genus.item(), bsz)
            losses_species.update(loss_species.item(), bsz)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Métricas
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % self.cfg.train.print_freq == 0:
                logger.info(f'Train: [{epoch}/{self.cfg.train.epochs}][{idx + 1}/{len(self.loader)}] '
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            f'Loss {losses.val:.3f} ({losses.avg:.3f}) '
                            f'[Genus {losses_genus.val:.3f} ({losses_genus.avg:.3f}) | '
                            f'Species {losses_species.val:.3f} ({losses_species.avg:.3f})]')

        return losses.avg, losses_genus.avg, losses_species.avg, tsne_data, tsne_data2

    def save_features(self, tsne_data: dict, epoch: int):
        """
        Gera e salva o t-SNE (nível gênero e nível espécie) a partir dos
        embeddings acumulados durante o train_epoch desta epoch de checkpoint.
        Salvo na MESMA pasta do checkpoint (.pth) deste epoch.
        Cada cor é identificada na legenda pelo NOME real da classe.
        """
        if len(tsne_data.keys()) == 4:
            self.save_features_projection_head(epoch, tsne_data)
        else:
            self.save_features_backbone(epoch, tsne_data)

    def save_features_backbone(self, epoch: int, tsne_data: dict[Any, Any]):
        X = np.concatenate(tsne_data['feats'], axis=0)
        y_genus = np.concatenate(tsne_data['genus_labels'], axis=0)
        y_species = np.concatenate(tsne_data['species_labels'], axis=0)

        ckpt_dir = os.path.join(self.cfg.train.checkpoint_dir, 'features', 'backbone')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Nomes reais das classes, vindos do dataset (GenusSpeciesImageFolder)
        dataset = self.loader.dataset
        idx_to_genus_name = getattr(dataset, 'idx_to_genus_name', None)
        idx_to_species_name = {idx: name for name, idx in dataset.class_to_idx.items()}

        filename = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}')
        np.savez(filename, X=X, y_genus=y_genus, y_species=y_species, idx_to_genus_name=idx_to_genus_name, idx_to_species_name=idx_to_species_name)

    def save_features_projection_head(self, epoch: int, tsne_data: dict[Any, Any]):
        X_genus = np.concatenate(tsne_data['genus_feats'], axis=0)
        X_species = np.concatenate(tsne_data['species_feats'], axis=0)
        y_genus = np.concatenate(tsne_data['genus_labels'], axis=0)
        y_species = np.concatenate(tsne_data['species_labels'], axis=0)

        ckpt_dir = os.path.join(self.cfg.train.checkpoint_dir, 'features', 'projection_head')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Nomes reais das classes, vindos do dataset (GenusSpeciesImageFolder)
        dataset = self.loader.dataset
        idx_to_genus_name = getattr(dataset, 'idx_to_genus_name', None)
        idx_to_species_name = {idx: name for name, idx in dataset.class_to_idx.items()}

        filename = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}_genus')
        np.savez(filename, X=X_genus, y=y_genus, idx_to_name=idx_to_genus_name)
        filename = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}_species')
        np.savez(filename, X=X_species, y=y_species, idx_to_name=idx_to_species_name)

    def run(self):
        logger.info(f"Iniciando treinamento no dispositivo: {self.cfg.train.device}")

        for epoch in range(1, self.cfg.train.epochs + 1):
            self.adjust_learning_rate(epoch)

            # Esta epoch vai gerar checkpoint? Se sim, também coletamos embeddings para o t-SNE.
            is_checkpoint_epoch = (epoch % self.cfg.train.save_freq == 0
                                   or epoch == self.cfg.train.epochs)

            time_start = time.time()
            loss, loss_genus, loss_species, tsne_data, tsne_data2 = self.train_epoch(
                epoch, collect_tsne=is_checkpoint_epoch
            )

            logger.info(f'Epoch {epoch} finalizada. Loss total: {loss:.4f} '
                        f'(Genus: {loss_genus:.4f} | Species: {loss_species:.4f}). '
                        f'Tempo: {time.time() - time_start:.2f}s')
            self.writer.add_scalar('loss/total', loss, epoch)
            self.writer.add_scalar('loss/genus', loss_genus, epoch)
            self.writer.add_scalar('loss/species', loss_species, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.save_log(epoch, loss, loss_genus, loss_species, time.time() - time_start)

            # Save Checkpoint + t-SNE (juntos, na mesma pasta)
            if is_checkpoint_epoch:
                self.save_model(epoch)
                if tsne_data2 is not None:
                    self.save_features(tsne_data2, epoch)
                if tsne_data is not None:
                    self.save_features(tsne_data, epoch)

    def save_model(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': asdict(self.cfg)
        }

        full_path = os.path.join(self.cfg.train.checkpoint_dir, 'checkpoints')

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        path = os.path.join(full_path, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, path)
        logger.info(f"Modelo salvo em: {path}")

    def save_log(self, epoch, loss, loss_genus, loss_species, time_elapsed):
        data = {
            'epoch': epoch,
            'loss': loss,
            'loss_genus': loss_genus,
            'loss_species': loss_species,
            'time': time_elapsed
        }
        df = pd.DataFrame([data])

        csv_out_path = os.path.join(self.cfg.get_checkpoint_dir(), 'results', 'loss.csv')
        header = not os.path.exists(csv_out_path)
        df.to_csv(csv_out_path, mode='a', header=header, index=False)


def split_views(bsz, features):
    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    return torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)