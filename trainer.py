import os
import sys
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

from config import Config
from losses import SupConLoss
from network import SupConResNet
from utils import AverageMeter, accuracy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupConTrainer:
    def __init__(self, config: Config, train_loader, use_pretrained):
        self.cfg = config
        self.loader = train_loader
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.train.checkpoint_dir, 'logs'))
        
        self.model = SupConResNet(name=self.cfg.model.name, feat_dim=self.cfg.model.feat_dim, use_pretrained=use_pretrained)
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

        self.feats_backbone = {
            'feats': [],
            'labels': []
        }
        self.feats_projection_head = {
            'feats': [],
            'labels': []
        }

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

    def train_epoch(self, epoch):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for idx, (images, labels) in enumerate(self.loader):
            # Lógica de Warmup
            self.warmup_learning_rate(epoch, idx, len(self.loader))

            # Preparação dos dados
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.to(self.cfg.train.device, non_blocking=True)
                labels = labels.to(self.cfg.train.device, non_blocking=True)
            
            bsz = labels.shape[0]
            # Forward
            features, out = self.model(images)

            self.set_feats_backbone(bsz, features, labels)

            f1, f2 = torch.split(out, [bsz, bsz], dim=0)
            out = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            self.set_feats_projection_head(bsz, out, labels)

            # Loss
            loss = self.criterion(out, labels)
            losses.update(loss.item(), bsz)

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
                            f'Loss {losses.val:.3f} ({losses.avg:.3f})')

        return losses.avg

    def run(self):
        logger.info(f"Iniciando treinamento no dispositivo: {self.cfg.train.device}")
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            self.adjust_learning_rate(epoch)
            
            time_start = time.time()
            loss = self.train_epoch(epoch)
            
            logger.info(f'Epoch {epoch} finalizada. Loss média: {loss:.4f}. Tempo: {time.time() - time_start:.2f}s')
            self.writer.add_scalar('loss', loss, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.save_log(epoch, loss, time.time() - time_start)

            # Save Checkpoint
            if epoch % self.cfg.train.save_freq == 0 or epoch == self.cfg.train.epochs:
                self.save_model(epoch)
                self.save_feats(epoch, self.feats_backbone)
                self.save_feats(epoch, self.feats_projection_head, is_backbone=False)

    def save_feats(self, epoch: int, features, is_backbone=True):
        X = np.concatenate(features['feats'], axis=0)
        y = np.concatenate(features['labels'], axis=0)

        ckpt_dir = os.path.join(self.cfg.train.checkpoint_dir, 'features', 'backbone') if is_backbone else os.path.join(self.cfg.train.checkpoint_dir, 'features', 'projection_head')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Nomes reais das classes, vindos do dataset (GenusSpeciesImageFolder)
        dataset = self.loader.dataset
        idx_to_label = getattr(dataset, 'class_to_idx', None)

        filename = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}')
        np.savez(filename, X=X, y=y, idx_to_label=idx_to_label)

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
        
    def save_log(self, epoch, loss, time_elapsed):
        data = {
            'epoch': epoch,
            'loss': loss,
            'time': time_elapsed
        }
        df = pd.DataFrame([data])
        
        csv_out_path = os.path.join(self.cfg.get_checkpoint_dir(), 'results', 'loss.csv')
        header = not os.path.exists(csv_out_path)
        df.to_csv(csv_out_path, mode='a', header=header, index=False)

    def set_feats_backbone(self, bsz, features, labels):
        self.feats_backbone['feats'].append(features[:bsz].detach().cpu().numpy())
        self.feats_backbone['labels'].append(labels.detach().cpu().numpy())

    def set_feats_projection_head(self, bsz, features, labels):
        self.feats_projection_head['feats'].append(features[:bsz].detach().cpu().numpy())
        self.feats_projection_head['labels'].append(labels.detach().cpu().numpy())