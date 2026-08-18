import torch
import torch.nn as nn
from typing import Optional

from torch import Tensor, device


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Também suporta a loss não supervisionada do SimCLR.

    Versão Hierárquica (Gênero → Espécie):
        - Par positivo se e somente se GÊNERO == GÊNERO E ESPÉCIE == ESPÉCIE.
        - Passos:
            1) Compara gênero com gênero. Se diferentes, NÃO é par positivo.
            2) Se gêneros iguais, compara espécie com espécie.
               Se espécies iguais, é par positivo.
               Se espécies diferentes, NÃO é par positivo.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1: torch.Tensor,
                features2: torch.Tensor,
                genus_label: Optional[torch.Tensor] = None,
                species_label: Optional[torch.Tensor] = None,
                mask1: Optional[torch.Tensor] = None,
                mask2: Optional[torch.Tensor] = None,
                loss_weight: float = 0.3) -> torch.Tensor:
        """
        Calcula a loss para um batch de features.

        Args:
            features1: Vetor oculto de shape [bsz, n_views, ...].
            species_label: Ground truth de shape [bsz].
            mask1: Máscara de contraste de shape [bsz, bsz], onde mask_{i,j}=1 se a amostra j tem a mesma classe que a amostra i. Pode ser assimétrica.
        Returns:
            Um escalar (loss).
        """
        # Detecção automática do dispositivo baseada nos dados de entrada
        device1 = features1.device
        device2 = features2.device

        if len(features1.shape) < 3 or len(features2.shape) < 3:
            raise ValueError('`features` precisa ser [bsz, n_views, ...], pelo menos 3 dimensões são necessárias')

        # Achatar dimensões extras se houver (ex: [bsz, n_views, feat_dim])
        if len(features1.shape) > 3 or len(features2.shape) < 3:
            features1 = features1.view(features1.shape[0], features1.shape[1], -1)

        batch_size1 = features1.shape[0]
        batch_size2 = features2.shape[0]

        # Criação da Máscara
        if species_label is not None and mask1 is not None:
            raise ValueError('Não é possível definir `species_label` e `mask` simultaneamente')
        elif species_label is None and mask1 is None:
            # Caso SimCLR (sem species_label): Máscara é identidade (cada imagem é par positivo dela mesma apenas)
            mask1 = torch.eye(batch_size1, dtype=torch.float32).to(device1)
            mask2 = torch.eye(batch_size2, dtype=torch.float32).to(device2)
        elif species_label is not None and genus_label is not None:
            # Caso SupCon: Cria máscara baseada nas classes iguais
            species_label = species_label.contiguous().view(-1, 1)
            if species_label.shape[0] != batch_size1:
                raise ValueError('Número de species_label não corresponde ao número de features')
            mask1 = torch.eq(species_label, species_label.T).float().to(device1)
            genus_label = genus_label.contiguous().view(-1, 1)
            if genus_label.shape[0] != batch_size2:
                raise ValueError('Número de genus_label não corresponde ao número de features')
            mask2 = torch.eq(genus_label, genus_label.T).float().to(device2)
        else:
            mask1 = mask1.float().to(device1)
            mask2 = mask2.float().to(device2)

        loss_species = self.calculate_loss(batch_size1, device1, features1, mask1)
        loss_genus = self.calculate_loss(batch_size2, device2, features2, mask2)

        return loss_genus * loss_weight + loss_species, loss_species.item(), loss_genus.item()

    def calculate_loss(self, batch_size: int, device: device, features: Tensor, mask: Tensor) -> Tensor:
        contrast_count = features.shape[1]

        # Desempacota as features de todas as views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Modo desconhecido: {}'.format(self.contrast_mode))

        # Computa logits (produto escalar / temperatura)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # Estabilidade Numérica (Log-Sum-Exp trick)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expansão da máscara para cobrir todas as views
        mask = mask.repeat(anchor_count, contrast_count)

        # Mascarar os casos de auto-contraste (a imagem comparada com ela mesma)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # Computa log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Computa a média da log-likelihood sobre os pares positivos
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss final
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss