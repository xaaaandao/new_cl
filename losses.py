import torch
import torch.nn as nn
from typing import Optional

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Também suporta a loss não supervisionada do SimCLR.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcula a loss para um batch de features.

        Args:
            features: Vetor oculto de shape [bsz, n_views, ...].
            labels: Ground truth de shape [bsz].
            mask: Máscara de contraste de shape [bsz, bsz], onde mask_{i,j}=1 se a amostra j tem a mesma classe que a amostra i. Pode ser assimétrica.
        Returns:
            Um escalar (loss).
        """
        # Detecção automática do dispositivo baseada nos dados de entrada
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` precisa ser [bsz, n_views, ...], pelo menos 3 dimensões são necessárias')
        
        # Achatar dimensões extras se houver (ex: [bsz, n_views, feat_dim])
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Criação da Máscara
        if labels is not None and mask is not None:
            raise ValueError('Não é possível definir `labels` e `mask` simultaneamente')
        elif labels is None and mask is None:
            # Caso SimCLR (sem labels): Máscara é identidade (cada imagem é par positivo dela mesma apenas)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Caso SupCon: Cria máscara baseada nas classes iguais
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Número de labels não corresponde ao número de features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        
        # Desempacota as features de todas as views
        # Se temos [Batch, 2, Dim], vira [Batch*2, Dim]
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
        # Matmul: [Anchor_Count, Dim] x [Contrast_Count, Dim]^T -> [Anchor_Count, Contrast_Count]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # Estabilidade Numérica (Log-Sum-Exp trick)
        # Subtrair o máximo melhora a estabilidade numérica da exponencial
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expansão da máscara para cobrir todas as views
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mascarar os casos de auto-contraste (a imagem comparada com ela mesma)
        # A diagonal principal e as diagonais deslocadas (devido às views) não devem contar
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        
        mask = mask * logits_mask

        # Computa log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # Soma das exponenciais (denominador do Softmax) apenas para os negativos e positivos válidos
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6) # +1e-6 para evitar log(0)

        # Computa a média da log-likelihood sobre os pares positivos
        mask_pos_pairs = mask.sum(1)
        
        # Evitar divisão por zero caso não haja pares positivos
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.ones_like(mask_pos_pairs), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss final
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss