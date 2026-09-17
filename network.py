import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Literal, Dict, Optional

class SupConResNet(nn.Module):
    """
    Backbone (ResNet) + Projection Head para Contrastive Learning.
    Utiliza implementação oficial do torchvision (ImageNet-style).
    """
    
    # Mapeamento para saber a dimensão de saída de cada backbone
    BACKBONE_DIM_DICT = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
    }

    def __init__(self, name: str = 'resnet50', 
                head: Literal['linear', 'mlp'] = 'mlp', 
                feat_dim: int = 128,
                use_pretrained: bool = False):
        """
        Args:
            name: Nome da arquitetura (resnet18, resnet50, etc).
            head: Tipo de projection head ('linear' ou 'mlp').
            feat_dim: Dimensão do espaço latente para projeção (ex: 128).
            use_pretrained: Se True, inicia com pesos da ImageNet (recomendado).
        """
        super(SupConResNet, self).__init__()
        
        # 1. Carregar Backbone (Encoder)
        self.encoder = self._get_backbone(name, use_pretrained)
        
        # Descobrir dimensão de entrada da head (in_features da última layer original)
        dim_in = self.BACKBONE_DIM_DICT.get(name)
        
        if dim_in is None:
            raise ValueError(f"Modelo {name} não suportado ou não mapeado.")

        # 2. Construir Projection Head
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f'Head não suportada: {head}')

    def _get_backbone(self, name: str, pretrained: bool) -> nn.Module:
        """Carrega o modelo do torchvision e remove a camada de classificação (fc)."""
        try:
            # Carrega o modelo base (ex: models.resnet50(weights=...))
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            
            # Dinamicamente pega a função construtora do módulo models
            # Ex: models.resnet50
            model_fn = getattr(models, name)
            
            # Tratamento para versões novas do Torchvision
            try:
                model = model_fn(weights=weights) if pretrained else model_fn(weights=None)
            except TypeError:
                model = model_fn(pretrained=pretrained)

            # Remove a última camada (Fully Connected) original da ResNet
            # Substituímos por Identity para que o encoder retorne apenas as features puras
            model.fc = nn.Identity()
            
            return model
            
        except AttributeError:
            raise ValueError(f"Arquitetura {name} não encontrada no torchvision.models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extração de Features
        feat = self.encoder(x)
        
        # Projeção
        feat = self.head(feat)
        
        # Normalização (Crucial para Contrastive Loss operar na hiperesfera)
        feat = F.normalize(feat, dim=1)
        
        return feat

class LinearClassifier(nn.Module):
    """
    Classificador Linear simples para a etapa de avaliação (Linear Probing).
    Geralmente treinado com o Encoder congelado.
    """
    def __init__(self, name: str = 'resnet50', num_classes: int = 10):
        super(LinearClassifier, self).__init__()
        dim_in = SupConResNet.BACKBONE_DIM_DICT[name]
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class MultiHeadSupConResNet(nn.Module):
    """
    Backbone (ResNet) COMPARTILHADO + DUAS Projection Heads independentes:
    uma para o nível de GÊNERO e outra para o nível de ESPÉCIE.

    A ideia é a mesma do SupConResNet, mas em vez de uma única cabeça de
    projeção, existem duas, cada uma aprendendo seu próprio espaço latente
    hiperesférico. Isso permite treinar contrastive learning simultaneamente
    em dois níveis de granularidade taxonômica, usando a mesma SupConLoss
    duas vezes (uma por rótulo de gênero, outra por rótulo de espécie).
    """

    BACKBONE_DIM_DICT = SupConResNet.BACKBONE_DIM_DICT

    def __init__(self, name: str = 'resnet50',
                 head: Literal['linear', 'mlp'] = 'mlp',
                 feat_dim_genus: int = 128,
                 feat_dim_species: int = 128,
                 use_pretrained: bool = False):
        """
        Args:
            name: Nome da arquitetura (resnet18, resnet50, etc).
            head: Tipo de projection head ('linear' ou 'mlp'), usado para as duas cabeças.
            feat_dim_genus: Dimensão do espaço latente da cabeça de GÊNERO.
            feat_dim_species: Dimensão do espaço latente da cabeça de ESPÉCIE.
            use_pretrained: Se True, inicia com pesos da ImageNet (recomendado).
        """
        super(MultiHeadSupConResNet, self).__init__()

        # 1. Backbone compartilhado (reaproveita o mesmo carregador do SupConResNet)
        self.encoder = SupConResNet._get_backbone(self, name, use_pretrained)

        dim_in = self.BACKBONE_DIM_DICT.get(name)
        if dim_in is None:
            raise ValueError(f"Modelo {name} não suportado ou não mapeado.")

        # 2. Duas Projection Heads independentes, partindo do mesmo dim_in
        self.head_genus = self._build_head(head, dim_in, feat_dim_genus)
        self.head_species = self._build_head(head, dim_in, feat_dim_species)

    @staticmethod
    def _build_head(head: str, dim_in: int, feat_dim: int) -> nn.Module:
        if head == 'linear':
            return nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f'Head não suportada: {head}')

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extração de Features (backbone compartilhado)
        feat = self.encoder(x)

        # Projeção + normalização para cada nível
        feat_genus = F.normalize(self.head_genus(feat), dim=1)
        feat_species = F.normalize(self.head_species(feat), dim=1)

        return feat, {'genus': feat_genus, 'species': feat_species}