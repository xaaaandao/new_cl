import os
import torch
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class DataConfig:
    dataset_name: str = 'pr_dataset'
    data_path: str = "data"
    
    image_size: int = 512
    color_space: str = 'rgb'
    
    resize_size: int = 224
    
    batch_size: int = 30
    num_workers: int = 4
    
    mean: Tuple[float, ...] = (0.894, 0.885, 0.870)
    std: Tuple[float, ...] = (0.236, 0.256, 0.285)

@dataclass
class ModelConfig:
    name: str = 'resnet50'
    method: str = 'SupCon'
    temp: float = 0.07
    feat_dim: int = 128

@dataclass
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    lr_decay_epochs: List[int] = field(default_factory=lambda: [100, 200, 300])
    lr_decay_rate: float = 0.1
    
    cosine_annealing: bool = True
    warmup: bool = True
    warmup_epochs: int = 10
    
    print_freq: int = 1
    save_freq: int = 50
    checkpoint_dir: str = "saved_models"
    
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)