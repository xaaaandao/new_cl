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
    
    resize_size: int = 112
    batch_size: int = 100
    num_workers: int = 4
    
    mean: Tuple[float, ...] = (0.894, 0.885, 0.870)
    std: Tuple[float, ...] = (0.236, 0.256, 0.285)
    
    data_config_str: str = field(init=False) 

    def __post_init__(self):
        self.data_config_str = f"IMG[{self.resize_size}]_B[{self.batch_size}]"

@dataclass
class ModelConfig:
    name: str = 'resnet50'
    method: str = 'SupCon'
    temp: float = 0.07
    feat_dim: int = 128
    model_config_str: str = field(init=False) 

    def __post_init__(self):
        self.model_config_str = f"R[{self.name}]_M[{self.method}]"

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
    train_config_str: str = field(init=False) 

    def __post_init__(self):
        self.train_config_str = f"E[{self.epochs}]"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_str: str = field(init=False) 

    def __post_init__(self):
        self.out_str = f"{self.data.data_config_str}_{self.model.model_config_str}_{self.train.train_config_str}"
        full_path = os.path.join(self.train.checkpoint_dir, self.out_str)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)