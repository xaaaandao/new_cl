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
    batch_size: int = 128
    num_workers: int = 4
    
    mean: Tuple[float, ...] = None
    std: Tuple[float, ...] = None
    
    data_config_str: str = field(init=False) 

    def __post_init__(self):
        self.data_config_str = f"{self.dataset_name}_IMG[{self.resize_size}]_B[{self.batch_size}]"

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
    epochs: int = 50
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
class EvalConfig:
    svm_c: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    svm_kernel: List[str] = field(default_factory=lambda: ['linear', 'rbf'])
    n_jobs: int = -1
    
    batch_size_inference: int = 64

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def name(self) -> str:
        return (
            f"{self.data.dataset_name}_"
            f"IMG[{self.data.resize_size}]_"
            f"B[{self.data.batch_size}]_"
            f"E[{self.train.epochs}]_"
            f"{self.model.name}_"
            f"{self.model.method}"
        )

    def get_checkpoint_dir(self) -> str:
        """Retorna o caminho completo onde o modelo será salvo."""
        return os.path.join("saved_models", self.name)